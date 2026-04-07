"""Constructeur du graphe LangGraph unifié et classe RAGAgent.

Interface publique compatible avec rag_pipeline.RAGAgent :
  - RAGAgent(weaviate_store, openai_key, cohere_key, ...)
  - agent.stream_query(question, source, conversation_summary) → generator
  - agent.query(question, source) → dict
"""
from __future__ import annotations

import functools
from typing import Any, Optional

from loguru import logger


def build_unified_graph(config, weaviate_store, cohere_client=None):
    """Compile et retourne le graphe LangGraph RAG unifié.

    Args:
        config:          RAGConfig — paramétrage complet du pipeline.
        weaviate_store:  Instance WeaviateStore (None → mode mock).
        cohere_client:   Client Cohere (None → fallback LLM reranking).
    """
    from functools import partial

    from openai import OpenAI
    from langgraph.graph import END, START, StateGraph

    from .config import RAGConfig
    from .llm import make_llm_caller, make_embedder
    from .state import UnifiedRAGState
    from .tools.query import QueryTool
    from .nodes.planning import analyze_and_plan
    from .nodes.reasoning import agent_reason, agent_action, consolidate_chunks, route_agent, route_after_action
    from .nodes.compression import compress_context
    from .nodes.reranking import rerank
    from .nodes.generation import generate, generate_follow_up, generate_title

    client   = OpenAI(api_key=config.openai_key)
    llm_call = make_llm_caller(client, config.llm_model, config.llm_timeout)
    embedder = make_embedder(client, config.embedding_model, config.llm_timeout)
    query_tool = QueryTool(weaviate_store, embedder)

    # Closure : max_agent_iter injecté dans route_agent via state
    _max_iter = config.max_agent_iter

    def _route_agent(state):
        s = dict(state)
        s["_max_agent_iter"] = _max_iter
        return route_agent(s)

    def _route_after_action(state):
        return route_after_action(state, rag_config=config)

    # Nodes avec injection via partial
    _analyze    = partial(analyze_and_plan,    llm_call=llm_call,    rag_config=config)
    _reason     = partial(agent_reason,        llm_call=llm_call,    rag_config=config)
    _action     = partial(agent_action,        query_tool=query_tool, rag_config=config, weaviate_store=weaviate_store)
    _compress   = partial(compress_context,    llm_call=llm_call,    rag_config=config)
    _consolidate= partial(consolidate_chunks,  query_tool=query_tool, rag_config=config)
    _rerank     = partial(rerank,              llm_call=llm_call,    cohere_client=cohere_client, rag_config=config)
    _generate   = partial(generate,            llm_call=llm_call,    rag_config=config)
    _follow_up  = partial(generate_follow_up,  llm_call=llm_call,    rag_config=config)
    _title      = partial(generate_title,      llm_call=llm_call,    rag_config=config)

    builder = StateGraph(UnifiedRAGState)
    builder.add_node("analyze_and_plan",   _analyze)
    builder.add_node("agent_reason",       _reason)
    builder.add_node("agent_action",       _action)
    builder.add_node("compress_context",   _compress)
    builder.add_node("consolidate",        _consolidate)
    builder.add_node("rerank",             _rerank)
    builder.add_node("generate",           _generate)
    builder.add_node("generate_follow_up", _follow_up)
    builder.add_node("generate_title",     _title)

    # Edges fixes
    builder.add_edge(START,                "analyze_and_plan")
    builder.add_edge("analyze_and_plan",   "agent_reason")
    builder.add_edge("compress_context",   "agent_reason")
    builder.add_edge("consolidate",        "rerank")
    builder.add_edge("rerank",             "generate")
    builder.add_edge("generate",           "generate_follow_up")
    builder.add_edge("generate_follow_up", "generate_title")
    builder.add_edge("generate_title",     END)

    # Edges conditionnels
    builder.add_conditional_edges(
        "agent_reason",
        _route_agent,
        {"agent_action": "agent_action", "rerank_prep": "consolidate"},
    )
    builder.add_conditional_edges(
        "agent_action",
        _route_after_action,
        {"compress_context": "compress_context", "agent_reason": "agent_reason"},
    )

    return builder.compile()


# ── Classe wrapper ─────────────────────────────────────────────────────────────

class RAGAgent:
    """Agent RAG unifié — interface compatible avec rag_pipeline.RAGAgent.

    Usage :
        agent = RAGAgent(weaviate_store, openai_key, cohere_key)
        for event in agent.stream_query(question):
            ...
        result = agent.query(question)
    """

    def __init__(
        self,
        weaviate_store,
        openai_key: str,
        cohere_key: Optional[str] = None,
        *,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4.1",
        top_k_retrieve: int = 20,
        top_k_final: int = 5,
        hybrid_alpha: float = 0.5,
        max_tokens: int = 4000,
        max_agent_iter: int = 60,
        llm_timeout: float = 30.0,
        enable_compression: bool = False,
    ) -> None:
        from .config import RAGConfig

        self._config = RAGConfig(
            openai_key      = openai_key,
            llm_model       = llm_model,
            embedding_model = embedding_model,
            cohere_key      = cohere_key,
            top_k_retrieve  = top_k_retrieve,
            top_k_final     = top_k_final,
            hybrid_alpha    = hybrid_alpha,
            max_tokens      = max_tokens,
            max_agent_iter  = max_agent_iter,
            llm_timeout     = llm_timeout,
            enable_compression = enable_compression,
        )
        self._store = weaviate_store

        cohere_client = _init_cohere(cohere_key)
        self._graph = build_unified_graph(self._config, weaviate_store, cohere_client)

    # ── API publique ───────────────────────────────────────────────────────────

    def stream_query(
        self,
        question: str,
        source: Optional[str] = None,
        conversation_summary: str = "",
    ):
        """Exécute le pipeline et yield les événements nœud par nœud.

        Compatible avec app.py : for event in agent.stream_query(...): ...
        """
        from .state import create_unified_state
        from .tools.query import weaviate_with_retry

        try:
            available_sources = weaviate_with_retry(self._store.list_sources)
        except Exception:
            available_sources = []

        initial_state = create_unified_state(
            question             = question,
            source               = source,
            conversation_summary = conversation_summary,
            available_sources    = available_sources,
            max_tree_depth       = self._config.max_tree_depth,
        )

        for event in self._graph.stream(initial_state, stream_mode="updates"):
            yield event

    def query(
        self,
        question: str,
        source: Optional[str] = None,
    ) -> dict:
        """Exécute le pipeline et retourne la réponse.

        Retourne un dict compatible avec rag_pipeline.RAGAgent.query() :
            - answer        (str)
            - sources       (list[dict])  chunks rerankés
            - question      (str)
            - question_id   (str)
            - n_retrieved   (int)
            - decision_log  (list[dict])
            - error         (str|None)
            # Champs supplémentaires du package unifié :
            - follow_up_suggestions (list[str])
            - conversation_title    (str|None)
        """
        from .state import create_unified_state
        from .tools.query import weaviate_with_retry

        try:
            available_sources = weaviate_with_retry(self._store.list_sources)
        except Exception:
            available_sources = []

        initial_state = create_unified_state(
            question          = question,
            source            = source,
            available_sources = available_sources,
            max_tree_depth    = self._config.max_tree_depth,
        )

        final = self._graph.invoke(initial_state)

        return {
            "answer":               final.get("answer",               ""),
            "sources":              final.get("reranked_docs",        []),
            "question":             question,
            "question_id":          final.get("question_id",          ""),
            "n_retrieved":          len(final.get("retrieved_docs",   [])),
            "decision_log":         final.get("decision_log",         []),
            "error":                final.get("error"),
            "follow_up_suggestions": final.get("follow_up_suggestions", []),
            "conversation_title":   final.get("conversation_title"),
        }


# ── Helpers privés ─────────────────────────────────────────────────────────────

def _init_cohere(cohere_key: Optional[str]) -> Optional[Any]:
    """Initialise le client Cohere si la clé est disponible."""
    if not cohere_key:
        return None
    try:
        import cohere
        client = cohere.Client(api_key=cohere_key)
        logger.info("Cohere Rerank activé")
        return client
    except ImportError:
        logger.warning("Package 'cohere' non installé — fallback reranking LLM")
        return None
