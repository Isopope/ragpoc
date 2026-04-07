"""Registre des agents disponibles pour le mode battle royale."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class AgentSpec:
    """Description d'un backend agentique comparable."""

    key: str
    label: str
    description: str
    factory: Callable[..., Any]


def _build_rag_chain(**kwargs):
    from rag_chain import RAGChain

    return RAGChain(
        kwargs["weaviate_store"],
        kwargs["openai_key"],
        embedding_model=kwargs["embedding_model"],
        llm_model=kwargs["llm_model"],
        top_k_final=kwargs["top_k_final"],
        hybrid_alpha=kwargs["hybrid_alpha"],
        max_tokens=kwargs["max_tokens"],
    )


def _build_rag_pipeline(**kwargs):
    from rag_pipeline import RAGAgent

    return RAGAgent(
        kwargs["weaviate_store"],
        kwargs["openai_key"],
        cohere_key=kwargs.get("cohere_key"),
        embedding_model=kwargs["embedding_model"],
        llm_model=kwargs["llm_model"],
        top_k_retrieve=kwargs["top_k_retrieve"],
        top_k_final=kwargs["top_k_final"],
        hybrid_alpha=kwargs["hybrid_alpha"],
        max_tokens=kwargs["max_tokens"],
        max_agent_iter=kwargs["max_agent_iter"],
        llm_timeout=kwargs["llm_timeout"],
    )


def _build_unified_rag_agent(**kwargs):
    from rag_agent import RAGAgent

    return RAGAgent(
        kwargs["weaviate_store"],
        kwargs["openai_key"],
        cohere_key=kwargs.get("cohere_key"),
        embedding_model=kwargs["embedding_model"],
        llm_model=kwargs["llm_model"],
        top_k_retrieve=kwargs["top_k_retrieve"],
        top_k_final=kwargs["top_k_final"],
        hybrid_alpha=kwargs["hybrid_alpha"],
        max_tokens=kwargs["max_tokens"],
        max_agent_iter=kwargs["max_agent_iter"],
        llm_timeout=kwargs["llm_timeout"],
    )


def _build_elysia(**kwargs):
    from langgraph_implementation.rag_agent import ElysiaRAGAgent

    return ElysiaRAGAgent(
        kwargs["weaviate_store"],
        kwargs["openai_key"],
        llm_model=kwargs["llm_model"],
        embedding_model=kwargs["embedding_model"],
    )


def available_agents() -> list[AgentSpec]:
    """Retourne les agents exposés au runner."""
    return [
        AgentSpec(
            key="rag_chain",
            label="RAG Chain",
            description="Baseline simple: retrieve + generate sans boucle agentique.",
            factory=_build_rag_chain,
        ),
        AgentSpec(
            key="rag_pipeline",
            label="RAG Pipeline",
            description="Backend historique monolithique base sur LangGraph.",
            factory=_build_rag_pipeline,
        ),
        AgentSpec(
            key="rag_agent",
            label="Unified RAG Agent",
            description="Backend modulaire unifie, cible principale du projet.",
            factory=_build_unified_rag_agent,
        ),
        AgentSpec(
            key="elysia",
            label="Elysia LangGraph",
            description="Implementation experimentale orientee decision tree.",
            factory=_build_elysia,
        ),
    ]
