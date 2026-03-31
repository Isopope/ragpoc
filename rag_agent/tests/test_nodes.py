"""Tests des nœuds LangGraph (avec mocks LLM)."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rag_agent.config import RAGConfig
from rag_agent.state import create_unified_state
from rag_agent.tools.query import QueryTool


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return RAGConfig(openai_key="test-key", llm_timeout=5.0, use_follow_up=True, use_title_generation=True)


@pytest.fixture
def mock_query_tool():
    return QueryTool(weaviate_store=None, embedder=None)  # mode mock


def _make_llm_call(content: str = '{"target": null, "reason": "test", "sub_queries": ["q1"], "confidence": 0.9}'):
    """Retourne un callable LLM simulé."""
    def _call(messages, **kwargs):
        resp = MagicMock()
        resp.choices[0].message.content = content
        resp.choices[0].finish_reason = "stop"
        resp.choices[0].message.tool_calls = None
        return resp
    return _call


def _make_llm_call_with_tool(tool_name: str, tool_args: dict):
    """Retourne un callable LLM simulé qui émet des tool_calls."""
    def _call(messages, **kwargs):
        import json
        resp = MagicMock()
        tc   = MagicMock()
        tc.id = "call_123"
        tc.type = "function"
        tc.function.name      = tool_name
        tc.function.arguments = json.dumps(tool_args)
        resp.choices[0].message.content    = "Je vais faire une recherche."
        resp.choices[0].message.tool_calls = [tc]
        resp.choices[0].finish_reason      = "tool_calls"
        return resp
    return _call


# ── Nœud 1 : planning ─────────────────────────────────────────────────────────

def test_analyze_and_plan_basic(config):
    from rag_agent.nodes.planning import analyze_and_plan
    state  = create_unified_state("Quel est le budget 2024 ?", available_sources=["/docs/budget.pdf"])
    result = analyze_and_plan(state, llm_call=_make_llm_call(), config=config)
    assert "sub_queries" in result
    assert len(result["sub_queries"]) >= 1
    assert result["current_branch"] == "plan"
    assert "plan.analyze" in result["decision_history"]


def test_analyze_and_plan_target_resolution(config):
    from rag_agent.nodes.planning import analyze_and_plan
    json_content = '{"target": "budget.pdf", "reason": "PDF mentionné", "sub_queries": ["budget 2024"], "confidence": 0.95}'
    state        = create_unified_state("Parle-moi de budget.pdf", available_sources=["/docs/budget.pdf"])
    result       = analyze_and_plan(state, llm_call=_make_llm_call(json_content), config=config)
    assert result["source_filter"] == "/docs/budget.pdf"


def test_analyze_and_plan_llm_failure_fallback(config):
    from rag_agent.nodes.planning import analyze_and_plan
    def failing_llm(messages, **kwargs):
        raise TimeoutError("LLM timeout")
    state  = create_unified_state("Question de test")
    result = analyze_and_plan(state, llm_call=failing_llm, config=config)
    # Fallback : la question brute est utilisée comme unique sous-requête
    assert result["sub_queries"] == ["Question de test"]


def test_analyze_and_plan_pydantic_validation_error(config):
    """JSON malformé → fallback gracieux."""
    from rag_agent.nodes.planning import analyze_and_plan
    bad_llm = _make_llm_call('{"sub_queries": []}')  # liste vide → validation error PlanningOutput
    state   = create_unified_state("Q ?")
    result  = analyze_and_plan(state, llm_call=bad_llm, config=config)
    # Fallback attendu
    assert result["sub_queries"] == ["Q ?"]


# ── Nœud 2 : agent_reason ─────────────────────────────────────────────────────

def test_agent_reason_builds_initial_prompt(config):
    from rag_agent.nodes.reasoning import agent_reason
    state = create_unified_state("Quelle est la politique ?")
    state["sub_queries"] = ["politique 2024"]  # type: ignore[index]
    result = agent_reason(state, llm_call=_make_llm_call("RECHERCHE_TERMINEE"), config=config)
    assert "messages" in result
    assert len(result["messages"]) >= 2  # user + assistant


def test_agent_reason_tool_call_appended(config):
    from rag_agent.nodes.reasoning import agent_reason
    state  = create_unified_state("Q ?")
    llm    = _make_llm_call_with_tool("search_documents", {"query": "test"})
    result = agent_reason(state, llm_call=llm, config=config)
    last_msg = result["messages"][-1]
    assert last_msg["role"] == "assistant"
    assert last_msg.get("tool_calls") is not None


def test_agent_reason_iteration_counter(config):
    from rag_agent.nodes.reasoning import agent_reason
    state = create_unified_state("Q ?")
    state["agent_iterations"] = 3  # type: ignore[index]
    result = agent_reason(state, llm_call=_make_llm_call("RECHERCHE_TERMINEE"), config=config)
    assert result["agent_iterations"] == 4


# ── Nœud 3 : agent_action ─────────────────────────────────────────────────────

def test_agent_action_search_documents(config, mock_query_tool):
    from rag_agent.nodes.reasoning import agent_action
    import json
    state    = create_unified_state("Q ?")
    state["messages"] = [  # type: ignore[index]
        {"role": "user", "content": "Cherche..."},
        {
            "role": "assistant",
            "content": "Je cherche.",
            "tool_calls": [{
                "id": "c1",
                "type": "function",
                "function": {"name": "search_documents", "arguments": json.dumps({"query": "budget 2024"})},
            }],
        },
    ]
    result = agent_action(state, query_tool=mock_query_tool, config=config)
    assert len(result["all_docs"]) > 0
    assert len(result["seen_keys"]) > 0
    assert ("budget 2024", 1.0) in result["seen_queries"]


def test_agent_action_duplicate_query_skipped(config, mock_query_tool):
    from rag_agent.nodes.reasoning import agent_action
    import json
    state = create_unified_state("Q ?")
    state["seen_queries"] = [("budget 2024", 1.0)]  # type: ignore[index]
    state["messages"] = [  # type: ignore[index]
        {"role": "user", "content": "Cherche."},
        {
            "role": "assistant",
            "content": "Recherche.",
            "tool_calls": [{
                "id": "c1",
                "type": "function",
                "function": {"name": "search_documents", "arguments": json.dumps({"query": "budget 2024"})},
            }],
        },
    ]
    result = agent_action(state, query_tool=mock_query_tool, config=config)
    # Aucun nouveau doc ajouté
    assert result["all_docs"] == []
    # Le message tool contient "notice"
    tool_resp = result["messages"][-1]
    assert "notice" in tool_resp["content"].lower() or "doublon" in tool_resp["content"].lower() or "déjà" in tool_resp["content"].lower()


def test_agent_action_invalid_chunk_index(config, mock_query_tool):
    from rag_agent.nodes.reasoning import agent_action
    import json
    state = create_unified_state("Q ?")
    state["messages"] = [  # type: ignore[index]
        {"role": "user", "content": "Cherche."},
        {
            "role": "assistant",
            "content": "Get chunk.",
            "tool_calls": [{
                "id": "c2",
                "type": "function",
                "function": {"name": "get_neighboring_chunk", "arguments": json.dumps({"source_name": "doc.pdf", "chunk_index": -999})},
            }],
        },
    ]
    result = agent_action(state, query_tool=mock_query_tool, config=config)
    tool_resp = result["messages"][-1]
    assert "error" in tool_resp["content"].lower() or "invalide" in tool_resp["content"].lower()


# ── Nœud : compress_context ───────────────────────────────────────────────────

def test_compress_context_resets_messages(config):
    from rag_agent.nodes.compression import compress_context
    state    = create_unified_state("Q ?")
    state["all_docs"] = [{"source": "a.pdf", "chunk_index": 0, "page_content": "Content.", "kind": "text", "title_path": ""}]  # type: ignore[index]
    state["messages"] = [{"role": "user", "content": "msg"}]  # type: ignore[index]
    llm      = _make_llm_call("# Résumé compressé\n\nContenu synthétisé du document.")
    result   = compress_context(state, llm_call=llm, config=config)
    assert result["messages"] == []  # réinitialisé
    assert len(result["context_summary"]) > 0


def test_compress_context_llm_failure_keeps_existing(config):
    from rag_agent.nodes.compression import compress_context
    def failing_llm(messages, **kwargs):
        raise TimeoutError("timeout")
    state  = create_unified_state("Q ?")
    state["context_summary"] = "Résumé existant."  # type: ignore[index]
    result = compress_context(state, llm_call=failing_llm, config=config)
    assert result["context_summary"] == "Résumé existant."


# ── Nœud : consolidate_chunks ─────────────────────────────────────────────────

def test_consolidate_chunks_deduplicates(config, mock_query_tool):
    from rag_agent.nodes.reasoning import consolidate_chunks
    state = create_unified_state("Q ?")
    state["all_docs"] = [  # type: ignore[index]
        {"source": "a.pdf", "chunk_index": 1, "_score": 0.9, "page_content": "A"},
        {"source": "a.pdf", "chunk_index": 1, "_score": 0.7, "page_content": "A"},  # doublon
        {"source": "b.pdf", "chunk_index": 2, "_score": 0.8, "page_content": "B"},
    ]
    result = consolidate_chunks(state, query_tool=mock_query_tool, config=config)
    assert len(result["retrieved_docs"]) == 2


def test_consolidate_chunks_fallback_on_empty(config, mock_query_tool):
    from rag_agent.nodes.reasoning import consolidate_chunks
    state  = create_unified_state("Q ?")  # all_docs vide
    result = consolidate_chunks(state, query_tool=mock_query_tool, config=config)
    # Fallback → mock_query_tool retourne 3 docs
    assert len(result["retrieved_docs"]) > 0


# ── Nœud : rerank ─────────────────────────────────────────────────────────────

def test_rerank_llm_fallback(config):
    from rag_agent.nodes.reranking import rerank
    docs  = [
        {"source": "a.pdf", "chunk_index": 0, "page_content": "Doc A", "kind": "text", "title_path": ""},
        {"source": "b.pdf", "chunk_index": 1, "page_content": "Doc B", "kind": "text", "title_path": ""},
    ]
    state = create_unified_state("Q ?")
    state["retrieved_docs"] = docs  # type: ignore[index]
    llm   = _make_llm_call("[9, 3]")
    result = rerank(state, llm_call=llm, cohere_client=None, config=config)
    reranked = result["reranked_docs"]
    assert len(reranked) == 2
    assert reranked[0]["_rerank_score"] >= reranked[1]["_rerank_score"]


def test_rerank_malformed_llm_scores_fallback(config):
    from rag_agent.nodes.reranking import rerank
    docs  = [{"source": "a.pdf", "chunk_index": 0, "page_content": "X", "kind": "text", "title_path": ""}]
    state = create_unified_state("Q ?")
    state["retrieved_docs"] = docs  # type: ignore[index]
    llm   = _make_llm_call("Score: 7/10 et aussi 4/10")  # pas un JSON valide mais regex peut extraire
    result = rerank(state, llm_call=llm, cohere_client=None, config=config)
    assert len(result["reranked_docs"]) == 1


def test_rerank_no_docs(config):
    from rag_agent.nodes.reranking import rerank
    state = create_unified_state("Q ?")
    result = rerank(state, llm_call=_make_llm_call(), cohere_client=None, config=config)
    assert result["reranked_docs"] == []


# ── Nœud : generate ───────────────────────────────────────────────────────────

def test_generate_basic(config):
    from rag_agent.nodes.generation import generate
    docs  = [{"source": "/docs/budget.pdf", "chunk_index": 0, "page_content": "Le budget est 100M€.", "kind": "text", "title_path": "Budget", "_expanded": False}]
    state = create_unified_state("Quel est le budget ?")
    state["reranked_docs"] = docs  # type: ignore[index]
    llm   = _make_llm_call("Le budget est de 100 millions d'euros.\n\n---\n**Sources :**\n- budget.pdf")
    result = generate(state, llm_call=llm, config=config)
    assert "100" in result["answer"]
    assert result["error"] is None
    assert result["final_response"] == result["answer"]


def test_generate_no_docs(config):
    from rag_agent.nodes.generation import generate
    state  = create_unified_state("Q ?")
    result = generate(state, llm_call=_make_llm_call(), config=config)
    assert "Aucun extrait" in result["answer"]
    assert result["error"] is None


def test_generate_with_conversation_summary(config):
    from rag_agent.nodes.generation import generate
    docs  = [{"source": "/docs/r.pdf", "chunk_index": 0, "page_content": "Info.", "kind": "text", "title_path": "", "_expanded": False}]
    state = create_unified_state("Suite ?", conversation_summary="Précédemment : budget discuté.")
    state["reranked_docs"] = docs  # type: ignore[index]
    llm   = _make_llm_call("Réponse contextualisée.")
    result = generate(state, llm_call=llm, config=config)
    assert result["answer"] == "Réponse contextualisée."


def test_generate_follow_up(config):
    from rag_agent.nodes.generation import generate_follow_up
    state  = create_unified_state("Budget ?")
    state["answer"] = "Le budget est de 100M€."  # type: ignore[index]
    llm    = _make_llm_call('["Question 1 ?", "Question 2 ?", "Question 3 ?"]')
    result = generate_follow_up(state, llm_call=llm, config=config)
    assert len(result["follow_up_suggestions"]) <= 3
    assert "hidden_environment" in result


def test_generate_title(config):
    from rag_agent.nodes.generation import generate_title
    state  = create_unified_state("Politique budgétaire 2024 ?")
    state["answer"] = "Réponse longue sur la politique budgétaire."  # type: ignore[index]
    llm    = _make_llm_call("Politique budgétaire 2024")
    result = generate_title(state, llm_call=llm, config=config)
    assert result["conversation_title"] is not None
    assert len(result["conversation_title"]) <= 60
