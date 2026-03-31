"""Tests d'intégration du graphe LangGraph (mode mock — sans Weaviate ni LLM réels)."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers mock ───────────────────────────────────────────────────────────────

def _mock_llm_call(content: str = "Voici la réponse finale."):
    """Un callable LLM qui répond sans tool_calls (pour déclencher generate)."""
    def _call(messages, **kwargs):
        resp = MagicMock()
        resp.choices[0].message.content    = content
        resp.choices[0].message.tool_calls = None
        resp.choices[0].finish_reason      = "stop"
        return resp
    return _call


def _make_rag_agent_mock():
    """Crée un RAGAgent avec mocks — ne nécessite ni Weaviate ni OpenAI."""
    from unittest.mock import patch as _patch
    from rag_agent.graph import RAGAgent

    mock_store = MagicMock()
    mock_store.list_sources.return_value = ["/mock/doc.pdf"]

    with _patch("rag_agent.graph.build_unified_graph") as mock_build:
        # Graphe simulé qui retourne un état final minimal
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "question_id":           "test-uuid-123",
            "question":              "Question test ?",
            "answer":                "Réponse mock.",
            "final_response":        "Réponse mock.",
            "reranked_docs":         [{"source": "/mock/doc.pdf", "chunk_index": 0, "page_content": "Content", "_rerank_score": 0.9}],
            "retrieved_docs":        [{"source": "/mock/doc.pdf", "chunk_index": 0, "page_content": "Content"}],
            "decision_log":          [{"step": "analyze", "ts": "2024-01-01T00:00:00Z", "message": "OK", "metadata": {}}],
            "decision_history":      ["plan.analyze", "react.search", "synthesize.rerank"],
            "tree_depth":            2,
            "follow_up_suggestions": ["Suite 1 ?", "Suite 2 ?"],
            "conversation_title":    "Question test",
            "error":                 None,
        }

        def _stream(initial_state, **kwargs):
            yield {"analyze_and_plan": {"sub_queries": ["test"]}}
            yield {"agent_reason":     {"agent_iterations": 1}}
            yield {"consolidate":      {"retrieved_docs": []}}
            yield {"rerank":           {"reranked_docs": []}}
            yield {"generate":         {"answer": "Réponse mock."}}
            yield {"generate_follow_up": {"follow_up_suggestions": []}}
            yield {"generate_title":   {"conversation_title": "Test"}}

        mock_graph.stream  = _stream
        mock_build.return_value = mock_graph

        agent = RAGAgent(mock_store, "fake-key")
        return agent, mock_graph


# ── Import backward-compatible ────────────────────────────────────────────────

def test_from_rag_agent_import():
    """L'import `from rag_agent import RAGAgent` doit fonctionner."""
    from rag_agent import RAGAgent
    assert RAGAgent is not None


def test_rag_config_import():
    from rag_agent import RAGConfig
    config = RAGConfig(openai_key="test")
    assert config.llm_model == "gpt-4.1"
    assert config.hybrid_alpha == 0.5


# ── RAGAgent.query() ──────────────────────────────────────────────────────────

def test_query_returns_correct_keys():
    agent, _ = _make_rag_agent_mock()
    result   = agent.query("Question test ?")
    required_keys = {"answer", "sources", "question", "question_id", "n_retrieved", "decision_log", "error"}
    assert required_keys.issubset(set(result.keys()))


def test_query_new_keys_present():
    """Clés supplémentaires du package unifié."""
    agent, _ = _make_rag_agent_mock()
    result   = agent.query("Question test ?")
    assert "follow_up_suggestions" in result
    assert "conversation_title" in result


def test_query_answer_not_empty():
    agent, _ = _make_rag_agent_mock()
    result   = agent.query("Question test ?")
    assert result["answer"] == "Réponse mock."
    assert result["error"] is None


def test_query_n_retrieved():
    agent, _ = _make_rag_agent_mock()
    result   = agent.query("Question test ?")
    assert result["n_retrieved"] == 1  # 1 doc dans retrieved_docs mock


# ── RAGAgent.stream_query() ───────────────────────────────────────────────────

def test_stream_query_yields_events():
    agent, _ = _make_rag_agent_mock()
    events   = list(agent.stream_query("Question test ?"))
    assert len(events) > 0


def test_stream_query_contains_expected_node_names():
    agent, _ = _make_rag_agent_mock()
    nodes_seen = set()
    for event in agent.stream_query("Question test ?"):
        nodes_seen.update(event.keys())
    expected = {"analyze_and_plan", "generate"}
    assert expected.issubset(nodes_seen)


# ── State : decision_history et tree_depth ────────────────────────────────────

def test_query_decision_history_tracked():
    agent, _ = _make_rag_agent_mock()
    result   = agent.query("Q ?")
    # decision_log doit contenir au moins une entrée
    assert len(result["decision_log"]) >= 1


# ── UnifiedRAGState : create_unified_state ───────────────────────────────────

def test_create_unified_state_for_graph():
    from rag_agent.state import create_unified_state
    state = create_unified_state(
        "Politique de dépenses ?",
        source="/docs/budget.pdf",
        available_sources=["/docs/budget.pdf"],
        conversation_summary="Historique de conversation.",
    )
    assert state["question"]          == "Politique de dépenses ?"
    assert state["source_filter"]     == "/docs/budget.pdf"
    assert state["conversation_summary"] == "Historique de conversation."
    assert state["current_branch"]    == "plan"
    assert state["tree_depth"]        == 0


# ── Tree traversal ────────────────────────────────────────────────────────────

def test_rag_tree_structure():
    from rag_agent.tree import RAGTree
    tree = RAGTree()
    assert tree.root == "plan"
    assert "react" in tree.nodes
    assert "synthesize" in tree.nodes
    # Vérifie les successors depuis "plan"
    actions = tree.get_successive_actions("plan")
    assert "analyze" in actions


def test_multibranch_tree_structure():
    from rag_agent.tree import MultibranchTree
    tree = MultibranchTree()
    assert tree.root == "base"
    assert "search" in tree.nodes


def test_onebranch_tree_flat():
    from rag_agent.tree import OneBranchTree
    tree    = OneBranchTree()
    actions = tree.get_successive_actions("base")
    assert "query" in actions
    assert "aggregate" in actions
    assert "summarize" in actions


def test_get_tree_rag():
    from rag_agent.tree import get_tree, RAGTree
    tree = get_tree("rag")
    assert isinstance(tree, RAGTree)


def test_get_tree_invalid():
    from rag_agent.tree import get_tree
    with pytest.raises(ValueError, match="Mode d'arbre inconnu"):
        get_tree("invalid_mode")
