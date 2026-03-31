"""Tests des outils (QueryTool, AggregateTool, ToolExecutor)."""
from __future__ import annotations

import pytest
from rag_agent.tools import (
    QueryTool,
    AggregateTool,
    ToolExecutor,
    weighted_rrf,
    combine_chunks,
    deduplicate_queries,
    weaviate_with_retry,
)
from rag_agent.state import create_unified_state


# ── QueryTool ─────────────────────────────────────────────────────────────────

def test_query_tool_mock_mode():
    tool    = QueryTool(weaviate_store=None, embedder=None)
    results = tool.execute("apprentissage automatique")
    assert len(results) > 0
    for doc in results:
        assert "page_content" in doc
        assert "chunk_index" in doc
        assert "_score" in doc


def test_query_tool_mock_no_weaviate_needed():
    """Mode mock : pas besoin de Weaviate ni d'embedder."""
    tool    = QueryTool()
    results = tool.execute("test", top_k=2)
    assert len(results) <= 3  # mock retourne min(top_k, 3)


def test_query_tool_requires_embedder_when_real():
    """Sans embedder, une erreur est levée si weaviate_store est défini."""
    mock_store = object()
    tool = QueryTool(weaviate_store=mock_store, embedder=None)
    with pytest.raises(RuntimeError, match="embedder"):
        tool.execute("test")


def test_query_tool_duplicate_detection():
    """seen_queries détecte les doublons (logique dans agent_action, pas dans QueryTool)."""
    seen: list = []
    query = "quel est le budget ?"
    is_dup = any(q.lower().strip() == query.lower().strip() for q, _ in seen)
    assert not is_dup
    seen.append((query, 1.0))
    is_dup = any(q.lower().strip() == query.lower().strip() for q, _ in seen)
    assert is_dup


# ── weighted_rrf ──────────────────────────────────────────────────────────────

def test_weighted_rrf_single_list():
    docs    = [{"source": "a.pdf", "chunk_index": 0, "_score": 0.9}]
    result  = weighted_rrf([docs], [1.0])
    assert len(result) == 1
    assert result[0]["_score"] > 0  # score RRF = 1.0 / (60 + 1) ≈ 0.0164


def test_weighted_rrf_merge_two_lists():
    list1 = [
        {"source": "a.pdf", "chunk_index": 0, "_score": 0.9},
        {"source": "b.pdf", "chunk_index": 1, "_score": 0.8},
    ]
    list2 = [
        {"source": "a.pdf", "chunk_index": 0, "_score": 0.7},  # doublon
        {"source": "c.pdf", "chunk_index": 2, "_score": 0.6},
    ]
    result = weighted_rrf([list1, list2], [1.0, 0.5])
    # a.pdf idx=0 doit avoir le score le plus élevé (présent dans les deux listes)
    assert result[0]["source"] == "a.pdf"
    assert result[0]["chunk_index"] == 0
    assert len(result) == 3  # 3 documents uniques


def test_weighted_rrf_formula():
    """Vérifie la formule : score = weight / (k + rank)."""
    docs   = [{"source": "x", "chunk_index": 0, "_score": 1.0}]
    result = weighted_rrf([docs], [1.0], k=60)
    expected = 1.0 / (60 + 1)
    assert abs(result[0]["_score"] - expected) < 1e-9


# ── combine_chunks ────────────────────────────────────────────────────────────

def test_combine_chunks_dedup():
    docs1 = [{"source": "a.pdf", "chunk_index": 1, "_score": 0.9, "page_content": "A"}]
    docs2 = [{"source": "a.pdf", "chunk_index": 1, "_score": 0.7, "page_content": "A"}]
    result = combine_chunks([docs1, docs2])
    assert len(result) == 1
    assert result[0]["_score"] == 0.9  # garde le meilleur score


def test_combine_chunks_no_dedup():
    docs1 = [{"source": "a.pdf", "chunk_index": 1, "_score": 0.9}]
    docs2 = [{"source": "b.pdf", "chunk_index": 1, "_score": 0.7}]
    result = combine_chunks([docs1, docs2])
    assert len(result) == 2


# ── deduplicate_queries ───────────────────────────────────────────────────────

def test_deduplicate_queries_case_insensitive():
    queries = [("Budget 2024", 1.0), ("budget 2024", 1.0)]
    result  = deduplicate_queries(queries)
    assert len(result) == 1
    assert result[0][1] == 2.0  # poids sommés


def test_deduplicate_queries_different():
    queries = [("Budget 2024", 1.0), ("Dépenses publiques", 1.0)]
    result  = deduplicate_queries(queries)
    assert len(result) == 2


# ── weaviate_with_retry ───────────────────────────────────────────────────────

def test_weaviate_with_retry_success_first_try():
    calls = []
    def fn():
        calls.append(1)
        return "ok"
    result = weaviate_with_retry(fn, max_retries=3, base_delay=0)
    assert result == "ok"
    assert len(calls) == 1


def test_weaviate_with_retry_success_on_third():
    calls = []
    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("échec simulé")
        return "ok"
    result = weaviate_with_retry(fn, max_retries=3, base_delay=0)
    assert result == "ok"
    assert len(calls) == 3


def test_weaviate_with_retry_exhausted():
    def fn():
        raise ConnectionError("always fails")
    with pytest.raises(RuntimeError, match="Weaviate indisponible"):
        weaviate_with_retry(fn, max_retries=2, base_delay=0)


# ── AggregateTool ────────────────────────────────────────────────────────────

def test_aggregate_tool_mock_mode():
    tool  = AggregateTool(weaviate_store=None)
    state = create_unified_state("Q")
    state = tool.execute(state, collection_names=["RagChunk"])
    assert "aggregate" in state["environment"]
    objs  = state["environment"]["aggregate"]["RagChunk"]
    assert len(objs) == 1
    # Vérifie qu'il y a des objets dans le ToolResult
    assert len(objs[0].objects) > 0


# ── ToolExecutor ──────────────────────────────────────────────────────────────

def test_tool_executor_dispatch_query():
    executor = ToolExecutor(weaviate_store=None, embedder=None)
    state    = create_unified_state("Q")
    state    = executor.execute("query", state, search_query="test", collection_names=["RagChunk"])
    assert "query" in state["environment"]


def test_tool_executor_unknown_tool():
    executor = ToolExecutor()
    state    = create_unified_state("Q")
    state    = executor.execute("unknown_tool", state)
    assert any("unknown_tool" in e.get("tool_name", "") for e in state["errors"])
