"""Tests du module state."""
from __future__ import annotations

import pytest
from rag_agent.state import (
    UnifiedRAGState,
    create_unified_state,
    log_entry,
    format_environment_for_llm,
    tasks_completed_string,
    ToolResult,
    RetrievedObject,
    TaskStatus,
    add_to_environment,
)


def test_create_unified_state_defaults():
    state = create_unified_state("Ma question ?")
    assert state["question"] == "Ma question ?"
    assert state["answer"] == ""
    assert state["agent_iterations"] == 0
    assert state["sub_queries"] == []
    assert state["messages"] == []
    assert state["all_docs"] == []
    assert state["error"] is None
    assert state["current_branch"] == "plan"
    assert state["tree_depth"] == 0
    assert state["follow_up_suggestions"] == []
    assert state["conversation_title"] is None


def test_create_unified_state_with_source():
    state = create_unified_state("Q", source="/docs/file.pdf", available_sources=["/docs/file.pdf"])
    assert state["source_filter"] == "/docs/file.pdf"
    assert "/docs/file.pdf" in state["available_sources"]


def test_seen_keys_is_set():
    state = create_unified_state("Q")
    assert isinstance(state["seen_keys"], set), "seen_keys doit être un set (non JSON-serializable)"


def test_log_entry_format():
    entry = log_entry("analyze", "Test message", {"key": "val"})
    assert entry["step"]     == "analyze"
    assert entry["message"]  == "Test message"
    assert entry["metadata"] == {"key": "val"}
    assert "ts" in entry
    assert "Z" in entry["ts"] or "+" in entry["ts"]  # UTC ISO format


def test_format_environment_for_llm_empty():
    state = create_unified_state("Q")
    result = format_environment_for_llm(state)
    assert result == "No retrieved objects yet."


def test_format_environment_for_llm_with_data():
    state = create_unified_state("Q")
    obj = RetrievedObject(uuid="abc123", properties={"page_content": "Contenu test"}, collection_name="RagChunk", query_used="test")
    tr  = ToolResult(tool_name="query", collection_names=["RagChunk"], objects=[obj])
    add_to_environment(state, tr)
    result = format_environment_for_llm(state)
    assert "query" in result
    assert "RagChunk" in result


def test_tasks_completed_string_empty():
    state = create_unified_state("Q")
    assert tasks_completed_string(state) == ""


def test_tasks_completed_string_with_data():
    state = create_unified_state("Q")
    state["tasks_completed"] = [  # type: ignore[index]
        {"action": "search", "status": "completed", "details": "Trouvé 5 chunks"},
    ]
    result = tasks_completed_string(state)
    assert "<task_1" in result
    assert "search" in result
    assert "completed" in result


def test_add_to_environment():
    state = create_unified_state("Q")
    obj = RetrievedObject(uuid="x1", properties={}, collection_name="C1", query_used="q")
    tr  = ToolResult(tool_name="query", collection_names=["C1"], objects=[obj])
    add_to_environment(state, tr)
    assert "query" in state["environment"]
    assert "C1" in state["environment"]["query"]
    assert len(state["environment"]["query"]["C1"]) == 1
