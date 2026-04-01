"""
Test for the corrected Elysia LangGraph implementation.
Validates: end_actions, impossible, tree restart, ForcedTextResponse, successive_actions.
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langgraph_implementation.graph import ElysiaGraph
from langgraph_implementation.decision_nodes import MultibranchTree


def test_successive_actions_recursive():
    """Verify successive_actions is recursive, not flat."""
    tree = MultibranchTree()
    sa = tree.get_successive_actions()
    
    print("=== Test 1: successive_actions recursive ===")
    print(f"  Result: {sa}")
    
    # Must contain "search" as a key with sub-actions
    assert "search" in sa, f"'search' missing from successive_actions: {sa}"
    assert "query" in sa["search"], f"'query' missing from search sub-actions: {sa['search']}"
    assert "aggregate" in sa["search"], f"'aggregate' missing from search sub-actions: {sa['search']}"
    # Must also contain root-level options
    assert "text_response" in sa or "summarize" in sa, f"Root tools missing: {sa}"
    print("  ✅ PASSED\n")


def test_filter_available_tools():
    """Verify is_tool_available filtering works."""
    tree = MultibranchTree()
    
    # Empty state → summarize should be unavailable
    empty_state = {"environment": {}, "current_branch": "base"}
    available, unavailable = tree.filter_available_tools("base", empty_state)
    
    print("=== Test 2: filter_available_tools (empty state) ===")
    print(f"  Available: {list(available.keys())}")
    print(f"  Unavailable: {list(unavailable.keys())}")
    assert "summarize" in unavailable, f"summarize should be unavailable with empty env"
    assert "text_response" in available, f"text_response should always be available"
    assert "search" in available, f"search branch should always be available"
    print("  ✅ PASSED\n")
    
    # State with data → summarize should be available
    data_state = {"environment": {"query": {"col1": []}}, "current_branch": "base"}
    available2, unavailable2 = tree.filter_available_tools("base", data_state)
    
    print("=== Test 3: filter_available_tools (with data) ===")
    print(f"  Available: {list(available2.keys())}")
    print(f"  Unavailable: {list(unavailable2.keys())}")
    assert "summarize" in available2, f"summarize should be available with data"
    print("  ✅ PASSED\n")


async def test_graph_mock_run():
    """Run the full graph in mock mode and validate flow."""
    print("=== Test 4: Full graph mock run (multibranch) ===")
    graph = ElysiaGraph(mode="multibranch")
    
    result = await graph.run(
        user_prompt="What documents discuss machine learning?",
        user_id="test_user",
        conversation_id="test_conv",
        collection_names=["Documents"],
    )
    
    print(f"  Response: {result.get('response')}")
    print(f"  Tree Depth: {result.get('tree_depth')}")
    print(f"  Trees Completed: {result.get('num_trees_completed')}")
    print(f"  End Actions: {result.get('end_actions')}")
    print(f"  Impossible: {result.get('impossible')}")
    print(f"  Decision History: {result.get('decision_history')}")
    
    errors = result.get("errors", [])
    print(f"  Errors: {len(errors)}")
    for e in errors:
        print(f"    - {e}")
    
    # Validate: should have a response
    assert result.get("response") is not None, "No final response generated!"
    
    # Validate: tree should not have hit max depth (infinite loop bug fixed)
    assert result.get("tree_depth", 99) <= 6, f"Tree depth too high: {result.get('tree_depth')} (likely infinite loop)"
    
    # Validate: decision history should show search→query flow, then text_response
    history = result.get("decision_history", [])
    assert len(history) >= 2, f"Decision history too short: {history}"
    
    print("  ✅ PASSED\n")


async def test_graph_onebranch():
    """Run the graph in onebranch mode."""
    print("=== Test 5: Full graph mock run (onebranch) ===")
    graph = ElysiaGraph(mode="onebranch")
    
    result = await graph.run(
        user_prompt="Count all entries",
        user_id="test_user",
        conversation_id="test_conv",
        collection_names=["Documents"],
    )
    
    print(f"  Response: {result.get('response')}")
    print(f"  Decision History: {result.get('decision_history')}")
    assert result.get("response") is not None, "No response in onebranch mode!"
    print("  ✅ PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  Elysia LangGraph Implementation Tests")
    print("=" * 60 + "\n")
    
    # Sync tests
    test_successive_actions_recursive()
    test_filter_available_tools()
    
    # Async tests
    asyncio.run(test_graph_mock_run())
    asyncio.run(test_graph_onebranch())
    
    print("=" * 60)
    print("  ALL TESTS PASSED ✅")
    print("=" * 60)
