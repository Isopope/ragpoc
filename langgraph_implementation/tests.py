"""
Unit tests for the Elysia LangGraph implementation.
Run with: pytest tests.py -v
"""

import pytest
import asyncio
from datetime import datetime

from langgraph_implementation.state import (
    ElysiaState,
    create_initial_state,
    TaskStatus,
    RetrievedObject,
    ToolResult,
    add_to_environment,
)

from langgraph_implementation.decision_nodes import (
    DecisionNode,
    TreeBuilder,
    MultibranchTree,
    OneBranchTree,
)

from langgraph_implementation.tools import (
    QueryTool,
    AggregateTool,
    SummarizationTool,
)

from langgraph_implementation.graph import ElysiaGraph


# ============================================
# State Tests
# ============================================

def test_create_initial_state():
    """Test initial state creation."""
    state = create_initial_state(
        user_prompt="Test query",
        user_id="user-1",
        conversation_id="conv-1",
        collection_names=["test_collection"],
    )
    
    assert state["user_prompt"] == "Test query"
    assert state["user_id"] == "user-1"
    assert state["tree_depth"] == 0
    assert state["max_tree_depth"] == 5
    assert state["current_branch"] == "base"
    assert len(state["environment"]) == 0


def test_add_to_environment():
    """Test adding results to environment."""
    state = create_initial_state(
        user_prompt="Test",
        user_id="user-1",
        conversation_id="conv-1",
        collection_names=["coll"],
    )
    
    obj = RetrievedObject(
        uuid="test-uuid",
        properties={"title": "Test Document"},
        collection_name="coll",
    )
    
    result = ToolResult(
        tool_name="query",
        collection_names=["coll"],
        objects=[obj],
        metadata={"count": 1},
    )
    
    state = add_to_environment(state, "query", result)
    
    assert "query" in state["environment"]
    assert "coll" in state["environment"]["query"]
    assert len(state["environment"]["query"]["coll"]) == 1


# ============================================
# Decision Node Tests
# ============================================

def test_decision_node_creation():
    """Test DecisionNode creation."""
    node = DecisionNode(
        node_id="test_node",
        instruction="Test instruction",
        is_root=True,
    )
    
    assert node.node_id == "test_node"
    assert node.instruction == "Test instruction"
    assert node.is_root is True
    assert node.visited_count == 0


def test_tree_builder():
    """Test TreeBuilder functionality."""
    builder = TreeBuilder()
    
    # Add a branch
    builder.add_branch(
        branch_id="base",
        instruction="Root instruction",
        tools=[{"name": "query"}],
        is_root=True,
    )
    
    assert "base" in builder.nodes
    assert builder.root == "base"
    assert builder.nodes["base"].is_root


def test_multibranch_tree():
    """Test MultibranchTree structure."""
    tree = MultibranchTree()
    
    # Should have base and search branches
    assert "base" in tree.branches
    assert "search" in tree.branches
    
    # Base should be root
    assert tree.nodes["base"].is_root
    assert tree.nodes["search"].parent_node == "base"


def test_onebranch_tree():
    """Test OneBranchTree structure."""
    tree = OneBranchTree()
    
    # Should only have base
    assert "base" in tree.branches
    assert tree.nodes["base"].is_root
    assert len(tree.branches) == 1


# ============================================
# Tool Tests
# ============================================

@pytest.mark.asyncio
async def test_query_tool_basic():
    """Test basic QueryTool execution."""
    tool = QueryTool()
    state = create_initial_state(
        user_prompt="Test query",
        user_id="user-1",
        conversation_id="conv-1",
        collection_names=["docs"],
    )
    
    state = await tool.execute(
        state,
        search_query="machine learning",
        collection_names=["docs"],
        limit=3,
    )
    
    assert "query" in state["environment"]
    assert len(state["errors"]) == 0


@pytest.mark.asyncio
async def test_aggregate_tool_basic():
    """Test basic AggregateTool execution."""
    tool = AggregateTool()
    state = create_initial_state(
        user_prompt="Count items",
        user_id="user-1",
        conversation_id="conv-1",
        collection_names=["products"],
    )
    
    state = await tool.execute(
        state,
        collection_names=["products"],
        aggregations={"count": ["COUNT"]},
    )
    
    assert "aggregate" in state["environment"]


@pytest.mark.asyncio
async def test_summarization_tool():
    """Test SummarizationTool."""
    tool = SummarizationTool()
    state = create_initial_state(
        user_prompt="Summarize",
        user_id="user-1",
        conversation_id="conv-1",
        collection_names=["docs"],
    )
    
    # Add some mock objects to environment first
    obj = RetrievedObject(
        uuid="doc-1",
        properties={"title": "Document 1", "content": "Test content"},
        collection_name="docs",
    )
    result = ToolResult(
        tool_name="query",
        collection_names=["docs"],
        objects=[obj],
        metadata={},
    )
    state = add_to_environment(state, "query", result)
    
    # Now summarize
    state = await tool.execute(state)
    
    assert "summarize" in state["environment"]
    assert state["final_response"] is not None


# ============================================
# Graph Tests
# ============================================

@pytest.mark.asyncio
async def test_graph_creation():
    """Test ElysiaGraph creation."""
    graph = ElysiaGraph(mode="onebranch")
    
    assert graph.mode == "onebranch"
    assert graph.graph is not None
    assert graph.tree_builder is not None


@pytest.mark.asyncio
async def test_graph_execution_basic():
    """Test basic graph execution."""
    graph = ElysiaGraph(mode="onebranch")
    
    result = await graph.run(
        user_prompt="What is artificial intelligence?",
        collection_names=["docs"],
    )
    
    assert result["response"] is not None
    assert result["user_prompt"] == "What is artificial intelligence?"
    assert result["tree_depth"] >= 0


@pytest.mark.asyncio
async def test_graph_multibranch():
    """Test multibranch graph execution."""
    graph = ElysiaGraph(mode="multibranch")
    
    result = await graph.run(
        user_prompt="Find expensive products",
        collection_names=["products"],
    )
    
    assert result["response"] is not None
    assert result["decision_history"] is not None


@pytest.mark.asyncio
async def test_graph_with_metadata():
    """Test graph execution with collection metadata."""
    graph = ElysiaGraph(mode="onebranch")
    
    metadata = {
        "docs": {
            "length": 1000,
            "summary": "Example documents",
            "fields": [
                {"name": "title", "type": "text"},
                {"name": "content", "type": "text"},
            ],
        }
    }
    
    result = await graph.run(
        user_prompt="Find documents",
        collection_names=["docs"],
        collection_metadata=metadata,
    )
    
    assert result["response"] is not None


# ============================================
# Integration Tests
# ============================================

@pytest.mark.asyncio
async def test_full_workflow():
    """Test complete workflow from query to response."""
    graph = ElysiaGraph(mode="multibranch")
    
    result = await graph.run(
        user_prompt="What are the top products?",
        user_id="test-user",
        conversation_id="test-conv",
        collection_names=["products", "reviews"],
    )
    
    # Verify result structure
    assert "response" in result
    assert "user_prompt" in result
    assert "decision_history" in result
    assert "retrieved_objects" in result
    assert "errors" in result
    assert "tree_depth" in result
    assert "metadata" in result
    
    # Verify metadata
    metadata = result["metadata"]
    assert metadata["user_id"] == "test-user"
    assert metadata["conversation_id"] == "test-conv"
    assert metadata["mode"] == "multibranch"


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling with empty collections."""
    graph = ElysiaGraph(mode="onebranch")
    
    result = await graph.run(
        user_prompt="Query nothing",
        collection_names=[],  # No collections
    )
    
    # Should still produce a result
    assert result["response"] is not None
    # May have errors since no collections
    # (depends on implementation)


# ============================================
# Performance Tests
# ============================================

@pytest.mark.asyncio
async def test_performance_single_execution():
    """Test performance of single execution."""
    import time
    
    graph = ElysiaGraph(mode="onebranch")
    
    start = time.time()
    result = await graph.run(
        user_prompt="Test performance",
        collection_names=["test"],
    )
    elapsed = time.time() - start
    
    print(f"\nSingle execution time: {elapsed:.2f}s")
    assert elapsed < 30, "Execution took too long"


@pytest.mark.asyncio
async def test_performance_multiple_executions():
    """Test performance with multiple executions."""
    import time
    
    graph = ElysiaGraph(mode="onebranch")
    
    queries = [
        "Query 1",
        "Query 2",
        "Query 3",
    ]
    
    start = time.time()
    for query in queries:
        await graph.run(query, collection_names=["test"])
    elapsed = time.time() - start
    
    avg_time = elapsed / len(queries)
    print(f"\nAverage execution time: {avg_time:.2f}s")


# ============================================
# Fixtures
# ============================================

@pytest.fixture
def sample_state():
    """Fixture providing a sample state."""
    return create_initial_state(
        user_prompt="Sample query",
        user_id="user-1",
        conversation_id="conv-1",
        collection_names=["collection1", "collection2"],
    )


@pytest.fixture
def sample_tree_builder():
    """Fixture providing a tree builder."""
    return MultibranchTree()


# ============================================
# Run Tests
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
