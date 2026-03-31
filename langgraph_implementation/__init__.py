"""
Elysia Decision Tree reimplemented in LangGraph.

This module provides a LangGraph-based implementation of the Elysia agentic decision tree system.
It maintains the core architecture and logic of Elysia while leveraging LangGraph's powerful
state management and graph execution framework.

Main Components:
- ElysiaGraph: Main orchestrator class
- DecisionNodes: Tree structure (MultibranchTree, OneBranchTree)
- ToolExecutor: Executes retrieval tools (Query, Aggregate, Summarize)
- DecisionMaker: LLM-based decision making
- State: Unified state management (mirrors TreeData and Environment)

Example Usage:
    from langgraph_implementation.graph import ElysiaGraph
    
    graph = ElysiaGraph(mode="multibranch")
    result = await graph.run(
        user_prompt="Find the most expensive products",
        collection_names=["products"],
    )
    print(result["response"])
"""

from .state import (
    ElysiaState,
    create_initial_state,
    TaskStatus,
    RetrievedObject,
    ToolResult,
)

from .decision_nodes import (
    DecisionNode,
    TreeBuilder,
    MultibranchTree,
    OneBranchTree,
    format_decision_prompt_context,
)

from .tools import (
    QueryTool,
    AggregateTool,
    SummarizationTool,
    ToolExecutor,
)

from .llm_integration import (
    DecisionMaker,
    DecisionOutput,
    DSPyCompatibleModule,
)

from .graph import ElysiaGraph
from .rag_agent import ElysiaRAGAgent

__all__ = [
    # State
    "ElysiaState",
    "create_initial_state",
    "TaskStatus",
    "RetrievedObject",
    "ToolResult",
    
    # Decision Nodes
    "DecisionNode",
    "TreeBuilder",
    "MultibranchTree",
    "OneBranchTree",
    
    # Tools
    "QueryTool",
    "AggregateTool",
    "SummarizationTool",
    "ToolExecutor",
    
    # LLM Integration
    "DecisionMaker",
    "DecisionOutput",
    "DSPyCompatibleModule",
    
    # Main Graph
    "ElysiaGraph",
    "ElysiaRAGAgent",
]

__version__ = "0.1.0"
__description__ = "Elysia Decision Tree in LangGraph"
