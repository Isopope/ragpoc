"""Package rag_agent — pipeline RAG unifié LangGraph.

Fusion de rag_pipeline.py (robustesse de production) et langgraph_implementation/
(architecture modulaire, arbre de décision, tests, mocks).

Usage minimal :
    from rag_agent import RAGAgent

    agent = RAGAgent(weaviate_store, openai_key="...")
    result = agent.query("Quelle est la politique budgétaire ?")
    print(result["answer"])
"""
from .graph import RAGAgent, build_unified_graph
from .config import RAGConfig
from .state import UnifiedRAGState, create_unified_state, log_entry
from .tree import MultibranchTree, OneBranchTree, RAGTree, get_tree
from .tools import QueryTool, AggregateTool, ToolExecutor
from .llm import DecisionMaker, DecisionOutput, PlanningOutput

__version__ = "1.0.0"

__all__ = [
    # Interface principale
    "RAGAgent",
    "build_unified_graph",
    "RAGConfig",
    # State
    "UnifiedRAGState",
    "create_unified_state",
    "log_entry",
    # Tree
    "MultibranchTree",
    "OneBranchTree",
    "RAGTree",
    "get_tree",
    # Tools
    "QueryTool",
    "AggregateTool",
    "ToolExecutor",
    # LLM
    "DecisionMaker",
    "DecisionOutput",
    "PlanningOutput",
]
