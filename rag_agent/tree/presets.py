"""Presets d'arbres de décision préconfigurés."""
from __future__ import annotations

from .builder import TreeBuilder


class MultibranchTree(TreeBuilder):
    """Arbre multi-branches (port de langgraph_implementation).

    Structure :
      [base] → tools: summarize, text_response
              → branche [search]
      [search] → tools: query, aggregate
    """

    def __init__(self) -> None:
        super().__init__()
        self.add_branch(
            branch_id="base",
            instruction=(
                "Choose a base-level task based on the user's prompt and available information. "
                "You can search (query or aggregate), summarize retrieved information, "
                "or generate a text response."
            ),
            tools=[
                {"name": "summarize",      "description": "Summarize retrieved information"},
                {"name": "text_response",  "description": "Generate a text response"},
            ],
            is_root=True,
        )
        self.add_branch(
            branch_id="search",
            instruction=(
                "Choose between querying the knowledge base via semantic/keyword search, "
                "or aggregating information.\n"
                "- Querying: For specific information requiring a search query.\n"
                "- Aggregating: For statistics, counting, or summary operations."
            ),
            tools=[
                {"name": "query",     "description": "Semantic/keyword search on knowledge base"},
                {"name": "aggregate", "description": "Perform aggregation operations"},
            ],
            is_root=False,
            parent_branch_id="base",
        )


class OneBranchTree(TreeBuilder):
    """Arbre mono-branche (port de langgraph_implementation).

    Tous les outils disponibles à la racine.
    """

    def __init__(self) -> None:
        super().__init__()
        self.add_branch(
            branch_id="base",
            instruction=(
                "Choose a task based on the user's prompt and available information. "
                "Decide based on the tools you have available as well as their descriptions."
            ),
            tools=[
                {"name": "query",         "description": "Search the knowledge base"},
                {"name": "aggregate",     "description": "Perform aggregation on the knowledge base"},
                {"name": "summarize",     "description": "Summarize retrieved information"},
                {"name": "text_response", "description": "Generate a text response"},
                {"name": "visualize",     "description": "Visualize data"},
            ],
            is_root=True,
        )


class RAGTree(TreeBuilder):
    """Arbre de décision adapté au pipeline RAG unifié.

    Modélise les trois phases d'exécution du pipeline comme branches navigables.
    Usage : mettre à jour current_branch, decision_history, tree_depth dans le graphe.

    Branches :
      [plan]       → analyze (analyze_and_plan)
      [react]      → search_documents, get_neighboring_chunk, compress, conclude
      [synthesize] → rerank, generate, follow_up, title
    """

    def __init__(self) -> None:
        super().__init__()
        self.add_branch(
            branch_id="plan",
            instruction=(
                "Analyze the user's question and plan the retrieval strategy. "
                "Identify the target document and decompose the question into sub-queries."
            ),
            tools=[
                {
                    "name":        "analyze",
                    "description": "Decompose question into sub-queries, identify source filter",
                }
            ],
            is_root=True,
        )
        self.add_branch(
            branch_id="react",
            instruction=(
                "Execute the ReAct retrieval loop. Search documents, expand context with "
                "neighboring chunks, or conclude when sufficient information is found."
            ),
            tools=[
                {
                    "name":        "search_documents",
                    "description": "Hybrid semantic+BM25 search on indexed document chunks",
                },
                {
                    "name":        "get_neighboring_chunk",
                    "description": "Retrieve adjacent chunk for truncated context",
                },
                {
                    "name":        "compress",
                    "description": "Compress accumulated context when token budget exceeded",
                },
                {
                    "name":        "conclude",
                    "description": "Signal end of retrieval loop when sufficient context collected",
                },
            ],
            is_root=False,
            parent_branch_id="plan",
        )
        self.add_branch(
            branch_id="synthesize",
            instruction=(
                "Post-retrieval synthesis: rerank retrieved chunks, generate the final answer, "
                "create follow-up suggestions, and title the conversation."
            ),
            tools=[
                {"name": "rerank",    "description": "Rerank retrieved documents by query relevance"},
                {"name": "generate",  "description": "Generate final markdown answer from ranked chunks"},
                {"name": "follow_up", "description": "Generate follow-up question suggestions"},
                {"name": "title",     "description": "Generate short conversation title"},
            ],
            is_root=False,
            parent_branch_id="react",
        )


def get_tree(mode: str) -> TreeBuilder:
    """Retourne l'arbre correspondant au mode demandé."""
    trees = {
        "rag":          RAGTree,
        "multibranch":  MultibranchTree,
        "onebranch":    OneBranchTree,
    }
    klass = trees.get(mode)
    if klass is None:
        raise ValueError(f"Mode d'arbre inconnu : {mode!r}. Valeurs : {list(trees)}")
    return klass()
