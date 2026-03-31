"""ToolExecutor — dispatche vers QueryTool, AggregateTool, etc."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Optional

from .query import QueryTool
from .aggregate import AggregateTool
from ..state import UnifiedRAGState


class ToolExecutor:
    """Gestionnaire unifié d'exécution des outils."""

    def __init__(
        self,
        weaviate_store=None,
        embedder: Optional[Callable] = None,
    ) -> None:
        self._query     = QueryTool(weaviate_store, embedder)
        self._aggregate = AggregateTool(weaviate_store)
        self._tools: dict[str, Any] = {
            "query":     self._query,
            "aggregate": self._aggregate,
        }

    def execute(
        self,
        tool_name: str,
        state: UnifiedRAGState,
        **kwargs,
    ) -> UnifiedRAGState:
        """Exécute l'outil demandé et met à jour state."""
        if tool_name not in self._tools:
            (state.setdefault("errors", [])).append({  # type: ignore[attr-defined]
                "tool_name": tool_name,
                "message":   f"Outil inconnu : {tool_name!r}",
                "timestamp": datetime.now().isoformat(),
            })
            return state

        state["branch_status"] = f"Executing {tool_name}..."  # type: ignore[index]
        tool = self._tools[tool_name]
        # QueryTool utilise execute_as_tool_result(state, ...) pour l'interface ToolExecutor
        if isinstance(tool, QueryTool):
            return tool.execute_as_tool_result(state, **kwargs)
        return tool.execute(state, **kwargs)
