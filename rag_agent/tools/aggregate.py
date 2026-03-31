"""AggregateTool — agrégations statistiques sur Weaviate.

Port de langgraph_implementation/tools.py::AggregateTool
avec adaptation pour UnifiedRAGState (synchrone, sans asyncio).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from ..state import RetrievedObject, TaskStatus, ToolResult, UnifiedRAGState, add_to_environment


class AggregateTool:
    """Agrégations COUNT/MIN/MAX/AVG sur les collections Weaviate."""

    def __init__(self, weaviate_store=None) -> None:
        self.weaviate_store = weaviate_store  # None → mode mock
        self.name = "aggregate"

    def execute(
        self,
        state: UnifiedRAGState,
        collection_names: Optional[list[str]] = None,
        groupby_property: Optional[str] = None,
        aggregations: Optional[dict[str, list[str]]] = None,
        filters: Optional[dict] = None,
    ) -> UnifiedRAGState:
        """Exécute une agrégation et met à jour state['environment']."""
        colls = collection_names or state.get("collection_names") or ["RagChunk"]
        try:
            if self.weaviate_store is not None:
                agg_result = self._aggregate_weaviate(
                    colls, groupby_property, aggregations, filters
                )
            else:
                agg_result = self._mock_aggregation(colls, aggregations or {})

            objects = [
                RetrievedObject(
                    uuid=f"agg-{i}",
                    properties=group,
                    collection_name=colls[0],
                    query_used="aggregate",
                )
                for i, group in enumerate(agg_result.get("groups", [agg_result]))
            ]
            result = ToolResult(
                tool_name="aggregate",
                collection_names=colls,
                objects=objects,
                metadata={
                    "aggregations":      aggregations,
                    "groupby":           groupby_property,
                    "filters":           filters,
                    "aggregation_result": agg_result,
                    "executed_at":       datetime.now().isoformat(),
                },
                status=TaskStatus.COMPLETED,
            )
            add_to_environment(state, result)
        except Exception as exc:
            (state.setdefault("errors", [])).append({  # type: ignore[attr-defined]
                "tool_name": "aggregate",
                "message":   str(exc),
                "timestamp": datetime.now().isoformat(),
            })
        return state

    def _aggregate_weaviate(
        self,
        collection_names: list[str],
        groupby_property: Optional[str],
        aggregations: Optional[dict],
        filters: Optional[dict],
    ) -> dict:
        source_filter = (filters or {}).get("source")
        store = self.weaviate_store

        if hasattr(store, "count"):
            total = store.count(source_filter)
            if groupby_property == "source" and hasattr(store, "list_sources"):
                groups = [
                    {"group_value": src, "count": store.count(src)}
                    for src in store.list_sources()
                ]
                return {"total_count": total, "groups": groups}
            return {"total_count": total, "filters": filters or {}, "groupby": groupby_property}

        raise NotImplementedError("Type de client Weaviate non supporté pour l'agrégation")

    def _mock_aggregation(self, collection_names: list[str], aggregations: dict) -> dict:
        return {
            "total_count": 42,
            "metrics":     {"average": 5.5, "count": 10, "min": 1, "max": 10},
            "groups": [
                {"group_value": "group_a", "count": 25},
                {"group_value": "group_b", "count": 17},
            ],
        }
