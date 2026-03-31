"""
Bridge between the existing RAG project and the Elysia-style LangGraph implementation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from openai import OpenAI

from .graph import ElysiaGraph


class ElysiaRAGAgent:
    """Wrapper exposing an Elysia-backed RAG agent with a familiar query API."""

    def __init__(
        self,
        weaviate_store,
        openai_key: str,
        *,
        llm_model: str = "gpt-4.1",
        embedding_model: str = "text-embedding-3-small",
        mode: str = "multibranch",
    ) -> None:
        self._store = weaviate_store
        self._client = OpenAI(api_key=openai_key)
        self._graph = ElysiaGraph(
            mode=mode,
            weaviate_client=weaviate_store,
            llm_client=self._client,
            model_name=llm_model,
            embedding_model=embedding_model,
        )

    async def aquery(
        self,
        question: str,
        source: str | None = None,
        conversation_summary: str = "",
    ) -> dict[str, Any]:
        try:
            sources = self._store.list_sources()
        except Exception:
            sources = []

        conversation_history = []
        if conversation_summary.strip():
            conversation_history.append({
                "role": "system",
                "content": conversation_summary.strip(),
            })

        result = await self._graph.run(
            user_prompt=question,
            collection_names=[Path(s).name for s in sources],
            collection_metadata={
                "filters": {"source": source} if source else {},
            },
            conversation_history=conversation_history,
        )

        retrieved_docs = [
            item["properties"]
            for item in result.get("retrieved_objects", [])
            if item.get("tool") == "query"
        ]

        decision_log = [
            {
                "step": "elysia.decision",
                "ts": result.get("metadata", {}).get("executed_at"),
                "message": f"Action choisie : {action}",
                "metadata": {},
            }
            for action in result.get("decision_history", [])
        ]

        return {
            "answer": result.get("response") or "",
            "sources": retrieved_docs,
            "question": question,
            "n_retrieved": len(retrieved_docs),
            "decision_log": decision_log,
            "error": None if result.get("response") else "Réponse vide",
            "raw_result": result,
        }

    def query(
        self,
        question: str,
        source: str | None = None,
        conversation_summary: str = "",
    ) -> dict[str, Any]:
        return asyncio.run(self.aquery(question, source=source, conversation_summary=conversation_summary))

    def stream_query(
        self,
        question: str,
        source: str | None = None,
        conversation_summary: str = "",
    ):
        result = self.query(
            question,
            source=source,
            conversation_summary=conversation_summary,
        )
        yield {
            "elysia_graph": {
                "answer": result["answer"],
                "reranked_docs": result["sources"],
                "decision_log": result["decision_log"],
            }
        }
