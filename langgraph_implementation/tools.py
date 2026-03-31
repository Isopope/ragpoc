"""
Tool implementations for the Elysia LangGraph system.
Implements Query, Aggregate, Summarize, and other retrieval tools.
"""

import asyncio
from typing import Optional, Any
from datetime import datetime
import json
from .state import ElysiaState, ToolResult, RetrievedObject, TaskStatus, add_to_environment


class QueryTool:
    """
    Implements the Query tool for searching collections.
    Mirrors elysia/tools/retrieval/query.py
    """
    
    def __init__(self, weaviate_client=None, llm_client=None, embedding_model: str = "text-embedding-3-small"):
        """
        Args:
            weaviate_client: Connected Weaviate client (optional for demo)
        """
        self.weaviate_client = weaviate_client
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.name = "query"
    
    async def execute(
        self,
        state: ElysiaState,
        search_query: str,
        collection_names: list[str],
        search_type: str = "hybrid",  # hybrid, vector, keyword, filter_only
        filters: Optional[dict] = None,
        limit: int = 5,
    ) -> ElysiaState:
        """
        Execute a query against one or more collections.
        
        Args:
            state: Current state
            search_query: The search query text
            collection_names: Collections to query
            search_type: Type of search (hybrid, vector, keyword, filter_only)
            filters: Optional filter conditions
            limit: Maximum results to return
        
        Returns:
            Updated state with retrieved objects
        """
        try:
            objects = []
            
            # In a real implementation, this would call Weaviate
            # For demo purposes, we'll create mock objects
            if self.weaviate_client:
                objects = await self._query_weaviate(
                    collection_names, 
                    search_query, 
                    search_type, 
                    filters, 
                    limit
                )
            else:
                # Demo: Return mock objects
                objects = self._create_mock_objects(collection_names, search_query, limit)
            
            # Create result
            retrieved_objs = [
                RetrievedObject(
                    uuid=obj.get("_uuid", f"mock-{i}"),
                    properties=obj,
                    collection_name=collection_names[0],
                    query_used=search_query,
                )
                for i, obj in enumerate(objects)
            ]
            
            result = ToolResult(
                tool_name="query",
                collection_names=collection_names,
                objects=retrieved_objs,
                metadata={
                    "search_type": search_type,
                    "search_query": search_query,
                    "filters": filters,
                    "limit": limit,
                    "results_count": len(retrieved_objs),
                    "executed_at": datetime.now().isoformat(),
                },
                status=TaskStatus.COMPLETED,
            )
            
            state = add_to_environment(state, "query", result)
            
            return state
            
        except Exception as e:
            state["errors"].append({
                "tool_name": "query",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            })
            return state
    
    async def _query_weaviate(
        self,
        collection_names: list[str],
        search_query: str,
        search_type: str,
        filters: Optional[dict],
        limit: int,
    ) -> list[dict]:
        """Execute actual Weaviate query."""
        if self.weaviate_client is None:
            raise RuntimeError("No Weaviate client configured")

        source_filter = (filters or {}).get("source")

        if hasattr(self.weaviate_client, "hybrid_search"):
            if self.llm_client is None or not hasattr(self.llm_client, "embeddings"):
                raise RuntimeError("OpenAI client with embeddings support is required for hybrid search")

            embedding_resp = await asyncio.to_thread(
                self.llm_client.embeddings.create,
                model=self.embedding_model,
                input=search_query or " ",
            )
            query_vector = embedding_resp.data[0].embedding

            docs = await asyncio.to_thread(
                self.weaviate_client.hybrid_search,
                query=search_query,
                query_vector=query_vector,
                top_k=limit,
                alpha=0.5 if search_type == "hybrid" else 1.0,
                source=source_filter,
            )
            return [
                {
                    **doc,
                    "source_name": doc.get("source_name") or doc.get("source", "").split("\\")[-1].split("/")[-1],
                }
                for doc in docs
            ]

        raise NotImplementedError("Unsupported Weaviate client type")
    
    def _create_mock_objects(
        self,
        collection_names: list[str],
        search_query: str,
        limit: int,
    ) -> list[dict]:
        """Create mock objects for demonstration."""
        return [
            {
                "_uuid": f"doc-{i}",
                "title": f"Document about {search_query} ({i+1})",
                "content": f"This is a mock document related to: {search_query}",
                "relevance_score": 0.9 - (i * 0.05),
            }
            for i in range(min(limit, 3))
        ]


class AggregateTool:
    """
    Implements the Aggregate tool for statistical operations.
    Mirrors elysia/tools/retrieval/aggregate.py
    """
    
    def __init__(self, weaviate_client=None):
        """
        Args:
            weaviate_client: Connected Weaviate client (optional for demo)
        """
        self.weaviate_client = weaviate_client
        self.name = "aggregate"
    
    async def execute(
        self,
        state: ElysiaState,
        collection_names: list[str],
        groupby_property: Optional[str] = None,
        aggregations: Optional[dict] = None,  # {property: [metrics]}
        filters: Optional[dict] = None,
    ) -> ElysiaState:
        """
        Execute an aggregation query.
        
        Args:
            state: Current state
            collection_names: Collections to aggregate over
            groupby_property: Optional property to group by
            aggregations: Aggregation operations to perform
            filters: Optional filters
        
        Returns:
            Updated state with aggregation results
        """
        try:
            agg_result = {}
            
            if self.weaviate_client:
                agg_result = await self._aggregate_weaviate(
                    collection_names,
                    groupby_property,
                    aggregations,
                    filters,
                )
            else:
                # Demo: Return mock aggregation
                agg_result = self._create_mock_aggregation(
                    collection_names,
                    aggregations or {},
                )
            
            # Convert aggregation result to objects format
            objects = [
                RetrievedObject(
                    uuid=f"agg-{i}",
                    properties=group,
                    collection_name=collection_names[0],
                )
                for i, group in enumerate(agg_result.get("groups", [agg_result]))
            ]
            
            result = ToolResult(
                tool_name="aggregate",
                collection_names=collection_names,
                objects=objects,
                metadata={
                    "aggregations": aggregations,
                    "groupby": groupby_property,
                    "filters": filters,
                    "aggregation_result": agg_result,
                    "executed_at": datetime.now().isoformat(),
                },
                status=TaskStatus.COMPLETED,
            )
            
            state = add_to_environment(state, "aggregate", result)
            
            return state
            
        except Exception as e:
            state["errors"].append({
                "tool_name": "aggregate",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            })
            return state
    
    async def _aggregate_weaviate(
        self,
        collection_names: list[str],
        groupby_property: Optional[str],
        aggregations: Optional[dict],
        filters: Optional[dict],
    ) -> dict:
        """Execute actual Weaviate aggregation."""
        if self.weaviate_client is None:
            raise RuntimeError("No Weaviate client configured")

        source_filter = (filters or {}).get("source")

        if hasattr(self.weaviate_client, "count"):
            total_count = await asyncio.to_thread(self.weaviate_client.count, source_filter)

            if groupby_property == "source" and hasattr(self.weaviate_client, "list_sources"):
                groups = []
                for source in await asyncio.to_thread(self.weaviate_client.list_sources):
                    groups.append({
                        "group_value": source,
                        "count": await asyncio.to_thread(self.weaviate_client.count, source),
                    })
                return {
                    "total_count": total_count,
                    "groups": groups,
                }

            return {
                "total_count": total_count,
                "filters": filters or {},
                "groupby": groupby_property,
            }

        raise NotImplementedError("Unsupported Weaviate client type")
    
    def _create_mock_aggregation(
        self,
        collection_names: list[str],
        aggregations: dict,
    ) -> dict:
        """Create mock aggregation results."""
        return {
            "total_count": 42,
            "metrics": {
                "average": 5.5,
                "count": 10,
                "min": 1,
                "max": 10,
            },
            "groups": [
                {"group_value": "group_a", "count": 25},
                {"group_value": "group_b", "count": 17},
            ],
        }


class SummarizationTool:
    """
    Implements the Summarization tool using LLM.
    Mirrors elysia/tools/postprocessing/summarise_items.py
    """
    
    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: Connected LLM client (optional for demo)
        """
        self.llm_client = llm_client
        self.name = "summarize"
    
    async def execute(
        self,
        state: ElysiaState,
        objects_to_summarize: Optional[list[dict]] = None,
    ) -> ElysiaState:
        """
        Summarize retrieved objects or information in the environment.
        
        Args:
            state: Current state
            objects_to_summarize: Specific objects to summarize (if None, uses environment)
        
        Returns:
            Updated state with summary
        """
        try:
            if objects_to_summarize is None:
                # Extract objects from environment
                objects_to_summarize = self._extract_from_environment(state)
            
            summary = await self._generate_summary(objects_to_summarize)
            
            result = ToolResult(
                tool_name="summarize",
                collection_names=[],
                objects=[
                    RetrievedObject(
                        uuid="summary-1",
                        properties={"summary": summary},
                        collection_name="summary",
                    )
                ],
                metadata={
                    "object_count": len(objects_to_summarize),
                    "summary": summary,
                    "executed_at": datetime.now().isoformat(),
                },
                status=TaskStatus.COMPLETED,
            )
            
            state = add_to_environment(state, "summarize", result)
            state["final_response"] = summary
            
            return state
            
        except Exception as e:
            state["errors"].append({
                "tool_name": "summarize",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            })
            return state
    
    def _extract_from_environment(self, state: ElysiaState) -> list[dict]:
        """Extract objects from the retrieved environment."""
        objects = []
        for tool_results in state["environment"].values():
            for results_list in tool_results.values():
                for result in results_list:
                    for obj in result.objects:
                        objects.append(obj.properties)
        return objects
    
    async def _generate_summary(self, objects: list[dict]) -> str:
        """Generate a summary using LLM or simple aggregation."""
        if self.llm_client and hasattr(self.llm_client, "chat") and hasattr(self.llm_client.chat, "completions"):
            serialized = json.dumps(objects[:10], ensure_ascii=False)
            response = await asyncio.to_thread(
                self.llm_client.chat.completions.create,
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You summarize retrieved RAG context. "
                            "Stay factual and only use the provided objects."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Summarize these retrieved objects:\n{serialized}",
                    },
                ],
                temperature=0.1,
                max_tokens=500,
            )
            return (response.choices[0].message.content or "").strip()

        titles = [obj.get("title", "Document") for obj in objects]
        return f"Summary of {len(objects)} items: {', '.join(titles[:3])}"


class ToolExecutor:
    """
    Manages execution of all tools in a unified way.
    """
    
    def __init__(self, weaviate_client=None, llm_client=None, embedding_model: str = "text-embedding-3-small"):
        self.tools = {
            "query": QueryTool(weaviate_client, llm_client, embedding_model=embedding_model),
            "aggregate": AggregateTool(weaviate_client),
            "summarize": SummarizationTool(llm_client),
        }
    
    async def execute(
        self,
        tool_name: str,
        state: ElysiaState,
        **tool_kwargs,
    ) -> ElysiaState:
        """
        Execute a tool with the given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            state: Current state
            **tool_kwargs: Arguments to pass to the tool
        
        Returns:
            Updated state
        """
        if tool_name not in self.tools:
            state["errors"].append({
                "tool_name": tool_name,
                "message": f"Unknown tool: {tool_name}",
                "timestamp": datetime.now().isoformat(),
            })
            return state
        
        tool = self.tools[tool_name]
        state["branch_status"] = f"Executing {tool_name}..."
        
        return await tool.execute(state, **tool_kwargs)
