"""
Main LangGraph graph implementation for the Elysia decision tree.

Key changes vs original version:
- Completion is driven by LLM (end_actions / impossible / text_response), NOT by object count
- After a tool executes in a sub-branch, current_branch resets to root (tree restart)
- ForcedTextResponse safety-net: if tree finishes without a response, one is generated
- _route_after_decision correctly handles branch transitions (e.g. "search")
"""

from typing import Optional, Any
from datetime import datetime
import json

from langgraph.graph import StateGraph, START, END
from .state import ElysiaState, create_initial_state
from .decision_nodes import MultibranchTree, OneBranchTree, format_decision_prompt_context
from .tools import ToolExecutor
from .llm_integration import DecisionMaker


class ElysiaGraph:
    """
    Main graph class for the Elysia decision tree in LangGraph.
    Orchestrates the flow between decision nodes and tool execution.
    """

    def __init__(
        self,
        mode: str = "multibranch",
        weaviate_client=None,
        llm_client=None,
        model_name: str = "gpt-4",
        embedding_model: str = "text-embedding-3-small",
    ):
        self.mode = mode
        self.weaviate_client = weaviate_client
        self.llm_client = llm_client
        self.model_name = model_name
        self.embedding_model = embedding_model

        # Initialize components
        if mode == "multibranch":
            self.tree_builder = MultibranchTree()
        else:
            self.tree_builder = OneBranchTree()

        self.tool_executor = ToolExecutor(
            weaviate_client,
            llm_client,
            embedding_model=embedding_model,
        )
        self.decision_maker = DecisionMaker(model_name, llm_client=llm_client)

        # Build the LangGraph
        self.graph = self._build_graph()

    # ── Graph Construction ─────────────────────────────────────────────────

    def _build_graph(self):
        """Build the LangGraph workflow."""
        workflow = StateGraph(ElysiaState)

        # Nodes
        workflow.add_node("decision", self.node_decide)
        workflow.add_node("execute_tool", self.node_execute_tool)
        workflow.add_node("process_result", self.node_process_result)
        workflow.add_node("generate_response", self.node_generate_response)
        workflow.add_node("generate_follow_up", self.node_generate_follow_up)
        workflow.add_node("generate_title", self.node_generate_title)

        # Edges
        workflow.add_edge(START, "decision")

        workflow.add_conditional_edges(
            "decision",
            self._route_after_decision,
            {
                "execute_tool": "execute_tool",
                "generate_response": "generate_response",
                "decision": "decision",
                "end": END,
            },
        )

        workflow.add_edge("execute_tool", "process_result")

        workflow.add_conditional_edges(
            "process_result",
            self._route_after_execution,
            {
                "decision": "decision",
                "generate_response": "generate_response",
                "end": END,
            },
        )

        workflow.add_edge("generate_response", "generate_follow_up")
        workflow.add_edge("generate_follow_up", "generate_title")
        workflow.add_edge("generate_title", END)

        return workflow.compile()

    # ── Nodes ──────────────────────────────────────────────────────────────

    async def node_decide(self, state: ElysiaState) -> ElysiaState:
        """
        Decision node: mirrors Elysia's DecisionNode.__call__().
        Uses LLM to decide which action to take next, returns end_actions / impossible.
        """
        state["tree_depth"] += 1

        # ── Recursion guard ───────────────────────────────────────────────
        if state["tree_depth"] > state["max_tree_depth"]:
            state["next_action"] = "end"
            return state

        # ── Format context for LLM ────────────────────────────────────────
        decision_context = format_decision_prompt_context(state, self.tree_builder)

        # ── Call LLM / mock ───────────────────────────────────────────────
        decision = await self.decision_maker.decide(
            context=decision_context,
            previous_failures=state.get("errors", []),
        )

        action = decision.get("action")
        state["next_action"] = action
        state["reasoning"] = decision.get("reasoning", "")
        state["end_actions"] = decision.get("end_actions", False)
        state["impossible"] = decision.get("impossible", False)
        state["decision_history"].append(action or "unknown")

        # ── Branch transition detection ───────────────────────────────────
        state["is_branch_transition"] = False
        current_branch = self.tree_builder.nodes.get(state["current_branch"])

        if current_branch and action in current_branch.options:
            if current_branch.options[action].get("is_branch"):
                state["current_branch"] = action
                state["is_branch_transition"] = True

        # ── Debug log ─────────────────────────────────────────────────────
        state["messages"].append({
            "role": "decision",
            "content": json.dumps({
                "action": action,
                "reasoning": state["reasoning"],
                "end_actions": state["end_actions"],
                "impossible": state["impossible"],
                "branch": state["current_branch"],
            }),
            "timestamp": datetime.now().isoformat(),
        })

        return state

    async def node_execute_tool(self, state: ElysiaState) -> ElysiaState:
        """Execute the selected tool with appropriate parameters."""
        tool_name = state.get("next_action")

        if not tool_name:
            state["errors"].append({
                "tool_name": "execute_tool",
                "message": "No action selected",
                "timestamp": datetime.now().isoformat(),
            })
            return state

        try:
            tool_params = self._extract_tool_params(state, tool_name)
            state = await self.tool_executor.execute(
                tool_name, state, **tool_params,
            )
        except Exception as e:
            state["errors"].append({
                "tool_name": tool_name,
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            })

        return state

    async def node_process_result(self, state: ElysiaState) -> ElysiaState:
        """
        Process result node — mirrors Elysia's _evaluate_result + tree restart logic.

        Key behaviors:
        1. Logs success/failure to tasks_completed
        2. After a tool executes in a sub-branch, resets current_branch to root (tree restart)
        3. Completion is driven by end_actions / impossible, NOT by object count
        """
        last_action = state.get("next_action")

        # ── Log result ────────────────────────────────────────────────────
        recent_errors = [
            e for e in state.get("errors", [])
            if e.get("tool_name") == last_action
        ]
        last_error = recent_errors[-1] if recent_errors else None
        timestamp_match = (
            last_error
            and state["errors"]
            and last_error is state["errors"][-1]
        )

        if timestamp_match:
            error_msg = last_error.get("message", "Unknown error")
            state["previous_attempts"].append({
                "action": last_action,
                "error": error_msg,
                "timestamp": datetime.now().isoformat(),
            })
            state["tasks_completed"].append({
                "action": last_action,
                "status": "failed",
                "reason": error_msg,
            })
        else:
            state["tasks_completed"].append({
                "action": last_action,
                "status": "success",
                "details": "Executed successfully",
            })

        # ── Tree restart: reset to root (mirrors Elysia's recursive async_run) ──
        root_branch = self.tree_builder.root or "base"
        if state["current_branch"] != root_branch:
            state["current_branch"] = root_branch
            state["num_trees_completed"] = state.get("num_trees_completed", 0) + 1

        # ── Determine completion ──────────────────────────────────────────
        # Mirrors Elysia L1625-1630: completed = text_response OR end_actions OR impossible OR recursion_limit
        completed = (
            state.get("end_actions", False)
            or state.get("impossible", False)
            or state.get("num_trees_completed", 0) > state["max_tree_depth"]
        )

        if completed:
            state["next_action"] = "generate_response"
        else:
            state["next_action"] = "decide"

        return state

    async def node_generate_response(self, state: ElysiaState) -> ElysiaState:
        """
        Generate response node — mirrors Elysia's ForcedTextResponse.
        Creates the final response to the user via LLM.
        """
        response = await self.decision_maker.generate_response(
            user_prompt=state["user_prompt"],
            environment=state["environment"],
            conversation_history=state["conversation_history"],
        )

        state["final_response"] = response

        state["messages"].append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat(),
        })

        return state

    async def node_generate_follow_up(self, state: ElysiaState) -> ElysiaState:
        """Generate follow-up questions. Mirrors Elysia's FollowUpSuggestionsPrompt."""
        follow_ups = [
            "Would you like more details on this?",
            "Do you want to search another collection?",
        ]
        state["tasks_completed"].append({
            "action": "generate_follow_up",
            "status": "success",
        })
        state["hidden_environment"]["follow_ups"] = follow_ups
        return state

    async def node_generate_title(self, state: ElysiaState) -> ElysiaState:
        """Generate conversation title. Mirrors Elysia's TitleCreatorPrompt."""
        prompt = state.get("user_prompt", "")
        title = f"{prompt[:20]}..." if len(prompt) > 20 else prompt

        state["tasks_completed"].append({
            "action": "generate_title",
            "status": "success",
        })
        state["hidden_environment"]["conversation_title"] = title
        return state

    # ── Routing ────────────────────────────────────────────────────────────

    def _route_after_decision(self, state: ElysiaState) -> str:
        """
        Route after the decision node.
        Mirrors the Elysia tree traversal: branch transitions loop back to decision,
        tools go to execute_tool, and termination goes to generate_response.
        """
        # Branch transition → re-enter decision at the sub-branch
        if state.get("is_branch_transition"):
            return "decision"

        action = state.get("next_action")

        # ── Termination conditions (mirrors Elysia L1625-1630) ────────────
        completed = (
            action in ("text_response", "end")
            or state.get("end_actions", False)
            or state.get("impossible", False)
            or state["tree_depth"] > state["max_tree_depth"]
        )
        if completed:
            return "generate_response"

        # ── Executable tools ──────────────────────────────────────────────
        if action in ("query", "aggregate", "summarize"):
            return "execute_tool"

        # ── Unknown action: safety net → generate response ────────────────
        return "generate_response"

    def _route_after_execution(self, state: ElysiaState) -> str:
        """Route after tool execution and result processing."""
        if state.get("final_response"):
            return "end"

        action = state.get("next_action")

        if action in ("generate_response", "text_response"):
            return "generate_response"
        elif action in ("decide", "decision"):
            return "decision"

        # Recursion limit exceeded → force response
        if state["tree_depth"] >= state["max_tree_depth"]:
            return "generate_response"

        return "decision"

    # ── Tool Parameter Extraction ──────────────────────────────────────────

    def _extract_tool_params(self, state: ElysiaState, tool_name: str) -> dict:
        """
        Extract parameters for the tool from state.
        Hardcoded logic (user preference); future work: parse from DecisionOutput.
        """
        params = {
            "collection_names": state.get("collection_names", []),
        }
        filters = state.get("collection_metadata", {}).get("filters")

        if tool_name == "query":
            params.update({
                "search_query": state.get("user_prompt"),
                "search_type": "hybrid",
                "limit": 5,
                "filters": filters,
            })
        elif tool_name == "aggregate":
            params.update({
                "aggregations": {"count": ["COUNT"]},
                "filters": filters,
            })
        elif tool_name == "summarize":
            params.update({})

        return params

    # ── Public API ─────────────────────────────────────────────────────────

    async def run(
        self,
        user_prompt: str,
        user_id: str = "user-1",
        conversation_id: str = "conv-1",
        collection_names: Optional[list[str]] = None,
        collection_metadata: Optional[dict] = None,
        conversation_history: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Run the graph on a user prompt."""
        state = create_initial_state(
            user_prompt=user_prompt,
            user_id=user_id,
            conversation_id=conversation_id,
            collection_names=collection_names or [],
            collection_metadata=collection_metadata or {},
        )
        state["conversation_history"] = conversation_history or []

        result = await self.graph.ainvoke(state)

        return {
            "response": result.get("final_response"),
            "user_prompt": user_prompt,
            "decision_history": result.get("decision_history"),
            "retrieved_objects": self._flatten_environment(result.get("environment", {})),
            "errors": result.get("errors", []),
            "tree_depth": result.get("tree_depth"),
            "num_trees_completed": result.get("num_trees_completed", 0),
            "end_actions": result.get("end_actions", False),
            "impossible": result.get("impossible", False),
            "metadata": {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "mode": self.mode,
                "executed_at": datetime.now().isoformat(),
            },
        }

    def _flatten_environment(self, environment: dict) -> list[dict]:
        """Flatten the environment structure for output."""
        flattened = []
        for tool_name, collections in environment.items():
            for collection_name, results in collections.items():
                for result in results:
                    for obj in result.objects:
                        flattened.append({
                            "tool": tool_name,
                            "collection": collection_name,
                            "uuid": obj.uuid,
                            "properties": obj.properties,
                        })
        return flattened
