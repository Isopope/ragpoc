"""
Main LangGraph graph implementation for the Elysia decision tree.
"""

from typing import Optional, AsyncGenerator, Any
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
        mode: str = "multibranch",  # multibranch, onebranch
        weaviate_client=None,
        llm_client=None,
        model_name: str = "gpt-4",
        embedding_model: str = "text-embedding-3-small",
    ):
        """
        Args:
            mode: Tree structure mode ('multibranch' or 'onebranch')
            weaviate_client: Optional Weaviate client for real data retrieval
            llm_client: Optional LLM client (will use LangChain by default)
            model_name: LLM model name to use
        """
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
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        workflow = StateGraph(ElysiaState)
        
        # Add nodes
        workflow.add_node("decision", self.node_decide)
        workflow.add_node("execute_tool", self.node_execute_tool)
        workflow.add_node("process_result", self.node_process_result)
        workflow.add_node("generate_response", self.node_generate_response)
        workflow.add_node("generate_follow_up", self.node_generate_follow_up)
        workflow.add_node("generate_title", self.node_generate_title)
        
        # Add edges
        workflow.add_edge(START, "decision")
        
        # Decision -> Execute tool or Generate response or Decision (if branch transition)
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
        
        # Execute tool -> Process result
        workflow.add_edge("execute_tool", "process_result")
        
        # Process result -> Decide again or End
        workflow.add_conditional_edges(
            "process_result",
            self._route_after_execution,
            {
                "decision": "decision",
                "generate_response": "generate_response",
                "end": END,
            },
        )
        
        # Generate response -> Follow up -> Title -> End
        workflow.add_edge("generate_response", "generate_follow_up")
        workflow.add_edge("generate_follow_up", "generate_title")
        workflow.add_edge("generate_title", END)
        
        return workflow.compile()
    
    async def node_decide(self, state: ElysiaState) -> ElysiaState:
        """
        Decision node: Uses LLM to decide which action to take next.
        Mirrors the DecisionPrompt logic in Elysia.
        """
        # Increment tree depth
        state["tree_depth"] += 1
        
        # Check recursion limit
        if state["tree_depth"] > state["max_tree_depth"]:
            state["next_action"] = "end"
            return state
        
        # Format context for decision-making
        decision_context = format_decision_prompt_context(state, self.tree_builder)
        
        # Use LLM to make decision
        decision = await self.decision_maker.decide(
            context=decision_context,
            previous_failures=state.get("errors", []),
        )
        
        state["next_action"] = decision.get("action")
        state["reasoning"] = decision.get("reasoning", "")
        state["decision_history"].append(decision.get("action", "unknown"))
        
        # Check if the chosen action is a branch transition
        action = state.get("next_action")
        current_branch = self.tree_builder.nodes.get(state["current_branch"])
        
        state["is_branch_transition"] = False
        if current_branch and action in current_branch.options:
            if current_branch.options[action].get("is_branch"):
                state["current_branch"] = action
                state["is_branch_transition"] = True
        
        # Store the decision for debugging
        state["messages"].append({
            "role": "decision",
            "content": f"Chose action: {decision.get('action')}",
            "timestamp": datetime.now().isoformat(),
        })
        
        return state
    
    async def node_execute_tool(self, state: ElysiaState) -> ElysiaState:
        """
        Execute node: Runs the selected tool with appropriate parameters.
        """
        tool_name = state.get("next_action")
        
        if not tool_name:
            state["errors"].append({
                "tool_name": "execute_tool",
                "message": "No action selected",
                "timestamp": datetime.now().isoformat(),
            })
            return state
        
        try:
            # Extract tool parameters from reasoning or context
            tool_params = self._extract_tool_params(state, tool_name)
            
            # Execute tool
            state = await self.tool_executor.execute(
                tool_name,
                state,
                **tool_params,
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
        Process result node: Evaluates the tool execution result.
        Determines whether to continue, end, or try different approach.
        """
        # Check if there were errors
        if state.get("errors") and state["errors"][-1].get("tool_name") == state.get("next_action"):
            # Tool execution failed
            error_msg = state["errors"][-1].get("message")
            state["previous_attempts"].append({
                "action": state.get("next_action"),
                "error": error_msg,
                "timestamp": datetime.now().isoformat(),
            })
            state["tasks_completed"].append({
                "action": state.get("next_action"),
                "status": "failed",
                "reason": error_msg,
            })
        else:
            state["tasks_completed"].append({
                "action": state.get("next_action"),
                "status": "success",
                "details": "Executed successfully",
            })
        
        # Determine next step
        if len(state["environment"]) > 0:
            # We have retrieved data
            num_objects = sum(
                sum(len(r.objects) for r in results)
                for tool_results in state["environment"].values()
                for results in tool_results.values()
            )
            
            if num_objects > 3:
                # Have enough data to generate response
                state["next_action"] = "generate_response"
            else:
                # Need more data, make another decision
                state["next_action"] = "decide"
        else:
            # No data yet, continue searching
            state["next_action"] = "decide"
        
        return state
    
    async def node_generate_response(self, state: ElysiaState) -> ElysiaState:
        """
        Generate response node: Creates the final response to the user.
        Uses LLM to synthesize retrieved information.
        """
        # Generate response from environment
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
        """
        Generate follow up questions node.
        Mirrors Elysia's FollowUpSuggestionsPrompt processing.
        """
        # In a full implementation, this would call DecisionMaker with the FollowUpPrompt.
        # Here we add a placeholder for structural completeness.
        follow_ups = ["Would you like more details on this?", "Do you want to search another collection?"]
        state["tasks_completed"].append({
            "action": "generate_follow_up",
            "status": "success"
        })
        state["hidden_environment"]["follow_ups"] = follow_ups
        return state

    async def node_generate_title(self, state: ElysiaState) -> ElysiaState:
        """
        Generate conversation title node.
        Mirrors Elysia's TitleCreatorPrompt processing.
        """
        # In a full implementation, this uses LLM to generate a title.
        prompt = state.get("user_prompt", "")
        title = f"{prompt[:20]}..." if len(prompt) > 20 else prompt
        
        state["tasks_completed"].append({
            "action": "generate_title",
            "status": "success"
        })
        state["hidden_environment"]["conversation_title"] = title
        return state

    
    def _route_after_decision(self, state: ElysiaState) -> str:
        """Route based on decision node output."""
        if state.get("is_branch_transition"):
            return "decision"
            
        action = state.get("next_action")
        
        if action == "end" or state["tree_depth"] > state["max_tree_depth"]:
            return "end"
        elif action in ["query", "aggregate", "summarize"]:
            return "execute_tool"
        elif action in ["generate_response", "text_response"]:
            return "generate_response"
        else:
            # Unknown action, end
            return "end"
    
    def _route_after_execution(self, state: ElysiaState) -> str:
        """Route after tool execution."""
        if state.get("final_response"):
            return "end"
            
        action = state.get("next_action")
        
        if action == "generate_response" or action == "text_response":
            return "generate_response"
        elif action == "decide" or action == "decision":
            return "decision"
        
        if state["tree_depth"] >= state["max_tree_depth"]:
            return "generate_response"
        
        # Make another decision
        return "decision"
    
    def _extract_tool_params(self, state: ElysiaState, tool_name: str) -> dict:
        """
        Extract parameters for the tool from state.
        In a real system, this would parse the LLM reasoning/output.
        """
        params = {
            "collection_names": state.get("collection_names", []),
        }
        filters = state.get("collection_metadata", {}).get("filters")
        
        if tool_name == "query":
            # Extract search parameters
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
    
    async def run(
        self,
        user_prompt: str,
        user_id: str = "user-1",
        conversation_id: str = "conv-1",
        collection_names: Optional[list[str]] = None,
        collection_metadata: Optional[dict] = None,
        conversation_history: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """
        Run the graph on a user prompt.
        
        Args:
            user_prompt: The user's input query
            user_id: User identifier
            conversation_id: Conversation identifier
            collection_names: Available collection names
            collection_metadata: Metadata about collections
        
        Returns:
            Final result with response and metadata
        """
        # Create initial state
        state = create_initial_state(
            user_prompt=user_prompt,
            user_id=user_id,
            conversation_id=conversation_id,
            collection_names=collection_names or [],
            collection_metadata=collection_metadata or {},
        )
        state["conversation_history"] = conversation_history or []
        
        # Run the graph
        result = await self.graph.ainvoke(state)
        
        return {
            "response": result.get("final_response"),
            "user_prompt": user_prompt,
            "decision_history": result.get("decision_history"),
            "retrieved_objects": self._flatten_environment(result.get("environment", {})),
            "errors": result.get("errors", []),
            "tree_depth": result.get("tree_depth"),
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
