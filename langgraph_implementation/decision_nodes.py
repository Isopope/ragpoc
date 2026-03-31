"""
Decision node logic for the Elysia LangGraph implementation.
Implements the DecisionPrompt and decision-making logic.
"""

from typing import Optional, Any
import json
from datetime import datetime
from .state import ElysiaState, ToolResult, RetrievedObject, tasks_completed_string


class DecisionNode:
    """
    Represents a decision node in the Elysia tree.
    Similar to the DecisionNode class in Elysia.
    """
    
    def __init__(
        self,
        node_id: str,
        instruction: str,
        status: str = "Processing...",
        options: Optional[dict[str, Any]] = None,
        parent_node: Optional[str] = None,
        is_root: bool = False,
    ):
        """
        Args:
            node_id: Unique identifier for this node
            instruction: Instruction for the decision-making LLM
            status: Current status message
            options: Available actions/options at this node
            parent_node: ID of parent node
            is_root: Whether this is the root node
        """
        self.node_id = node_id
        self.instruction = instruction
        self.status = status
        self.options = options or {}
        self.parent_node = parent_node
        self.is_root = is_root
        self.visited_count = 0
        self.error_history = []
    
    def to_dict(self) -> dict:
        """Convert to dictionary for state management."""
        return {
            "node_id": self.node_id,
            "instruction": self.instruction,
            "status": self.status,
            "options": self.options,
            "parent_node": self.parent_node,
            "is_root": self.is_root,
            "visited_count": self.visited_count,
            "error_history": self.error_history,
        }


class TreeBuilder:
    """Builds and manages the tree structure with decision nodes."""
    
    def __init__(self):
        self.nodes: dict[str, DecisionNode] = {}
        self.root: Optional[str] = None
        self.branches: dict[str, DecisionNode] = {}
    
    def add_branch(
        self,
        branch_id: str,
        instruction: str,
        tools: list[dict[str, str]],
        is_root: bool = False,
        parent_branch_id: Optional[str] = None,
    ) -> DecisionNode:
        """
        Add a decision branch/node to the tree.
        
        Args:
            branch_id: Unique identifier for this branch
            instruction: Instructions for decision-making at this branch
            tools: List of available tools with their descriptions
            is_root: Whether this is the root branch
            parent_branch_id: Parent branch ID
        
        Returns:
            The created DecisionNode
        """
        options = {tool["name"]: tool for tool in tools}
        node = DecisionNode(
            node_id=branch_id,
            instruction=instruction,
            options=options,
            parent_node=parent_branch_id,
            is_root=is_root,
        )
        
        self.nodes[branch_id] = node
        self.branches[branch_id] = node
        
        if is_root:
            self.root = branch_id
        
        if parent_branch_id and parent_branch_id in self.nodes:
            # Add this branch as an option in the parent branch
            self.nodes[parent_branch_id].options[branch_id] = {
                "name": branch_id,
                "description": f"Navigate to the {branch_id} branch",
                "inputs": {},
                "is_branch": True
            }
        
        return node
    
    def add_tool_to_branch(
        self,
        branch_id: str,
        tool_name: str,
        tool_description: str,
        tool_inputs: dict[str, str],
    ) -> None:
        """Add a tool to an existing branch."""
        if branch_id not in self.nodes:
            raise ValueError(f"Branch {branch_id} not found")
        
        self.nodes[branch_id].options[tool_name] = {
            "name": tool_name,
            "description": tool_description,
            "inputs": tool_inputs,
        }
    
    def get_successive_actions(self, current_branch: str) -> dict[str, Any]:
        """
        Get the tree of actions that can follow from the current branch.
        Similar to the successive_actions in DecisionPrompt.
        """
        current_node = self.nodes.get(current_branch)
        if not current_node:
            return {}
        
        return {
            tool_name: {
                "description": tool["description"],
                "inputs": tool.get("inputs", {}),
            }
            for tool_name, tool in current_node.options.items()
        }
    
    def get_branch_structure(self) -> dict[str, Any]:
        """Get the complete tree structure for display/debugging."""
        structure = {}
        for branch_id, node in self.branches.items():
            structure[branch_id] = {
                "instruction": node.instruction,
                "options": list(node.options.keys()),
                "parent": node.parent_node,
                "is_root": node.is_root,
            }
        return structure


def format_decision_prompt_context(
    state: ElysiaState,
    tree_builder: TreeBuilder,
) -> dict[str, Any]:
    """
    Format the context for the decision-making LLM prompt.
    Mirrors the inputs to DecisionPrompt in Elysia.
    """
    current_branch = tree_builder.nodes.get(state["current_branch"])
    
    # Format available actions
    available_actions = {}
    if current_branch:
        for tool_name, tool_info in current_branch.options.items():
            available_actions[tool_name] = {
                "function_name": tool_name,
                "description": tool_info.get("description", ""),
                "inputs": tool_info.get("inputs", {}),
            }
    
    # Format error history for learning
    previous_errors = []
    for error in state["errors"][-3:]:  # Keep last 3 errors
        previous_errors.append({
            "tool": error.get("tool_name"),
            "error_message": error.get("message"),
            "timestamp": error.get("timestamp"),
        })
    
    # Tree count as "current/max"
    tree_count = f"{state['tree_depth']}/{state['max_tree_depth']}"
    
    return {
        "instruction": state.get("branch_instruction", current_branch.instruction if current_branch else ""),
        "user_prompt": state["user_prompt"],
        "tree_count": tree_count,
        "conversation_history": state["conversation_history"][-5:] if state["conversation_history"] else [],
        "available_actions": available_actions,
        "unavailable_actions": state.get("unavailable_actions", {}),
        "successive_actions": tree_builder.get_successive_actions(state["current_branch"]),
        "previous_errors": previous_errors,
        "retrieved_objects_summary": format_environment_for_llm(state["environment"]),
        "tasks_completed_summary": tasks_completed_string(state),
        "previous_attempts": state.get("previous_attempts", [])[-3:],  # Last 3 attempts
    }


def format_environment_for_llm(environment: dict) -> str:
    """Format environment for LLM context."""
    if not environment:
        return "No objects retrieved yet."
    
    summary = "**Retrieved Objects Summary:**\n"
    for tool_name, collections in environment.items():
        summary += f"\n*From {tool_name}:*\n"
        for collection_name, results in collections.items():
            total_objs = sum(len(r.objects) for r in results)
            summary += f"  - {collection_name}: {total_objs} objects\n"
    
    return summary


class MultibranchTree(TreeBuilder):
    """
    Multi-branch tree configuration similar to Elysia's multi_branch_init.
    """
    
    def __init__(self):
        super().__init__()
        self._build_multibranch_structure()
    
    def _build_multibranch_structure(self):
        """Build the multi-branch tree structure."""
        # Root branch: Base decision
        self.add_branch(
            branch_id="base",
            instruction="""
            Choose a base-level task based on the user's prompt and available information.
            You can search, which includes aggregating or querying information - this should be used if the user needs (more) information.
            You can end the conversation by choosing text response, or summarise some retrieved information.
            Base your decision on what information is available and what the user is asking for.
            """,
            tools=[
                {"name": "summarize", "description": "Summarize retrieved information"},
                {"name": "text_response", "description": "Generate a text response"},
            ],
            is_root=True,
        )
        
        # Search branch: Query vs Aggregate
        self.add_branch(
            branch_id="search",
            instruction="""
            Choose between querying the knowledge base via semantic/keyword search, or aggregating information.
            - Querying: For specific information related to dataset content, requiring a specific search query
            - Aggregating: For summary statistics, counting, averaging, or other statistical operations
            """,
            tools=[
                {"name": "query", "description": "Semantic/keyword search on knowledge base"},
                {"name": "aggregate", "description": "Perform aggregation operations"},
            ],
            is_root=False,
            parent_branch_id="base",
        )


class OneBranchTree(TreeBuilder):
    """
    Single-branch tree configuration similar to Elysia's one_branch_init.
    All tools available at the root level.
    """
    
    def __init__(self):
        super().__init__()
        self._build_onebranch_structure()
    
    def _build_onebranch_structure(self):
        """Build the single-branch tree structure."""
        self.add_branch(
            branch_id="base",
            instruction="""
            Choose a task based on the user's prompt and available information.
            Decide based on the tools you have available as well as their descriptions.
            """,
            tools=[
                {"name": "query", "description": "Search the knowledge base"},
                {"name": "aggregate", "description": "Perform aggregation on the knowledge base"},
                {"name": "summarize", "description": "Summarize retrieved information"},
                {"name": "text_response", "description": "Generate a text response"},
                {"name": "visualize", "description": "Visualize data"},
            ],
            is_root=True,
        )
