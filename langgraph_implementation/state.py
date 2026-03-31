"""
State definitions for the Elysia LangGraph implementation.
Mirrors the TreeData and Environment structure from Elysia.
"""

from typing import Any, TypedDict, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    """Possible statuses for tasks."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RetrievedObject:
    """Represents a retrieved object from a collection."""
    uuid: str
    properties: dict[str, Any]
    collection_name: str
    query_used: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool_name: str
    collection_names: list[str]
    objects: list[RetrievedObject]
    metadata: dict[str, Any]
    status: TaskStatus = TaskStatus.COMPLETED


class ElysiaState(TypedDict):
    """
    Main state dictionary for the Elysia decision tree in LangGraph.
    Combines TreeData and Environment concepts.
    """
    # User/Session Information
    user_id: str
    conversation_id: str
    user_prompt: str
    
    # Conversation History
    conversation_history: list[dict[str, Any]]  # [{role: str, content: str, timestamp: str}, ...]
    
    # Decision Tree State
    current_branch: str  # Current branch node identifier
    decision_history: list[str]  # History of decisions made
    tree_depth: int  # Current depth in the tree
    max_tree_depth: int  # Maximum allowed depth (recursion limit)
    
    # Available Tools & Actions
    available_actions: dict[str, dict[str, Any]]  # Actions available at current node
    unavailable_actions: dict[str, dict[str, str]]  # Actions not yet available
    successive_actions: dict[str, Any]  # Tree of possible future actions
    
    # Environment (Retrieved Data)
    environment: dict[str, dict[str, list[ToolResult]]]  # {tool_name: {result_name: [results]}}
    hidden_environment: dict[str, Any]  # Internal storage, not shown to LLM
    
    # Collection Information
    collection_metadata: dict[str, dict[str, Any]]  # Schema and metadata for each collection
    collection_names: list[str]  # Available collections
    
    # Error Tracking
    errors: list[dict[str, Any]]  # Track errors for feedback loops
    previous_attempts: list[dict[str, Any]]  # Previous failed attempts
    
    # Agent State
    messages: list[dict[str, str]]  # Formatted messages for LLM
    tasks_completed: list[dict[str, Any]]  # Narrative log of completed tasks
    next_action: Optional[str]  # Which action to take next
    reasoning: str  # LLM reasoning for the decision
    
    # Response
    final_response: Optional[str]  # The final response to the user
    
    # Metadata
    branch_instruction: str  # Instruction for current branch
    branch_status: str  # Status message for current operation
    is_branch_transition: bool  # Whether the current decision is navigating to a sub-branch


def create_initial_state(
    user_prompt: str,
    user_id: str,
    conversation_id: str,
    collection_names: list[str],
    collection_metadata: Optional[dict] = None,
) -> ElysiaState:
    """Create an initial state for a new conversation/tree execution."""
    return ElysiaState(
        user_id=user_id,
        conversation_id=conversation_id,
        user_prompt=user_prompt,
        conversation_history=[],
        current_branch="base",
        decision_history=[],
        tree_depth=0,
        max_tree_depth=5,
        available_actions={},
        unavailable_actions={},
        successive_actions={},
        environment={},
        hidden_environment={},
        collection_metadata=collection_metadata or {},
        collection_names=collection_names,
        errors=[],
        previous_attempts=[],
        messages=[],
        tasks_completed=[],
        next_action=None,
        reasoning="",
        final_response=None,
        branch_instruction="",
        branch_status="",
        is_branch_transition=False,
    )


def add_to_environment(
    state: ElysiaState,
    tool_name: str,
    result: ToolResult,
) -> ElysiaState:
    """Add a tool result to the environment."""
    if tool_name not in state["environment"]:
        state["environment"][tool_name] = {}
    
    collection_key = result.collection_names[0] if result.collection_names else "unknown"
    
    if collection_key not in state["environment"][tool_name]:
        state["environment"][tool_name][collection_key] = []
    
    state["environment"][tool_name][collection_key].append(result)
    return state


def format_environment_for_llm(state: ElysiaState) -> str:
    """Format the environment dictionary into readable text for LLM context."""
    if not state["environment"]:
        return "No retrieved objects yet."
    
    formatted = "## Retrieved Objects\n"
    for tool_name, results_dict in state["environment"].items():
        formatted += f"\n### From {tool_name}:\n"
        for collection_name, results in results_dict.items():
            formatted += f"**Collection: {collection_name}**\n"
            for result in results:
                formatted += f"- Objects retrieved: {len(result.objects)}\n"
                formatted += f"  Query: {result.metadata.get('query', 'N/A')}\n"
    
    return formatted

def tasks_completed_string(state: ElysiaState) -> str:
    """Format tasks completed into a readable string for the LLM."""
    if not state.get("tasks_completed"):
        return "No tasks completed yet."
    
    out = ""
    for i, task in enumerate(state["tasks_completed"]):
        out += f"<task_{i+1}>\n"
        for k, v in task.items():
            out += f"{k.capitalize()}: {v}\n"
        out += f"</task_{i+1}>\n"
    return out

