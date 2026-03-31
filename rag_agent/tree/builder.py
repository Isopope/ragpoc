"""Nœuds de décision et constructeur d'arbre.

Port direct de langgraph_implementation/decision_nodes.py,
adapté pour utiliser UnifiedRAGState au lieu de ElysiaState.
"""
from __future__ import annotations

from typing import Any, Optional

from ..state import UnifiedRAGState, tasks_completed_string


class DecisionNode:
    """Représente un nœud de décision dans l'arbre."""

    def __init__(
        self,
        node_id: str,
        instruction: str,
        status: str = "Processing...",
        options: Optional[dict[str, Any]] = None,
        parent_node: Optional[str] = None,
        is_root: bool = False,
    ) -> None:
        self.node_id       = node_id
        self.instruction   = instruction
        self.status        = status
        self.options       = options or {}
        self.parent_node   = parent_node
        self.is_root       = is_root
        self.visited_count = 0
        self.error_history: list[str] = []

    def to_dict(self) -> dict:
        return {
            "node_id":       self.node_id,
            "instruction":   self.instruction,
            "status":        self.status,
            "options":       self.options,
            "parent_node":   self.parent_node,
            "is_root":       self.is_root,
            "visited_count": self.visited_count,
            "error_history": self.error_history,
        }


class TreeBuilder:
    """Construit et gère la structure d'arbre avec des nœuds de décision."""

    def __init__(self) -> None:
        self.nodes:    dict[str, DecisionNode] = {}
        self.root:     Optional[str]           = None
        self.branches: dict[str, DecisionNode] = {}

    def add_branch(
        self,
        branch_id: str,
        instruction: str,
        tools: list[dict[str, str]],
        is_root: bool = False,
        parent_branch_id: Optional[str] = None,
    ) -> DecisionNode:
        """Ajoute une branche/nœud de décision à l'arbre."""
        options = {tool["name"]: tool for tool in tools}
        node = DecisionNode(
            node_id=branch_id,
            instruction=instruction,
            options=options,
            parent_node=parent_branch_id,
            is_root=is_root,
        )
        self.nodes[branch_id]    = node
        self.branches[branch_id] = node

        if is_root:
            self.root = branch_id

        if parent_branch_id and parent_branch_id in self.nodes:
            self.nodes[parent_branch_id].options[branch_id] = {
                "name":        branch_id,
                "description": f"Navigate to the {branch_id} branch",
                "inputs":      {},
                "is_branch":   True,
            }
        return node

    def add_tool_to_branch(
        self,
        branch_id: str,
        tool_name: str,
        tool_description: str,
        tool_inputs: dict[str, str],
    ) -> None:
        """Ajoute un outil à une branche existante."""
        if branch_id not in self.nodes:
            raise ValueError(f"Branch {branch_id!r} not found")
        self.nodes[branch_id].options[tool_name] = {
            "name":        tool_name,
            "description": tool_description,
            "inputs":      tool_inputs,
        }

    def get_successive_actions(self, current_branch: str) -> dict[str, Any]:
        """Retourne les actions accessibles depuis la branche courante."""
        node = self.nodes.get(current_branch)
        if not node:
            return {}
        return {
            name: {
                "description": tool.get("description", ""),
                "inputs":      tool.get("inputs", {}),
            }
            for name, tool in node.options.items()
        }

    def get_branch_structure(self) -> dict[str, Any]:
        """Retourne la structure complète de l'arbre (debug)."""
        return {
            bid: {
                "instruction": n.instruction,
                "options":     list(n.options.keys()),
                "parent":      n.parent_node,
                "is_root":     n.is_root,
            }
            for bid, n in self.branches.items()
        }


def format_decision_prompt_context(
    state: UnifiedRAGState,
    tree_builder: TreeBuilder,
) -> dict[str, Any]:
    """Formate le contexte pour le prompt LLM de décision."""
    current_branch_node = tree_builder.nodes.get(state.get("current_branch", ""))

    available_actions: dict[str, Any] = {}
    if current_branch_node:
        for tool_name, tool_info in current_branch_node.options.items():
            available_actions[tool_name] = {
                "function_name": tool_name,
                "description":   tool_info.get("description", ""),
                "inputs":        tool_info.get("inputs", {}),
            }

    previous_errors = [
        {
            "tool":          e.get("tool_name"),
            "error_message": e.get("message"),
            "timestamp":     e.get("timestamp"),
        }
        for e in (state.get("errors") or [])[-3:]
    ]

    tree_count = f"{state.get('tree_depth', 0)}/{state.get('max_tree_depth', 5)}"

    return {
        "instruction":            (
            state.get("branch_instruction", "")
            or (current_branch_node.instruction if current_branch_node else "")
        ),
        "user_prompt":            state.get("question", ""),
        "tree_count":             tree_count,
        "conversation_history":   [],  # conversation_summary est un résumé texte ici
        "available_actions":      available_actions,
        "unavailable_actions":    {},
        "successive_actions":     tree_builder.get_successive_actions(state.get("current_branch", "")),
        "previous_errors":        previous_errors,
        "retrieved_objects_summary": _format_env(state.get("environment", {})),
        "tasks_completed_summary":   tasks_completed_string(state),
        "previous_attempts":      (state.get("previous_attempts") or [])[-3:],
    }


def _format_env(environment: dict) -> str:
    if not environment:
        return "No objects retrieved yet."
    lines = ["**Retrieved Objects Summary:**"]
    for tool_name, collections in environment.items():
        lines.append(f"\n*From {tool_name}:*")
        for coll_name, results in collections.items():
            total = sum(len(getattr(r, "objects", [])) for r in results)
            lines.append(f"  - {coll_name}: {total} objects")
    return "\n".join(lines)
