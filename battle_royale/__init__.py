"""Outils d'evaluation multi-agents pour comparer plusieurs backends RAG."""

from .registry import AgentSpec, available_agents
from .runner import BattleRoyaleRunner, QuestionSpec, RunResult

__all__ = [
    "AgentSpec",
    "BattleRoyaleRunner",
    "QuestionSpec",
    "RunResult",
    "available_agents",
]
