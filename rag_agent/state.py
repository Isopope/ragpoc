"""État unifié du pipeline RAG — fusion de RAGState (rag_pipeline.py) et ElysiaState (langgraph_implementation).

Notes importantes :
- `seen_keys` est un `set` Python → non JSON-serializable.
  Le checkpointing LangGraph (persistance en base) est donc impossible dans cette version.
  Si nécessaire, remplacer par list[list] et adapter la logique de déduplication.
- `decision_log` est destiné à l'observabilité UI (affiché dans l'expander Streamlit).
- `tasks_completed` est destiné aux prompts de l'arbre de décision (format XML pour le LLM).
- `all_docs` est le store principal des chunks RAG ; `environment` est le store structuré ToolResult.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from typing_extensions import TypedDict


# ── Enums & Dataclasses (portés de langgraph_implementation/state.py) ─────────

class TaskStatus(Enum):
    PENDING    = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED  = "completed"
    FAILED     = "failed"


@dataclass
class RetrievedObject:
    """Un objet récupéré depuis Weaviate."""
    uuid: str
    properties: dict[str, Any]
    collection_name: str
    query_used: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ToolResult:
    """Résultat structuré d'un outil (QueryTool, AggregateTool, etc.)."""
    tool_name: str
    collection_names: list[str]
    objects: list[RetrievedObject]
    metadata: dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.COMPLETED


# ── État unifié ────────────────────────────────────────────────────────────────

class UnifiedRAGState(TypedDict):
    """État partagé propagé entre tous les nœuds LangGraph.

    Fusionne RAGState (15 champs, rag_pipeline.py) et ElysiaState (24 champs, langgraph_implementation).
    Chaque nœud reçoit l'état complet et retourne uniquement les champs modifiés.
    """

    # ── Identité & session ────────────────────────────────────────────────────
    question_id: str
    """UUID généré par requête, propagé dans tous les logs."""

    user_id: str
    """Identité appelant optionnelle. Défaut : 'anonymous'."""

    conversation_id: str
    """Identifiant de session/thread. En pratique = question_id sauf si géré externement."""

    # ── Entrée ────────────────────────────────────────────────────────────────
    question: str
    """Question brute de l'utilisateur (canonical, alias de user_prompt dans ElysiaState)."""

    available_sources: list[str]
    """Chemins complets de tous les documents indexés dans Weaviate."""

    source_filter: Optional[str]
    """Si défini, restreint toutes les recherches à ce chemin de source."""

    target_sources: list[str]
    """Documents explicitement ciblés par la planification, résolus en chemins complets."""

    conversation_summary: str
    """Résumé des tours précédents, injecté dans planning et génération (fourni par app.py)."""

    collection_metadata: dict[str, dict[str, Any]]
    """Schéma optionnel des collections Weaviate : {collection_name: {fields, length, summary}}."""

    collection_names: list[str]
    """Noms des collections Weaviate actives. Par défaut : ['RagChunk']."""

    # ── Planification ─────────────────────────────────────────────────────────
    sub_queries: list[str]
    """1-3 sous-requêtes générées par analyze_and_plan pour amorcer la boucle ReAct."""

    # ── Boucle ReAct ──────────────────────────────────────────────────────────
    messages: list[Any]
    """Historique de messages OpenAI (role/content/tool_calls). Réinitialisé par compress_context."""

    all_docs: list[dict]
    """Chunks bruts accumulés par agent_action au fil des itérations."""

    seen_keys: set  # set[tuple[str, int]]
    """Paires (source_path, chunk_index) déjà présentes dans all_docs.
    AVERTISSEMENT : non JSON-serializable → incompatible avec le checkpointing LangGraph."""

    seen_queries: list  # list[tuple[str, float]]
    """Liste (query_text, weight) des recherches déjà exécutées. Évite les doublons."""

    agent_iterations: int
    """Nombre d'itérations ReAct complétées. Stop à max_agent_iter."""

    context_summary: str
    """Résumé LLM compressé de all_docs, injecté dans agent_reason après compression."""

    # ── Navigation dans l'arbre de décision ───────────────────────────────────
    current_branch: str
    """Branche courante du RAGTree : 'plan' | 'react' | 'synthesize'."""

    decision_history: list[str]
    """Liste ordonnée des branches/actions visitées. Ex : ['plan.analyze', 'react.search']."""

    tree_depth: int
    """Nombre de transitions de branche effectuées."""

    max_tree_depth: int
    """Plafond sur tree_depth (depuis RAGConfig)."""

    is_branch_transition: bool
    """True si la dernière décision a changé de branche."""

    branch_instruction: str
    """Texte d'instruction du nœud de branche courant."""

    branch_status: str
    """Libellé de l'opération en cours (ex : 'Executing search...'). Affiché dans l'UI."""

    next_action: Optional[str]
    """Action sélectionnée par le routeur ou le LLM. Défini par les routers."""

    reasoning: str
    """Texte de raisonnement du dernier appel DecisionMaker/analyze_and_plan."""

    # ── Environnement structuré ───────────────────────────────────────────────
    environment: dict[str, dict[str, list[Any]]]
    """Store structuré des résultats outils : {tool_name: {collection: [ToolResult]}}.
    Parallèle de all_docs mais utilise le dataclass ToolResult pour les outils de type Elysia."""

    hidden_environment: dict[str, Any]
    """Stockage interne non exposé au LLM. Clés : 'follow_ups' (list), 'conversation_title' (str)."""

    tasks_completed: list[dict[str, Any]]
    """Log narratif de chaque nœud/action complété. Format : {action, status, details, timestamp}.
    Utilisé dans les prompts de l'arbre de décision (format XML pour le LLM)."""

    previous_attempts: list[dict[str, Any]]
    """Historique des exécutions d'outils échouées. Informe les décisions de retry."""

    # ── Post-boucle ReAct ─────────────────────────────────────────────────────
    retrieved_docs: list[dict]
    """Chunks finaux dédupliqués après la boucle ReAct. Produit par consolidate_chunks."""

    reranked_docs: list[dict]
    """Chunks triés par pertinence après rerank. Chaque dict gagne '_rerank_score'."""

    # ── Sortie ────────────────────────────────────────────────────────────────
    answer: str
    """Réponse finale en Markdown. Écrite par le nœud generate."""

    final_response: Optional[str]
    """Alias de answer. Maintenu pour compatibilité avec les consommateurs ElysiaState."""

    follow_up_suggestions: list[str]
    """2-3 questions de suivi générées par generate_follow_up."""

    conversation_title: Optional[str]
    """Titre court de la conversation généré par generate_title."""

    # ── Observabilité ─────────────────────────────────────────────────────────
    decision_log: list[dict]
    """Entrées structurées {step, ts, message, metadata} pour l'UI Streamlit (expander Decisions)."""

    errors: list[dict[str, Any]]
    """Erreurs non-fatales au niveau outil/nœud : {tool_name, message, timestamp}."""

    error: Optional[str]
    """Erreur fatale propagée à l'appelant. None si succès."""


# ── Factory ────────────────────────────────────────────────────────────────────

def create_unified_state(
    question: str,
    source: Optional[str] = None,
    conversation_summary: str = "",
    available_sources: Optional[list[str]] = None,
    user_id: str = "anonymous",
    conversation_id: Optional[str] = None,
    collection_metadata: Optional[dict] = None,
    collection_names: Optional[list[str]] = None,
    max_tree_depth: int = 5,
) -> UnifiedRAGState:
    """Crée un état initial avec toutes les valeurs par défaut."""
    import uuid
    qid = str(uuid.uuid4())
    return UnifiedRAGState(
        question_id=qid,
        user_id=user_id,
        conversation_id=conversation_id or qid,
        question=question,
        available_sources=available_sources or [],
        source_filter=source,
        target_sources=[source] if source else [],
        conversation_summary=conversation_summary,
        collection_metadata=collection_metadata or {},
        collection_names=collection_names or ["RagChunk"],
        sub_queries=[],
        messages=[],
        all_docs=[],
        seen_keys=set(),
        seen_queries=[],
        agent_iterations=0,
        context_summary="",
        current_branch="plan",
        decision_history=[],
        tree_depth=0,
        max_tree_depth=max_tree_depth,
        is_branch_transition=False,
        branch_instruction="",
        branch_status="",
        next_action=None,
        reasoning="",
        environment={},
        hidden_environment={},
        tasks_completed=[],
        previous_attempts=[],
        retrieved_docs=[],
        reranked_docs=[],
        answer="",
        final_response=None,
        follow_up_suggestions=[],
        conversation_title=None,
        decision_log=[],
        errors=[],
        error=None,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def log_entry(step: str, message: str, metadata: Optional[dict] = None) -> dict:
    """Crée une entrée de log structurée pour decision_log."""
    return {
        "step":     step,
        "ts":       datetime.now(timezone.utc).isoformat(),
        "message":  message,
        "metadata": metadata or {},
    }


def format_environment_for_llm(state: UnifiedRAGState) -> str:
    """Formate l'environnement structuré (ToolResult) en Markdown pour les prompts LLM."""
    env = state.get("environment", {})
    if not env:
        return "No retrieved objects yet."

    lines: list[str] = []
    for tool_name, collections in env.items():
        for coll_name, results in collections.items():
            lines.append(f"### {tool_name} → {coll_name}")
            for r in results:
                if isinstance(r, ToolResult):
                    for obj in r.objects[:3]:
                        props = obj.properties
                        snippet = str(props.get("page_content", props))[:200]
                        lines.append(f"- [{obj.uuid[:8]}] {snippet}")
                else:
                    lines.append(f"- {str(r)[:200]}")
    return "\n".join(lines)


def tasks_completed_string(state: UnifiedRAGState) -> str:
    """Formate tasks_completed en XML-like pour les prompts de l'arbre de décision."""
    tasks = state.get("tasks_completed", [])
    if not tasks:
        return ""
    parts = []
    for i, t in enumerate(tasks, start=1):
        action  = t.get("action", "unknown")
        status  = t.get("status", "completed")
        details = t.get("details", "")
        parts.append(f"<task_{i} action='{action}' status='{status}'>{details}</task_{i}>")
    return "\n".join(parts)


def add_to_environment(state: UnifiedRAGState, tool_result: ToolResult) -> None:
    """Mute state['environment'] pour ajouter un ToolResult (modification en place)."""
    tool_name = tool_result.tool_name
    env = state.setdefault("environment", {})  # type: ignore[call-overload]
    for coll in tool_result.collection_names:
        env.setdefault(tool_name, {}).setdefault(coll, []).append(tool_result)
