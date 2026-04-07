"""Nœuds generate, generate_follow_up, generate_title.

Port de rag_pipeline.py:975-1023 + nouveaux nœuds de langgraph_implementation.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from loguru import logger

from ..config import RAGConfig
from ..llm import parse_json_llm
from ..state import UnifiedRAGState, log_entry


_SYSTEM_PROMPT = """Tu es un assistant expert, précis et bienveillant.

Ta tâche est de générer une réponse complète et structurée basée UNIQUEMENT sur les extraits fournis.

Règles strictes :
1. Utilise UNIQUEMENT les informations présentes dans les extraits fournis.
2. Si l'information demandée n'est pas dans les extraits, dis-le explicitement.
3. Préserve les chiffres, versions, termes techniques et détails exacts.
4. Rédige en français, dans un style clair et professionnel.
5. Ne conclus pas avec des remarques finales, notes, avis ou répétitions après la section Sources.
   La section Sources est toujours le dernier élément de ta réponse.

Mise en forme :
- Utilise le Markdown (titres, gras, listes) pour la lisibilité.
- Rédige en paragraphes fluides quand c'est possible.
- Conclus par une section Sources comme décrit ci-dessous.

Règles pour la section Sources :
- Inclure "---\\n**Sources :**\\n" à la fin, suivi d'une liste à puces des noms de fichiers.
- Lister UNIQUEMENT les entrées ayant une vraie extension de fichier (.pdf, .docx, .txt…).
- Dédupliquer : si le même fichier apparaît plusieurs fois, le lister une seule fois.
- Si aucun nom de fichier valide n'est présent, omettre la section Sources.
- LA SECTION SOURCES EST LA DERNIÈRE CHOSE QUE TU ÉCRIS."""


def _build_context_entry(index: int, doc: dict) -> str:
    """Formate un chunk pour le prompt de génération."""
    source_name = Path(doc.get("source", "inconnu")).name
    title_path  = (doc.get("title_path") or "").strip()
    content     = (doc.get("page_content") or "").strip()
    kind        = doc.get("kind", "text")
    expanded    = " (contexte étendu)" if doc.get("_expanded") else ""

    header = f"[Source {index}] {source_name}"
    if title_path:
        header += f" — {title_path}"
    header += f" [{kind}{expanded}]"
    return f"{header}\n{content}"


def generate(state: UnifiedRAGState, *, llm_call: Callable, rag_config: RAGConfig) -> dict:
    """Nœud 5 : génère la réponse finale à partir des chunks rerankés."""
    qid      = state["question_id"]
    docs     = state.get("reranked_docs", [])
    question = state["question"]
    log      = list(state.get("decision_log", []))

    if not docs:
        answer = "Aucun extrait pertinent n'a été trouvé pour répondre à votre question."
        log.append(log_entry("generate", "Aucun document disponible"))
        return {"answer": answer, "final_response": answer, "error": None, "decision_log": log}

    context = "\n\n".join(_build_context_entry(i, doc) for i, doc in enumerate(docs, start=1))

    user_content = f"Contexte :\n{context}\n\nQuestion : {question}"
    if state.get("conversation_summary"):
        user_content = (
            f"Contexte de la conversation précédente :\n{state['conversation_summary']}\n\n"
            + user_content
        )

    try:
        resp = llm_call(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            temperature=0.1,
            max_tokens=rag_config.max_tokens,
            timeout=rag_config.llm_timeout * 2,
        )
        answer = resp.choices[0].message.content or ""
        if not answer:
            reason = resp.choices[0].finish_reason or "UNKNOWN"
            raise RuntimeError(f"Réponse vide (finish_reason: {reason})")
    except Exception as exc:
        msg = f"Erreur de génération : {exc}"
        logger.error("[{}] generate — {}", qid, exc)
        log.append(log_entry("generate", msg, {"error": str(exc)}))
        return {"answer": msg, "final_response": msg, "error": msg, "decision_log": log}

    log.append(log_entry(
        "generate",
        f"Réponse produite ({len(answer)} caractères)",
        {"n_chars": len(answer), "n_sources": len(docs)},
    ))
    return {"answer": answer, "final_response": answer, "error": None, "decision_log": log}


def generate_follow_up(state: UnifiedRAGState, *, llm_call: Callable, rag_config: RAGConfig) -> dict:
    """Génère 2-3 questions de suivi pertinentes.

    Nouveau nœud inspiré de langgraph_implementation.
    """
    if not rag_config.use_follow_up:
        return {"follow_up_suggestions": [], "hidden_environment": state.get("hidden_environment", {})}

    question = state["question"]
    answer   = state.get("answer", "")
    log      = list(state.get("decision_log", []))

    suggestions: list[str] = []
    try:
        resp = llm_call(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un assistant qui génère des questions de suivi pertinentes. "
                        "Réponds UNIQUEMENT avec un tableau JSON de 2-3 chaînes de caractères. "
                        "Exemple : [\"Question 1 ?\", \"Question 2 ?\"]"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question initiale : {question}\n\n"
                        f"Réponse fournie (extrait) : {answer[:800]}\n\n"
                        "Génère 2-3 questions de suivi pertinentes en français."
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=300,
        )
        raw    = resp.choices[0].message.content or "[]"
        parsed = parse_json_llm(raw)
        if isinstance(parsed, list):
            suggestions = [str(q) for q in parsed[:3]]
    except Exception as exc:
        logger.warning("generate_follow_up — {}", exc)

    hidden = dict(state.get("hidden_environment") or {})
    hidden["follow_ups"] = suggestions
    log.append(log_entry("follow_up", f"{len(suggestions)} suggestions générées"))
    return {
        "follow_up_suggestions": suggestions,
        "hidden_environment":    hidden,
        "decision_log":          log,
    }


def generate_title(state: UnifiedRAGState, *, llm_call: Callable, rag_config: RAGConfig) -> dict:
    """Génère un titre court pour la conversation.

    Nouveau nœud inspiré de langgraph_implementation.
    """
    if not rag_config.use_title_generation:
        return {"conversation_title": None, "hidden_environment": state.get("hidden_environment", {})}

    question = state["question"]
    answer   = state.get("answer", "")
    log      = list(state.get("decision_log", []))

    title: Optional[str] = None  # type: ignore[assignment]
    try:
        resp = llm_call(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu génères des titres courts (≤60 caractères) pour des conversations. "
                        "Réponds UNIQUEMENT avec le titre, sans guillemets ni ponctuation finale."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question : {question[:200]}\n"
                        f"Réponse (extrait) : {answer[:300]}\n\n"
                        "Génère un titre court en français."
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=60,
        )
        raw   = (resp.choices[0].message.content or "").strip()
        title = raw[:60] if raw else None
    except Exception as exc:
        logger.warning("generate_title — {}", exc)
        title = question[:50] + ("…" if len(question) > 50 else "")

    if not title:
        title = question[:50] + ("…" if len(question) > 50 else "")

    hidden = dict(state.get("hidden_environment") or {})
    hidden["conversation_title"] = title
    log.append(log_entry("title", f"Titre généré : {title!r}"))
    return {
        "conversation_title":  title,
        "hidden_environment":  hidden,
        "decision_log":        log,
    }
