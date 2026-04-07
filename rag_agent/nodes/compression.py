"""Nœud compress_context — compresse le contexte accumulé.

Port de rag_pipeline.py:755-806.
"""
from __future__ import annotations

from typing import Callable

from loguru import logger

from ..config import RAGConfig
from ..state import UnifiedRAGState, log_entry
from .generation import _build_context_entry


_COMPRESSION_PROMPT = """Tu es un expert en compression de contexte de recherche.

Ta tâche est de condenser le contenu récupéré en un résumé concis, axé sur la question, directement utilisable par un agent RAG pour continuer ou finaliser sa réponse.

Règles :
1. Conserve UNIQUEMENT les informations pertinentes pour répondre à la question de l'utilisateur.
2. Préserve les chiffres, noms, versions, termes techniques et configurations exacts.
3. Supprime les doublons, détails non pertinents ou administratifs.
4. N'inclus pas les requêtes de recherche, IDs de chunks ni identifiants internes.
5. Organise les résultats par fichier source. Chaque section DOIT commencer par : ### nom_fichier.pdf
6. Signale les informations manquantes dans une section « Lacunes ».
7. Limite à environ 400-600 mots. Priorité aux faits critiques et données structurées.
8. Produis uniquement du contenu Markdown structuré, sans explications.

Structure requise :
# Résumé du Contexte de Recherche

## Focalisation
[Reformulation technique brève de la question]

## Résultats Structurés

### nom_fichier.pdf
- Faits directement pertinents
- Contexte de soutien (si nécessaire)

## Lacunes
- Aspects manquants ou incomplets"""


def compress_context(state: UnifiedRAGState, *, llm_call: Callable, rag_config: RAGConfig) -> dict:
    """Compresse les docs récupérés quand le budget token est dépassé.

    Réinitialise messages = [] pour que agent_reason reparte avec la summary injectée.
    """
    qid      = state["question_id"]
    log      = list(state.get("decision_log", []))
    all_docs = state.get("all_docs", [])
    question = state["question"]
    existing = state.get("context_summary", "")

    content_parts = [
        _build_context_entry(i, doc)
        for i, doc in enumerate(all_docs, start=1)
    ]
    raw_content = "\n\n".join(content_parts)

    compress_input = (
        f"RÉSUMÉ EXISTANT :\n{existing}\n\nNOUVEAU CONTENU :\n{raw_content}"
        if existing
        else raw_content
    )

    context_summary = existing
    try:
        resp = llm_call(
            messages=[
                {"role": "system", "content": _COMPRESSION_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question : {question}\n\n"
                        f"Contenu à compresser :\n{compress_input[:40_000]}"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=1200,
        )
        context_summary = resp.choices[0].message.content or existing
        log.append(log_entry(
            "compress",
            f"Contexte compressé : {len(raw_content)} → {len(context_summary)} chars",
            {"raw_chars": len(raw_content), "summary_chars": len(context_summary)},
        ))
    except Exception as exc:
        logger.warning("[{}] compress_context — erreur : {}", qid, exc)
        log.append(log_entry("compress", f"Compression échouée : {exc}"))

    # Réinitialise messages : agent_reason repart avec la summary injectée dans le prompt initial
    return {"context_summary": context_summary, "messages": [], "decision_log": log}
