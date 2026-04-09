"""Métriques d'évaluation pour le pipeline RAG.

Deux familles de métriques :
  1. **Retrieval** — évaluent la qualité de la récupération de chunks (purement algorithmique)
  2. **Generation** — évaluent la qualité de la réponse finale (LLM-as-judge via OpenAI)

Toutes les fonctions sont pures et stateless (pas d'effets de bord).
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from loguru import logger


# ══════════════════════════════════════════════════════════════════════════════
#  MÉTRIQUES DE RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════


def _extract_chunk_keys(docs: list[dict]) -> list[tuple[str, int]]:
    """Extrait les clés (source_name, chunk_index) d'une liste de chunks.

    Normalise le source en basename pour la comparaison.
    """
    keys: list[tuple[str, int]] = []
    for doc in docs:
        source = Path(doc.get("source", "")).name.lower()
        chunk_index = int(doc.get("chunk_index", -1))
        keys.append((source, chunk_index))
    return keys


def _match_by_keywords(
    docs: list[dict],
    expected_keywords: list[str],
    expected_source_names: list[str],
) -> list[bool]:
    """Détermine pour chaque chunk s'il est 'pertinent' selon les mots-clés attendus.

    Un chunk est jugé pertinent s'il :
      - Contient au moins un mot-clé attendu dans son page_content OU
      - Provient d'un des fichiers sources attendus

    Cette approche est un proxy quand on n'a pas de chunk_index de référence.
    """
    relevance: list[bool] = []
    normalized_sources = {s.lower() for s in expected_source_names}
    lowered_keywords = [kw.lower() for kw in expected_keywords]

    for doc in docs:
        content = (doc.get("page_content") or "").lower()
        title = (doc.get("title_path") or "").lower()
        source_name = Path(doc.get("source", "")).name.lower()

        # Condition 1 : provient d'une source attendue
        source_match = source_name in normalized_sources if normalized_sources else False

        # Condition 2 : contient au moins un mot-clé attendu
        keyword_match = any(kw in content or kw in title for kw in lowered_keywords)

        relevance.append(source_match or keyword_match)

    return relevance


def recall_at_k(
    retrieved_docs: list[dict],
    expected_keywords: list[str],
    expected_source_names: list[str],
    k: int = 10,
) -> float:
    """Calcule le Recall@K : proportion de 'documents pertinents' retrouvés dans le top-K.

    Utilise un matching par mots-clés + sources car on n'a pas de chunk_index de référence.

    Returns:
        float entre 0.0 et 1.0. 0.0 si aucun mot-clé/source attendu.
    """
    if not expected_keywords and not expected_source_names:
        return 0.0  # Pas de référence → question hors-sujet, recall non applicable

    top_k = retrieved_docs[:k]
    relevance = _match_by_keywords(top_k, expected_keywords, expected_source_names)
    relevant_count = sum(relevance)

    # On estime le nombre total de documents pertinents comme le nombre de sources attendues
    # (au minimum 1 pour éviter la division par zéro)
    total_relevant = max(len(expected_source_names), 1)

    return min(relevant_count / total_relevant, 1.0)


def precision_at_k(
    retrieved_docs: list[dict],
    expected_keywords: list[str],
    expected_source_names: list[str],
    k: int = 10,
) -> float:
    """Calcule la Precision@K : proportion de chunks du top-K qui sont pertinents.

    Returns:
        float entre 0.0 et 1.0.
    """
    top_k = retrieved_docs[:k]
    if not top_k:
        return 0.0

    relevance = _match_by_keywords(top_k, expected_keywords, expected_source_names)
    return sum(relevance) / len(top_k)


def mrr(
    retrieved_docs: list[dict],
    expected_keywords: list[str],
    expected_source_names: list[str],
) -> float:
    """Mean Reciprocal Rank : 1/rang du premier chunk pertinent.

    Returns:
        float entre 0.0 et 1.0. 0.0 si aucun chunk pertinent trouvé.
    """
    relevance = _match_by_keywords(
        retrieved_docs, expected_keywords, expected_source_names
    )
    for i, is_relevant in enumerate(relevance):
        if is_relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    retrieved_docs: list[dict],
    expected_keywords: list[str],
    expected_source_names: list[str],
    k: int = 10,
) -> float:
    """Normalized Discounted Cumulative Gain @ K.

    Mesure la qualité de l'ordre : les docs pertinents doivent être en haut.

    Returns:
        float entre 0.0 et 1.0. 0.0 si aucun chunk pertinent.
    """
    top_k = retrieved_docs[:k]
    relevance = _match_by_keywords(top_k, expected_keywords, expected_source_names)

    # DCG = Σ rel_i / log2(i + 2)  (i est 0-indexed)
    dcg = sum(
        (1.0 if rel else 0.0) / math.log2(i + 2)
        for i, rel in enumerate(relevance)
    )

    # iDCG = meilleur DCG possible (tous les pertinents en premier)
    n_relevant = sum(relevance)
    if n_relevant == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))

    return dcg / idcg if idcg > 0 else 0.0


def source_coverage(
    retrieved_docs: list[dict],
    expected_source_names: list[str],
) -> float:
    """Proportion des sources attendues qui apparaissent dans les résultats.

    Returns:
        float entre 0.0 et 1.0. 0.0 si pas de sources attendues.
    """
    if not expected_source_names:
        return 0.0

    retrieved_sources = {
        Path(doc.get("source", "")).name.lower() for doc in retrieved_docs
    }
    expected_lower = {s.lower() for s in expected_source_names}

    covered = retrieved_sources & expected_lower
    return len(covered) / len(expected_lower)


# ══════════════════════════════════════════════════════════════════════════════
#  MÉTRIQUES DE GENERATION (LLM-as-judge)
# ══════════════════════════════════════════════════════════════════════════════


_FAITHFULNESS_PROMPT = """\
Tu es un évaluateur objectif de systèmes RAG.

Évalue si la réponse de l'assistant est **fidèle** au contexte fourni.
Une réponse fidèle n'affirme que des faits présents dans le contexte.
Une réponse non fidèle invente des informations absentes du contexte.

**Contexte fourni :**
{context}

**Question :**
{question}

**Réponse de l'assistant :**
{answer}

Donne un score entre 0.0 et 1.0 :
- 1.0 = Toutes les affirmations sont soutenues par le contexte
- 0.5 = Certaines affirmations ne sont pas vérifiables dans le contexte
- 0.0 = La réponse invente largement ou contredit le contexte

Réponds UNIQUEMENT avec un nombre décimal (ex: 0.85). Rien d'autre."""

_RELEVANCE_PROMPT = """\
Tu es un évaluateur objectif de systèmes RAG.

Évalue si la réponse de l'assistant **répond effectivement** à la question posée.

**Question :**
{question}

**Réponse de l'assistant :**
{answer}

Donne un score entre 0.0 et 1.0 :
- 1.0 = La réponse traite directement et complètement la question
- 0.5 = La réponse est partiellement pertinente ou incomplète
- 0.0 = La réponse est hors sujet ou ne répond pas du tout

Réponds UNIQUEMENT avec un nombre décimal (ex: 0.85). Rien d'autre."""

_CORRECTNESS_PROMPT = """\
Tu es un évaluateur objectif de systèmes RAG.

Compare la réponse de l'assistant avec la réponse de référence.
Évalue la **correction factuelle** : les informations fournies sont-elles exactes ?

**Question :**
{question}

**Réponse de référence :**
{expected_answer}

**Réponse de l'assistant :**
{answer}

Donne un score entre 0.0 et 1.0 :
- 1.0 = La réponse est factuellement identique à la référence
- 0.5 = La réponse contient certains éléments corrects mais omet ou altère des faits
- 0.0 = La réponse est factuellement incorrecte ou n'a rien en commun avec la référence

Réponds UNIQUEMENT avec un nombre décimal (ex: 0.85). Rien d'autre."""


def _parse_llm_score(text: str) -> float:
    """Extrait un score flottant d'une réponse LLM.

    Cherche le premier nombre décimal dans la réponse.
    Retourne 0.0 en cas d'échec de parsing.
    """
    text = text.strip()
    # Essai direct
    try:
        score = float(text)
        return max(0.0, min(1.0, score))
    except ValueError:
        pass

    # Chercher un nombre dans le texte
    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        score = float(match.group(1))
        return max(0.0, min(1.0, score))

    return 0.0


def _call_judge(
    prompt: str,
    openai_client: Any,
    model: str = "gpt-4.1-mini",
) -> float:
    """Appelle le LLM-as-judge et retourne un score.

    En cas d'erreur, retourne -1.0 pour indiquer un échec.
    """
    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Tu es un évaluateur objectif et précis."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        content = resp.choices[0].message.content or ""
        return _parse_llm_score(content)
    except Exception as exc:
        logger.warning("LLM-as-judge error: {}", exc)
        return -1.0


def faithfulness(
    answer: str,
    context: str,
    question: str,
    openai_client: Any,
    model: str = "gpt-4.1-mini",
) -> float:
    """Évalue la fidélité de la réponse au contexte fourni (0.0 - 1.0).

    -1.0 si l'appel LLM échoue.
    """
    prompt = _FAITHFULNESS_PROMPT.format(
        context=context[:8000],
        question=question,
        answer=answer[:4000],
    )
    return _call_judge(prompt, openai_client, model)


def answer_relevance(
    answer: str,
    question: str,
    openai_client: Any,
    model: str = "gpt-4.1-mini",
) -> float:
    """Évalue si la réponse est pertinente par rapport à la question (0.0 - 1.0).

    -1.0 si l'appel LLM échoue.
    """
    prompt = _RELEVANCE_PROMPT.format(
        question=question,
        answer=answer[:4000],
    )
    return _call_judge(prompt, openai_client, model)


def answer_correctness(
    answer: str,
    expected_answer: str,
    question: str,
    openai_client: Any,
    model: str = "gpt-4.1-mini",
) -> float:
    """Évalue la correction factuelle vs la réponse de référence (0.0 - 1.0).

    -1.0 si l'appel LLM échoue.
    """
    if not expected_answer:
        return -1.0  # Pas de référence → métrique non applicable

    prompt = _CORRECTNESS_PROMPT.format(
        question=question,
        expected_answer=expected_answer,
        answer=answer[:4000],
    )
    return _call_judge(prompt, openai_client, model)
