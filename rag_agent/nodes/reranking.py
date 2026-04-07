"""Nœuds rerank — Cohere + fallback LLM.

Port de rag_pipeline.py:855-972.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Optional

from loguru import logger

from ..config import RAGConfig
from ..llm import parse_json_llm
from ..state import UnifiedRAGState, log_entry


def _cohere_rerank(
    docs: list[dict],
    question: str,
    cohere_client: Any,
) -> list[dict]:
    """Reranking Cohere rerank-multilingual-v3.0."""
    docs_texts = [(doc.get("page_content") or "")[:2048] for doc in docs]
    response   = cohere_client.rerank(
        query=question,
        documents=docs_texts,
        model="rerank-multilingual-v3.0",
        top_n=len(docs),
        return_documents=False,
    )
    scored = [{**doc} for doc in docs]
    for result in response.results:
        scored[result.index]["_rerank_score"] = result.relevance_score
    return sorted(scored, key=lambda d: float(d.get("_rerank_score", 0.0)), reverse=True)[:20]


def _llm_rerank(
    docs: list[dict],
    question: str,
    llm_call: Callable,
    llm_timeout: float,
) -> list[dict]:
    """Reranking LLM fallback — note chaque chunk de 0 à 10."""
    summaries = []
    for i, doc in enumerate(docs):
        kind   = doc.get("kind", "text")
        title  = (doc.get("title_path") or "").strip()
        text_  = (doc.get("page_content") or "")[:400].replace("\n", " ")
        extra  = " *(contexte étendu)*" if doc.get("_expanded") else ""
        summaries.append(f"[{i}] {kind}{extra} | {title} | {text_}…")

    prompt = (
        f"Question : {question}\n\n"
        "Note la pertinence de chaque extrait de 0 à 10 "
        "(10 = répond parfaitement à la question, 0 = hors sujet).\n"
        "Réponds UNIQUEMENT avec un tableau JSON compact d'entiers sur UNE SEULE LIGNE, "
        "dans l'ordre exact des extraits (sans balise markdown, sans indentation, sans saut de ligne).\n"
        f"Exemple pour {len(docs)} extraits : [{', '.join(['8'] * min(len(docs), 4))}"
        f"{'...' if len(docs) > 4 else ''}]\n\n"
        + "\n".join(summaries)
    )

    scores: list[int] = []
    try:
        resp = llm_call(
            messages=[
                {"role": "system", "content": "Tu es un expert en pertinence documentaire."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=max(600, len(docs) * 15),
        )
        text_resp = resp.choices[0].message.content or "[]"
        try:
            parsed = parse_json_llm(text_resp)
            if not isinstance(parsed, list):
                raise ValueError(f"Pas une liste JSON : {text_resp[:50]}")
            scores = [int(s) for s in parsed]
        except Exception:
            nums   = re.findall(r"\b(?:10|[0-9])\b", text_resp)
            scores = [int(n) for n in nums] if nums else []
    except TimeoutError:
        logger.warning("rerank LLM — timeout")
    except Exception as exc:
        logger.warning("rerank LLM — erreur : {}", exc)

    if not scores:
        scores = [5] * len(docs)
    if len(scores) < len(docs):
        scores += [0] * (len(docs) - len(scores))
    elif len(scores) > len(docs):
        scores = scores[: len(docs)]

    scored = [{**doc} for doc in docs]
    for i, doc in enumerate(scored):
        doc["_rerank_score"] = scores[i]
    return sorted(scored, key=lambda d: float(d.get("_rerank_score", 0.0)), reverse=True)[:20]


def rerank(
    state: UnifiedRAGState,
    *,
    llm_call: Callable,
    cohere_client: Optional[Any],
    rag_config: RAGConfig,
) -> dict:
    """Nœud 4 : reranking Cohere (si dispo) ou fallback LLM."""
    qid      = state["question_id"]
    docs     = state.get("retrieved_docs", [])
    question = state["question"]
    log      = list(state.get("decision_log", []))

    if not docs:
        log.append(log_entry("rerank", "Aucun document à reranker"))
        return {"reranked_docs": [], "decision_log": log}

    # ── Cohere ────────────────────────────────────────────────────────────────
    if rag_config.use_cohere_rerank and cohere_client is not None:
        try:
            reranked = _cohere_rerank(docs, question, cohere_client)
            log.append(log_entry(
                "rerank.cohere",
                f"Cohere Rerank : {len(docs)} → {len(reranked)} chunks",
                {"n_input": len(docs), "n_output": len(reranked)},
            ))
            return {"reranked_docs": reranked, "decision_log": log}
        except Exception as exc:
            logger.warning("[{}] Cohere Rerank échoué, fallback LLM : {}", qid, exc)
            log.append(log_entry(
                "rerank.cohere",
                f"Cohere Rerank échoué : {exc} — fallback LLM",
                {"error": str(exc)},
            ))

    # ── Fallback LLM ──────────────────────────────────────────────────────────
    reranked = _llm_rerank(docs, question, llm_call, rag_config.llm_timeout)
    log.append(log_entry(
        "rerank.llm",
        f"LLM Rerank : {len(docs)} → {len(reranked)} chunks",
        {"n_input": len(docs), "n_output": len(reranked)},
    ))
    return {"reranked_docs": reranked, "decision_log": log}
