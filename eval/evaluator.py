"""Orchestrateur d'évaluation : RunResult + EvalQuestion → EvalResult.

Point d'entrée principal : ``evaluate_run()`` prend un RunResult (du battle_royale runner)
et un EvalQuestion enrichi (avec expected_answer, expected_chunks_keywords, etc.)
puis calcule toutes les métriques applicables.

Usage :
    from eval.evaluator import evaluate_run, evaluate_batch, load_eval_questions

    questions = load_eval_questions("eval_sets/sample_questions.json")
    results = [...]  # RunResult du battle_royale runner

    eval_results = evaluate_batch(results, questions, openai_client)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from loguru import logger

from eval.metrics import (
    answer_correctness,
    answer_relevance,
    faithfulness,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    source_coverage,
)


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class EvalQuestion:
    """Question d'évaluation enrichie avec les réponses de référence."""

    id: str
    question: str
    source: str | None = None
    conversation_summary: str = ""
    expected_answer: str = ""
    expected_chunks_keywords: list[str] = field(default_factory=list)
    expected_source_names: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    difficulty: str = "medium"
    notes: str = ""


@dataclass
class EvalResult:
    """Résultat d'évaluation pour un (question, agent) spécifique."""

    # Identifiants
    question_id: str
    agent_key: str
    agent_label: str
    question: str
    difficulty: str
    tags: list[str]

    # Métriques de retrieval
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    precision_at_10: float
    mrr_score: float
    ndcg_at_5: float
    ndcg_at_10: float
    source_coverage_score: float

    # Métriques de generation (LLM-as-judge)
    faithfulness_score: float
    answer_relevance_score: float
    answer_correctness_score: float

    # Méta
    latency_ms: int
    has_error: bool
    answer_length: int
    sources_count: int
    n_retrieved: int
    eval_time_ms: int  # Temps passé dans l'évaluation elle-même

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Chargement du dataset ─────────────────────────────────────────────────────


def load_eval_questions(path: str | Path) -> list[EvalQuestion]:
    """Charge un fichier JSON d'évaluation.

    Format attendu : liste d'objets avec au minimum 'id' et 'question'.
    Les champs enrichis (expected_answer, expected_chunks_keywords, etc.)
    sont optionnels.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Fichier d'évaluation introuvable : {src}")

    data = json.loads(src.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("questions", [])

    questions: list[EvalQuestion] = []
    for idx, row in enumerate(rows, start=1):
        questions.append(
            EvalQuestion(
                id=str(row.get("id") or f"q{idx}"),
                question=str(row["question"]),
                source=row.get("source"),
                conversation_summary=row.get("conversation_summary", ""),
                expected_answer=row.get("expected_answer", ""),
                expected_chunks_keywords=list(
                    row.get("expected_chunks_keywords", [])
                ),
                expected_source_names=list(row.get("expected_source_names", [])),
                tags=list(row.get("tags", [])),
                difficulty=row.get("difficulty", "medium"),
                notes=str(row.get("notes", "")),
            )
        )
    return questions


# ── Évaluation unitaire ──────────────────────────────────────────────────────


def evaluate_run(
    run_result: Any,
    eval_question: EvalQuestion,
    openai_client: Any,
    judge_model: str = "gpt-4.1-mini",
    retrieval_k_values: tuple[int, ...] = (5, 10),
) -> EvalResult:
    """Évalue un RunResult (battle_royale) contre une EvalQuestion.

    Calcule toutes les métriques de retrieval et generation.

    Args:
        run_result:     Résultat normalisé du battle_royale runner (RunResult ou dict).
        eval_question:  Question avec les références attendues.
        openai_client:  Client OpenAI pour le LLM-as-judge.
        judge_model:    Modèle OpenAI utilisé pour le judge.
        retrieval_k_values: Valeurs de K pour les métriques retrieval.

    Returns:
        EvalResult avec toutes les métriques.
    """
    eval_start = time.perf_counter()

    # Normaliser l'accès aux données (RunResult dataclass ou dict)
    if hasattr(run_result, "raw_result"):
        raw = run_result.raw_result or {}
        answer = run_result.answer
        latency = run_result.latency_ms
        has_error = run_result.has_error
        sources_count = run_result.sources_count
        agent_key = run_result.agent_key
        agent_label = run_result.agent_label
    else:
        raw = run_result
        answer = str(raw.get("answer", ""))
        latency = int(raw.get("latency_ms", 0))
        has_error = bool(raw.get("error"))
        sources_count = len(raw.get("sources", []))
        agent_key = str(raw.get("agent_key", "unknown"))
        agent_label = str(raw.get("agent_label", agent_key))

    # Récupérer les docs pour les métriques retrieval
    # Préférer retrieved_docs (avant reranking) si dispo, sinon sources (après reranking)
    retrieved_docs = raw.get("retrieved_docs") or raw.get("sources", [])
    reranked_docs = raw.get("sources", [])
    n_retrieved = len(retrieved_docs)

    # ── Métriques de retrieval ────────────────────────────────────────────
    kw = eval_question.expected_chunks_keywords
    src_names = eval_question.expected_source_names

    r_at_5 = recall_at_k(reranked_docs, kw, src_names, k=5)
    r_at_10 = recall_at_k(reranked_docs, kw, src_names, k=10)
    p_at_5 = precision_at_k(reranked_docs, kw, src_names, k=5)
    p_at_10 = precision_at_k(reranked_docs, kw, src_names, k=10)
    mrr_val = mrr(reranked_docs, kw, src_names)
    ndcg_5 = ndcg_at_k(reranked_docs, kw, src_names, k=5)
    ndcg_10 = ndcg_at_k(reranked_docs, kw, src_names, k=10)
    src_cov = source_coverage(reranked_docs, src_names)

    # ── Métriques de generation ───────────────────────────────────────────
    # Construire le contexte à partir des sources reranked
    context_parts = []
    for i, doc in enumerate(reranked_docs[:10], start=1):
        source_name = Path(doc.get("source", "")).name
        content = doc.get("page_content", "")
        context_parts.append(f"[Source {i}] {source_name}\n{content}")
    context = "\n\n".join(context_parts)

    faith = faithfulness(answer, context, eval_question.question, openai_client, judge_model)
    relevance = answer_relevance(answer, eval_question.question, openai_client, judge_model)
    correctness = answer_correctness(
        answer,
        eval_question.expected_answer,
        eval_question.question,
        openai_client,
        judge_model,
    )

    eval_time = int((time.perf_counter() - eval_start) * 1000)

    return EvalResult(
        question_id=eval_question.id,
        agent_key=agent_key,
        agent_label=agent_label,
        question=eval_question.question,
        difficulty=eval_question.difficulty,
        tags=eval_question.tags,
        recall_at_5=r_at_5,
        recall_at_10=r_at_10,
        precision_at_5=p_at_5,
        precision_at_10=p_at_10,
        mrr_score=mrr_val,
        ndcg_at_5=ndcg_5,
        ndcg_at_10=ndcg_10,
        source_coverage_score=src_cov,
        faithfulness_score=faith,
        answer_relevance_score=relevance,
        answer_correctness_score=correctness,
        latency_ms=latency,
        has_error=has_error,
        answer_length=len(answer),
        sources_count=sources_count,
        n_retrieved=n_retrieved,
        eval_time_ms=eval_time,
    )


# ── Évaluation batch ─────────────────────────────────────────────────────────


def evaluate_batch(
    run_results: list[Any],
    eval_questions: list[EvalQuestion],
    openai_client: Any,
    judge_model: str = "gpt-4.1-mini",
    progress_cb: Any | None = None,
) -> list[EvalResult]:
    """Évalue une liste de RunResult contre les EvalQuestions correspondantes.

    Matche les résultats aux questions par question_id.

    Args:
        run_results:    Liste de RunResult (du battle_royale runner).
        eval_questions: Liste de EvalQuestion (depuis le dataset).
        openai_client:  Client OpenAI.
        judge_model:    Modèle pour le LLM-as-judge.
        progress_cb:    Callback optionnel (current, total) pour le progrès.

    Returns:
        Liste de EvalResult.
    """
    # Index par question_id
    question_map = {q.id: q for q in eval_questions}

    eval_results: list[EvalResult] = []
    total = len(run_results)

    for i, run_result in enumerate(run_results):
        # Récupérer le question_id depuis le RunResult
        if hasattr(run_result, "question_id"):
            qid = run_result.question_id
        else:
            qid = str(run_result.get("question_id", ""))

        eq = question_map.get(qid)
        if eq is None:
            logger.warning("Question ID '{}' non trouvée dans le dataset d'évaluation", qid)
            continue

        logger.info(
            "[{}/{}] Évaluation : {} × {}",
            i + 1,
            total,
            qid,
            getattr(run_result, "agent_key", "unknown"),
        )

        eval_result = evaluate_run(
            run_result=run_result,
            eval_question=eq,
            openai_client=openai_client,
            judge_model=judge_model,
        )
        eval_results.append(eval_result)

        if progress_cb:
            progress_cb(i + 1, total)

    return eval_results
