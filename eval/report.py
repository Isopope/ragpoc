"""Génération de rapports d'évaluation — Markdown, CSV, et console.

Usage :
    from eval.report import save_eval_report

    save_eval_report(eval_results, output_dir="eval_runs")
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.evaluator import EvalResult


# ══════════════════════════════════════════════════════════════════════════════
#  SAUVEGARDE
# ══════════════════════════════════════════════════════════════════════════════


def save_eval_report(
    eval_results: list[EvalResult],
    output_dir: str | Path,
    run_id: str | None = None,
) -> Path:
    """Sauvegarde les résultats d'évaluation en JSONL, CSV, et Markdown.

    Returns:
        Path du fichier Markdown généré.
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if not eval_results:
        raise ValueError("Aucun résultat d'évaluation à sauvegarder")

    rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # ── JSONL ─────────────────────────────────────────────────────────────
    jsonl_path = target_dir / f"eval_{rid}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for r in eval_results:
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    # ── CSV ───────────────────────────────────────────────────────────────
    csv_path = target_dir / f"eval_{rid}.csv"
    rows = [_flatten_eval_result(r) for r in eval_results]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ── Markdown ──────────────────────────────────────────────────────────
    md_path = target_dir / f"eval_{rid}.md"
    md_path.write_text(build_markdown_report(eval_results, rid), encoding="utf-8")

    return md_path


# ══════════════════════════════════════════════════════════════════════════════
#  RAPPORT MARKDOWN
# ══════════════════════════════════════════════════════════════════════════════


def build_markdown_report(
    eval_results: list[EvalResult],
    run_id: str = "",
) -> str:
    """Génère un rapport Markdown complet avec agrégats et détails."""
    lines: list[str] = []

    # ── En-tête ───────────────────────────────────────────────────────────
    lines.extend([
        "# 📊 Rapport d'Évaluation RAG",
        "",
        f"- **Run ID** : `{run_id}`",
        f"- **Date** : `{datetime.now(timezone.utc).isoformat()}`",
        f"- **Questions** : `{len({r.question_id for r in eval_results})}`",
        f"- **Agents** : `{len({r.agent_key for r in eval_results})}`",
        "",
    ])

    # ── Scoreboard par agent ──────────────────────────────────────────────
    by_agent = _group_by(eval_results, "agent_key")
    lines.extend([
        "## 🏆 Scoreboard",
        "",
        _build_scoreboard_table(by_agent),
        "",
    ])

    # ── Détails retrieval par agent ───────────────────────────────────────
    lines.extend([
        "## 🔍 Métriques de Retrieval",
        "",
        _build_retrieval_table(by_agent),
        "",
    ])

    # ── Détails generation par agent ──────────────────────────────────────
    lines.extend([
        "## 💬 Métriques de Génération (LLM-as-judge)",
        "",
        _build_generation_table(by_agent),
        "",
    ])

    # ── Résultats par difficulté ──────────────────────────────────────────
    by_difficulty = _group_by(eval_results, "difficulty")
    if len(by_difficulty) > 1:
        lines.extend([
            "## 📈 Par Difficulté",
            "",
            _build_difficulty_table(by_difficulty),
            "",
        ])

    # ── Détails par question ──────────────────────────────────────────────
    lines.extend([
        "## 📝 Détails par Question",
        "",
    ])

    by_question = _group_by(eval_results, "question_id")
    for qid, q_results in by_question.items():
        first = q_results[0]
        lines.extend([
            f"### {qid}",
            "",
            f"**Question** : {first.question}",
            f"**Difficulté** : `{first.difficulty}` | **Tags** : {', '.join(f'`{t}`' for t in first.tags)}",
            "",
        ])

        # Tableau comparatif des agents pour cette question
        lines.append(
            "| Agent | Recall@5 | P@5 | MRR | NDCG@5 | Src Cov | Faith | Relev | Correct | Latency |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        for r in q_results:
            lines.append(
                f"| `{r.agent_key}` "
                f"| {_fmt(r.recall_at_5)} "
                f"| {_fmt(r.precision_at_5)} "
                f"| {_fmt(r.mrr_score)} "
                f"| {_fmt(r.ndcg_at_5)} "
                f"| {_fmt(r.source_coverage_score)} "
                f"| {_fmt(r.faithfulness_score)} "
                f"| {_fmt(r.answer_relevance_score)} "
                f"| {_fmt(r.answer_correctness_score)} "
                f"| {r.latency_ms}ms |"
            )
        lines.append("")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSOLE
# ══════════════════════════════════════════════════════════════════════════════


def print_summary(eval_results: list[EvalResult]) -> None:
    """Affiche un résumé compact dans la console."""
    by_agent = _group_by(eval_results, "agent_key")

    print("\n" + "=" * 80)
    print("  📊 RÉSUMÉ D'ÉVALUATION RAG")
    print("=" * 80)

    for agent_key, results in by_agent.items():
        n = len(results)
        avg_recall = _avg([r.recall_at_5 for r in results])
        avg_precision = _avg([r.precision_at_5 for r in results])
        avg_mrr = _avg([r.mrr_score for r in results])
        avg_faith = _avg_valid([r.faithfulness_score for r in results])
        avg_relev = _avg_valid([r.answer_relevance_score for r in results])
        avg_correct = _avg_valid([r.answer_correctness_score for r in results])
        avg_latency = sum(r.latency_ms for r in results) // n

        print(f"\n  🤖 {agent_key} ({n} questions)")
        print(f"     Retrieval  : Recall@5={avg_recall:.2f}  P@5={avg_precision:.2f}  MRR={avg_mrr:.2f}")
        print(f"     Generation : Faith={avg_faith:.2f}  Relev={avg_relev:.2f}  Correct={avg_correct:.2f}")
        print(f"     Perf       : Avg latency={avg_latency}ms")

    print("\n" + "=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _group_by(results: list[EvalResult], key: str) -> dict[str, list[EvalResult]]:
    groups: dict[str, list[EvalResult]] = {}
    for r in results:
        k = getattr(r, key)
        groups.setdefault(k, []).append(r)
    return groups


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _avg_valid(values: list[float]) -> float:
    """Moyenne en excluant les valeurs -1.0 (erreurs LLM)."""
    valid = [v for v in values if v >= 0.0]
    return sum(valid) / len(valid) if valid else -1.0


def _fmt(value: float) -> str:
    """Formate un score pour le Markdown."""
    if value < 0:
        return "N/A"
    return f"{value:.2f}"


def _build_scoreboard_table(by_agent: dict[str, list[EvalResult]]) -> str:
    lines = [
        "| Agent | Questions | Avg Recall@5 | Avg P@5 | Avg MRR | Avg Faithfulness | Avg Relevance | Avg Correctness | Avg Latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for agent_key, results in by_agent.items():
        n = len(results)
        lines.append(
            f"| `{agent_key}` "
            f"| {n} "
            f"| {_fmt(_avg([r.recall_at_5 for r in results]))} "
            f"| {_fmt(_avg([r.precision_at_5 for r in results]))} "
            f"| {_fmt(_avg([r.mrr_score for r in results]))} "
            f"| {_fmt(_avg_valid([r.faithfulness_score for r in results]))} "
            f"| {_fmt(_avg_valid([r.answer_relevance_score for r in results]))} "
            f"| {_fmt(_avg_valid([r.answer_correctness_score for r in results]))} "
            f"| {sum(r.latency_ms for r in results) // n}ms |"
        )
    return "\n".join(lines)


def _build_retrieval_table(by_agent: dict[str, list[EvalResult]]) -> str:
    lines = [
        "| Agent | Recall@5 | Recall@10 | P@5 | P@10 | MRR | NDCG@5 | NDCG@10 | Src Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for agent_key, results in by_agent.items():
        lines.append(
            f"| `{agent_key}` "
            f"| {_fmt(_avg([r.recall_at_5 for r in results]))} "
            f"| {_fmt(_avg([r.recall_at_10 for r in results]))} "
            f"| {_fmt(_avg([r.precision_at_5 for r in results]))} "
            f"| {_fmt(_avg([r.precision_at_10 for r in results]))} "
            f"| {_fmt(_avg([r.mrr_score for r in results]))} "
            f"| {_fmt(_avg([r.ndcg_at_5 for r in results]))} "
            f"| {_fmt(_avg([r.ndcg_at_10 for r in results]))} "
            f"| {_fmt(_avg([r.source_coverage_score for r in results]))} |"
        )
    return "\n".join(lines)


def _build_generation_table(by_agent: dict[str, list[EvalResult]]) -> str:
    lines = [
        "| Agent | Faithfulness | Relevance | Correctness | Avg Answer Length |",
        "|---|---:|---:|---:|---:|",
    ]
    for agent_key, results in by_agent.items():
        lines.append(
            f"| `{agent_key}` "
            f"| {_fmt(_avg_valid([r.faithfulness_score for r in results]))} "
            f"| {_fmt(_avg_valid([r.answer_relevance_score for r in results]))} "
            f"| {_fmt(_avg_valid([r.answer_correctness_score for r in results]))} "
            f"| {sum(r.answer_length for r in results) // len(results)} chars |"
        )
    return "\n".join(lines)


def _build_difficulty_table(by_difficulty: dict[str, list[EvalResult]]) -> str:
    lines = [
        "| Difficulté | N | Avg Recall@5 | Avg Faith | Avg Relev | Avg Correct |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for diff, results in sorted(by_difficulty.items()):
        lines.append(
            f"| `{diff}` "
            f"| {len(results)} "
            f"| {_fmt(_avg([r.recall_at_5 for r in results]))} "
            f"| {_fmt(_avg_valid([r.faithfulness_score for r in results]))} "
            f"| {_fmt(_avg_valid([r.answer_relevance_score for r in results]))} "
            f"| {_fmt(_avg_valid([r.answer_correctness_score for r in results]))} |"
        )
    return "\n".join(lines)


def _flatten_eval_result(result: EvalResult) -> dict[str, Any]:
    row = asdict(result)
    row["tags"] = json.dumps(result.tags, ensure_ascii=False)
    return row
