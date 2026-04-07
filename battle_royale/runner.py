"""Runner commun pour executer plusieurs agents sur les memes questions."""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

from .registry import AgentSpec, available_agents


@dataclass
class QuestionSpec:
    """Question d'evaluation."""

    id: str
    question: str
    source: str | None = None
    conversation_summary: str = ""
    expected_elements: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class RunResult:
    """Sortie normalisee d'une execution d'agent."""

    run_id: str
    question_id: str
    agent_key: str
    agent_label: str
    status: str
    latency_ms: int
    answer: str
    answer_chars: int
    sources_count: int
    decision_steps_count: int
    has_error: bool
    error: str | None
    has_sources_section: bool
    matched_expected_elements: int
    expected_elements_total: int
    source_filter: str | None
    question: str
    conversation_summary: str
    timestamp_utc: str
    raw_result: dict[str, Any] = field(default_factory=dict)


class BattleRoyaleRunner:
    """Orchestre l'execution comparee de plusieurs agents."""

    def __init__(
        self,
        *,
        weaviate_store,
        openai_key: str,
        cohere_key: str | None = None,
        llm_model: str = "gpt-4.1",
        embedding_model: str = "text-embedding-3-small",
        top_k_retrieve: int = 20,
        top_k_final: int = 5,
        hybrid_alpha: float = 0.5,
        max_tokens: int = 1000,
        max_agent_iter: int = 60,
        llm_timeout: float = 30.0,
    ) -> None:
        self._shared_kwargs = {
            "weaviate_store": weaviate_store,
            "openai_key": openai_key,
            "cohere_key": cohere_key,
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "top_k_retrieve": top_k_retrieve,
            "top_k_final": top_k_final,
            "hybrid_alpha": hybrid_alpha,
            "max_tokens": max_tokens,
            "max_agent_iter": max_agent_iter,
            "llm_timeout": llm_timeout,
        }

    def build_agents(self, selected: Iterable[str] | None = None) -> list[tuple[AgentSpec, Any]]:
        """Construit la liste des agents a executer."""
        allowed = set(selected or [])
        specs = available_agents()
        built: list[tuple[AgentSpec, Any]] = []
        for spec in specs:
            if allowed and spec.key not in allowed:
                continue
            built.append((spec, spec.factory(**self._shared_kwargs)))
        return built

    def run(
        self,
        questions: Iterable[QuestionSpec],
        *,
        selected_agents: Iterable[str] | None = None,
    ) -> list[RunResult]:
        """Execute tous les agents selectionnes sur toutes les questions."""
        agents = self.build_agents(selected_agents)
        results: list[RunResult] = []
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        for question in questions:
            for spec, agent in agents:
                results.append(self._run_single(run_id, question, spec, agent))
        return results

    def _run_single(self, run_id: str, question: QuestionSpec, spec: AgentSpec, agent: Any) -> RunResult:
        started = time.perf_counter()
        raw_result: dict[str, Any] = {}
        status = "ok"
        error: str | None = None

        try:
            raw_result = self._call_agent(agent, question)
        except Exception as exc:
            status = "error"
            error = str(exc)
            raw_result = {
                "answer": "",
                "sources": [],
                "decision_log": [],
                "error": error,
            }

        latency_ms = int((time.perf_counter() - started) * 1000)
        answer = str(raw_result.get("answer") or "")
        sources = raw_result.get("sources") or []
        decision_log = raw_result.get("decision_log") or []
        matched_expected = _count_expected_elements(answer, question.expected_elements)

        return RunResult(
            run_id=run_id,
            question_id=question.id,
            agent_key=spec.key,
            agent_label=spec.label,
            status=status,
            latency_ms=latency_ms,
            answer=answer,
            answer_chars=len(answer),
            sources_count=len(sources),
            decision_steps_count=len(decision_log),
            has_error=bool(error or raw_result.get("error")),
            error=error or raw_result.get("error"),
            has_sources_section="**Sources" in answer or "Sources :" in answer,
            matched_expected_elements=matched_expected,
            expected_elements_total=len(question.expected_elements),
            source_filter=question.source,
            question=question.question,
            conversation_summary=question.conversation_summary,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            raw_result=raw_result,
        )

    @staticmethod
    def _call_agent(agent: Any, question: QuestionSpec) -> dict[str, Any]:
        """Appelle chaque backend avec une signature tolerante."""
        try:
            return agent.query(
                question.question,
                source=question.source,
                conversation_summary=question.conversation_summary,
            )
        except TypeError:
            return agent.query(
                question.question,
                source=question.source,
            )


def load_questions(path: str | Path) -> list[QuestionSpec]:
    """Charge un fichier JSON ou JSONL de questions."""
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Fichier de questions introuvable: {src}")

    rows: list[dict[str, Any]] = []
    if src.suffix.lower() == ".jsonl":
        with src.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        data = json.loads(src.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("questions", [])

    questions: list[QuestionSpec] = []
    for idx, row in enumerate(rows, start=1):
        questions.append(
            QuestionSpec(
                id=str(row.get("id") or f"q{idx}"),
                question=str(row["question"]),
                source=row.get("source"),
                conversation_summary=row.get("conversation_summary", ""),
                expected_elements=list(row.get("expected_elements", [])),
                tags=list(row.get("tags", [])),
                notes=str(row.get("notes", "")),
            )
        )
    return questions


def save_results(results: list[RunResult], output_dir: str | Path) -> Path:
    """Ecrit les sorties normalisees en JSONL, CSV et resume Markdown."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if not results:
        raise ValueError("Aucun resultat a sauvegarder")

    run_id = results[0].run_id
    jsonl_path = target_dir / f"battle_royale_{run_id}.jsonl"
    csv_path = target_dir / f"battle_royale_{run_id}.csv"
    md_path = target_dir / f"battle_royale_{run_id}.md"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    rows = [_flatten_result(result) for result in results]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path.write_text(_build_markdown_summary(results), encoding="utf-8")
    return jsonl_path


def make_runner_from_env(*, weaviate_store) -> BattleRoyaleRunner:
    """Construit le runner a partir des variables d'environnement du projet."""
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY est requis pour lancer le battle royale.")

    return BattleRoyaleRunner(
        weaviate_store=weaviate_store,
        openai_key=openai_key,
        cohere_key=os.getenv("COHERE_API_KEY") or None,
        llm_model=os.getenv("LLM_MODEL", "gpt-4.1"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        top_k_retrieve=int(os.getenv("TOP_K_RETRIEVE", "20")),
        top_k_final=int(os.getenv("TOP_K_FINAL", "5")),
        hybrid_alpha=float(os.getenv("HYBRID_ALPHA", "0.5")),
        max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
        max_agent_iter=int(os.getenv("MAX_AGENT_ITER", "60")),
        llm_timeout=float(os.getenv("LLM_TIMEOUT", "30.0")),
    )


def _count_expected_elements(answer: str, expected_elements: list[str]) -> int:
    lowered = answer.lower()
    return sum(1 for item in expected_elements if item.lower() in lowered)


def _flatten_result(result: RunResult) -> dict[str, Any]:
    row = asdict(result)
    row["raw_result"] = json.dumps(result.raw_result, ensure_ascii=False)
    return row


def _build_markdown_summary(results: list[RunResult]) -> str:
    by_agent: dict[str, list[RunResult]] = {}
    for result in results:
        by_agent.setdefault(result.agent_key, []).append(result)

    lines = [
        "# Battle Royale RAG",
        "",
        f"- Run ID: `{results[0].run_id}`",
        f"- Questions: `{len({r.question_id for r in results})}`",
        f"- Agents: `{len(by_agent)}`",
        "",
        "## Scoreboard",
        "",
        "| Agent | Runs | Errors | Avg latency (ms) | Avg sources | Expected hits |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for agent_key, agent_results in by_agent.items():
        runs = len(agent_results)
        errors = sum(1 for r in agent_results if r.has_error)
        avg_latency = sum(r.latency_ms for r in agent_results) // runs
        avg_sources = sum(r.sources_count for r in agent_results) / runs
        expected_hits = sum(r.matched_expected_elements for r in agent_results)
        expected_total = sum(r.expected_elements_total for r in agent_results)
        lines.append(
            f"| `{agent_key}` | {runs} | {errors} | {avg_latency} | {avg_sources:.1f} | {expected_hits}/{expected_total} |"
        )

    lines.extend(["", "## Details", ""])
    for result in results:
        lines.extend(
            [
                f"### {result.question_id} - {result.agent_key}",
                "",
                f"- Status: `{result.status}`",
                f"- Latency: `{result.latency_ms} ms`",
                f"- Sources: `{result.sources_count}`",
                f"- Decision steps: `{result.decision_steps_count}`",
                f"- Expected hits: `{result.matched_expected_elements}/{result.expected_elements_total}`",
                "",
                "**Question**",
                "",
                result.question,
                "",
                "**Answer**",
                "",
                result.answer or "_No answer_",
                "",
            ]
        )
        if result.error:
            lines.extend(["**Error**", "", result.error, ""])

    return "\n".join(lines)
