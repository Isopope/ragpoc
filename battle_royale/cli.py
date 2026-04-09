"""CLI pour lancer des comparaisons multi-agents depuis le terminal.

Usage :
    # Battle royale classique (sans évaluation)
    python -m battle_royale.cli --agents rag_pipeline rag_agent

    # Avec évaluation automatique
    python -m battle_royale.cli --evaluate --questions eval_sets/sample_questions.json

    # Lister les agents disponibles
    python -m battle_royale.cli --list-agents
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from weaviate_store import WeaviateStore

from .registry import available_agents
from .runner import load_questions, make_runner_from_env, save_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lance un battle royale entre plusieurs agents RAG."
    )
    parser.add_argument(
        "--questions",
        default="eval_sets/sample_questions.json",
        help="Chemin vers un fichier JSON ou JSONL de questions.",
    )
    parser.add_argument(
        "--agents",
        nargs="*",
        default=None,
        help="Liste des agents a executer. Par defaut: tous.",
    )
    parser.add_argument(
        "--output-dir",
        default="battle_royale_runs",
        help="Dossier de sortie pour les resultats.",
    )
    parser.add_argument(
        "--weaviate-host", default=None, help="Override WEAVIATE_HOST."
    )
    parser.add_argument(
        "--weaviate-port", type=int, default=None, help="Override WEAVIATE_PORT."
    )
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="Affiche les agents disponibles et quitte.",
    )

    # ── Options d'évaluation ──────────────────────────────────────────────
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Lance l'évaluation automatique (métriques retrieval + LLM-as-judge) après l'exécution.",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4.1-mini",
        help="Modèle OpenAI pour le LLM-as-judge (défaut: gpt-4.1-mini).",
    )
    parser.add_argument(
        "--eval-only",
        type=str,
        default=None,
        help="Chemin vers un fichier JSONL de résultats existants à évaluer (sans relancer les agents).",
    )
    return parser


def main() -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    if args.list_agents:
        for spec in available_agents():
            print(f"{spec.key:12} {spec.label} - {spec.description}")
        return 0

    questions = load_questions(args.questions)
    host = args.weaviate_host or os.getenv("WEAVIATE_HOST", "localhost")
    port = args.weaviate_port or int(os.getenv("WEAVIATE_PORT", "8080"))

    # ── Mode eval-only : évaluer un fichier de résultats existant ─────────
    if args.eval_only:
        return _run_eval_only(args)

    store = WeaviateStore(host=host, port=port)
    store.connect()
    try:
        runner = make_runner_from_env(weaviate_store=store)
        results = runner.run(questions, selected_agents=args.agents)
        jsonl_path = save_results(results, args.output_dir)

        print(f"\nBattle royale terminé. Résultats écrits dans: {Path(jsonl_path).resolve()}")

        # ── Évaluation optionnelle ────────────────────────────────────────
        if args.evaluate:
            _run_evaluation(results, args)

    finally:
        store.close()

    return 0


def _run_evaluation(results: list, args: argparse.Namespace) -> None:
    """Exécute l'évaluation sur les résultats d'un run."""
    from openai import OpenAI

    from eval.evaluator import evaluate_batch, load_eval_questions
    from eval.report import print_summary, save_eval_report

    print("\n📊 Lancement de l'évaluation...")

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        print("⚠️  OPENAI_API_KEY requis pour le LLM-as-judge. Évaluation annulée.")
        return

    client = OpenAI(api_key=openai_key)
    eval_questions = load_eval_questions(args.questions)

    eval_results = evaluate_batch(
        run_results=results,
        eval_questions=eval_questions,
        openai_client=client,
        judge_model=args.judge_model,
    )

    if eval_results:
        # Sauvegarder le rapport
        md_path = save_eval_report(eval_results, args.output_dir)
        print(f"\n✅ Rapport d'évaluation : {md_path.resolve()}")

        # Afficher le résumé en console
        print_summary(eval_results)
    else:
        print("⚠️  Aucun résultat d'évaluation produit.")


def _run_eval_only(args: argparse.Namespace) -> int:
    """Évalue un fichier JSONL de résultats existants sans relancer les agents."""
    import json
    from dataclasses import fields

    from openai import OpenAI

    from eval.evaluator import evaluate_batch, load_eval_questions
    from eval.report import print_summary, save_eval_report
    from .runner import RunResult

    eval_path = Path(args.eval_only)
    if not eval_path.exists():
        print(f"❌ Fichier introuvable : {eval_path}")
        return 1

    print(f"📂 Chargement des résultats : {eval_path}")
    results_dicts: list[dict] = []
    with eval_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                results_dicts.append(json.loads(line))

    # Reconstruire les RunResult depuis le JSONL
    run_result_fields = {f.name for f in fields(RunResult)}
    results: list[RunResult] = []
    for rd in results_dicts:
        # Filtrer les clés inconnues
        filtered = {k: v for k, v in rd.items() if k in run_result_fields}
        results.append(RunResult(**filtered))

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        print("⚠️  OPENAI_API_KEY requis pour le LLM-as-judge.")
        return 1

    client = OpenAI(api_key=openai_key)
    eval_questions = load_eval_questions(args.questions)

    eval_results = evaluate_batch(
        run_results=results,
        eval_questions=eval_questions,
        openai_client=client,
        judge_model=args.judge_model,
    )

    if eval_results:
        md_path = save_eval_report(eval_results, args.output_dir)
        print(f"\n✅ Rapport d'évaluation : {md_path.resolve()}")
        print_summary(eval_results)
    else:
        print("⚠️  Aucun résultat d'évaluation produit.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
