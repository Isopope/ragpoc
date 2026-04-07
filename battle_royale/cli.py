"""CLI pour lancer des comparaisons multi-agents depuis le terminal."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from weaviate_store import WeaviateStore

from .registry import available_agents
from .runner import load_questions, make_runner_from_env, save_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lance un battle royale entre plusieurs agents RAG.")
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
    parser.add_argument("--weaviate-host", default=None, help="Override WEAVIATE_HOST.")
    parser.add_argument("--weaviate-port", type=int, default=None, help="Override WEAVIATE_PORT.")
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="Affiche les agents disponibles et quitte.",
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

    store = WeaviateStore(host=host, port=port)
    store.connect()
    try:
        runner = make_runner_from_env(weaviate_store=store)
        results = runner.run(questions, selected_agents=args.agents)
        jsonl_path = save_results(results, args.output_dir)
    finally:
        store.close()

    print(f"Battle royale termine. Resultats ecrits dans: {Path(jsonl_path).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
