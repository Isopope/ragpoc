"""Test de performance : parallel_initial_retrieval avec rapport détaillé."""
import os
import time
import json

from dotenv import load_dotenv

load_dotenv()

from weaviate_store import WeaviateStore
from rag_pipeline import RAGAgent

store = WeaviateStore(host="localhost", port=8080)
store.connect()

agent = RAGAgent(
    store,
    os.getenv("OPENAI_API_KEY"),
    embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
    llm_model=os.getenv("LLM_MODEL", "gpt-4.1"),
    max_agent_iter=10,
    llm_timeout=30.0,
)

question = "Quels sont les matieres du semestre 8 ?"

start = time.perf_counter()
result = agent.query(question)
total_ms = int((time.perf_counter() - start) * 1000)

# Build the report
report_lines = []
report_lines.append("=" * 70)
report_lines.append("RAPPORT DE TEST - PARALLEL INITIAL RETRIEVAL")
report_lines.append("=" * 70)
report_lines.append(f"Question: {question}")
report_lines.append(f"Latence totale: {total_ms}ms")
report_lines.append(f"Chunks retrieves: {result.get('n_retrieved', 0)}")
report_lines.append(f"Chunks rerankees: {len(result.get('sources', []))}")
report_lines.append(f"Longueur reponse: {len(result.get('answer', ''))} chars")
report_lines.append(f"Erreur: {result.get('error')}")
report_lines.append("")
report_lines.append("--- DECISION LOG ---")

for entry in result.get("decision_log", []):
    step = entry.get("step", "")
    msg = entry.get("message", "")
    meta = entry.get("metadata", {})
    
    # Highlight parallel_retrieval entries
    prefix = ">>>" if "parallel" in step.lower() else "   "
    report_lines.append(f"{prefix} [{step}] {msg}")
    
    # Show metadata for important steps
    if meta and step in ("parallel_retrieval", "analyze"):
        for k, v in meta.items():
            if k != "errors" or v:
                report_lines.append(f"       {k}: {v}")

report_lines.append("")
report_lines.append("--- REPONSE (debut) ---")
report_lines.append(result.get("answer", "")[:500])
report_lines.append("...")
report_lines.append("")

report = "\n".join(report_lines)
print(report)

# Save to file
with open("test_parallel_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

store.close()
print("Report saved to test_parallel_report.txt")
