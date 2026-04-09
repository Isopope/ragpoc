"""Framework d'évaluation pour le RAG POC.

Modules :
  - metrics    : Fonctions de métriques pures (retrieval + generation LLM-as-judge)
  - evaluator  : Orchestrateur qui calcule toutes les métriques pour un RunResult
  - report     : Génération de rapports Markdown / CSV
"""
