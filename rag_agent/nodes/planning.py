"""Nœud analyze_and_plan — décompose la question en sous-requêtes.

Port de rag_pipeline.py:402-475 avec validation Pydantic via PlanningOutput.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from ..config import RAGConfig
from ..llm import PlanningOutput, parse_json_llm
from ..state import UnifiedRAGState, log_entry


def _build_planning_prompt(
    question: str,
    sources_meta: list[dict],
    conv_ctx: str,
) -> str:
    def _fmt(m: dict) -> str:
        name = Path(m["source"]).name
        e = (m.get("entity") or "").strip()
        return f"{name} [{e}]" if e else name

    source_names = ", ".join(_fmt(m) for m in sources_meta) if sources_meta else "Aucun"
    entities = sorted({(m.get("entity") or "").strip() for m in sources_meta if (m.get("entity") or "").strip()})
    entities_line = f"\nEntités disponibles : {', '.join(entities)}" if entities else ""
    return (
        f"Tu es un expert en analyse de requêtes documentaires.{conv_ctx}\n"
        f"Question de l'utilisateur : {question}\n"
        f"Documents disponibles : {source_names}{entities_line}\n\n"
        "RÈGLES DE REFORMULATION (strictes) :\n"
        "1. La question DOIT être auto-suffisante — elle doit contenir toutes les informations nécessaires sans le contexte de conversation.\n"
        "2. Ne générer que des questions pertinentes au domaine documentaire disponible.\n"
        "3. Chaque sous-requête doit être grammaticalement correcte et en français.\n"
        "4. Si la question est complexe, la décomposer en 2-3 aspects distincts. Sinon, générer 1 seule sous-requête.\n"
        "5. Si la question fait référence à quelque chose mentionné dans la conversation précédente, l'intégrer explicitement dans la sous-requête.\n"
        "6. Si un ou plusieurs noms de fichiers sont explicitement mentionnés parmi les documents disponibles, indique-les dans \"targets\". Sinon [].\n"
        "7. En cas de comparaison entre plusieurs documents, conserve-les tous dans \"targets\" et n'en choisis pas un seul arbitrairement.\n"
        "8. Si l'utilisateur mentionne une organisation ou une entité connue (ex. 'chez Dassault', 'pour Thales', 'je travaille à...'), "
        "indique le nom exact de l'entité dans \"target_entity\" (minuscules, correspond exactement à une entité disponible). "
        "Sinon laisse \"target_entity\" vide.\n\n"
        "Réponds UNIQUEMENT en JSON (sans balise markdown) sous la forme :\n"
        '{\n'
        '  "targets": ["<nom_fichier_1>", "<nom_fichier_2>"],\n'
        '  "target_entity": "<nom_entité_ou_vide>",\n'
        '  "reason": "<explication courte>",\n'
        '  "sub_queries": ["<requête_1>", "<requête_2>"],\n'
        '  "confidence": 0.9\n'
        '}'
    )


def _resolve_source_filter(
    target_name: Optional[str],
    sources: list[str],
) -> Optional[str]:
    """Résout un nom de fichier en chemin complet."""
    if not target_name or target_name.lower() == "null":
        return None
    for s in sources:
        if Path(s).name == target_name:
            return s
    return None


def _resolve_source_filters(
    target_names: list[str],
    sources: list[str],
) -> list[str]:
    resolved: list[str] = []
    for target_name in target_names:
        target = _resolve_source_filter(target_name, sources)
        if target and target not in resolved:
            resolved.append(target)
    return resolved


def analyze_and_plan(state: UnifiedRAGState, *, llm_call: Callable, rag_config: RAGConfig) -> dict:
    """Nœud 1 : décompose la question et identifie le filtre source."""
    qid      = state["question_id"]
    log      = list(state.get("decision_log", []))
    question = state["question"]
    sources  = state.get("available_sources", [])
    filter_  = state.get("source_filter")

    # Métadonnées enrichies (entity, validity_date par source)
    sources_meta = state.get("available_sources_meta") or []
    if not sources_meta and sources:
        # Fallback : construire des méta minimales sans entité
        sources_meta = [{"source": s, "entity": "", "validity_date": ""} for s in sources]

    if filter_:
        log.append(log_entry("analyze", f"Filtre manuel → {Path(filter_).name}", {"source": filter_}))

    conv_ctx = ""
    if state.get("conversation_summary"):
        conv_ctx = f"\nContexte de la conversation précédente :\n{state['conversation_summary']}\n"

    prompt = _build_planning_prompt(question, sources_meta, conv_ctx)
    parsed_output: Optional[PlanningOutput] = None

    for attempt in range(2):
        try:
            resp = llm_call(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            raw           = resp.choices[0].message.content or "{}"
            pre_parsed    = parse_json_llm(raw)
            if isinstance(pre_parsed, dict) and "targets" not in pre_parsed and "target" in pre_parsed:
                legacy_target = pre_parsed.get("target")
                pre_parsed["targets"] = (
                    [legacy_target]
                    if isinstance(legacy_target, str) and legacy_target.strip() and legacy_target.lower() != "null"
                    else []
                )
            parsed_output = PlanningOutput.model_validate(pre_parsed)
            break
        except Exception as exc:
            logger.warning("[{}] analyze_and_plan attempt {}: {}", qid, attempt + 1, exc)

    if parsed_output is None:
        log.append(log_entry("analyze", "Fallback : LLM indisponible, utilisation de la question brute"))
        return {
            "sub_queries":    [question],
            "source_filter":  filter_,
            "reasoning":      "fallback: planning LLM unavailable",
            "current_branch": "plan",
            "decision_history": list(state.get("decision_history", [])) + ["plan.analyze"],
            "tree_depth":     state.get("tree_depth", 0) + 1,
            "decision_log":   log,
        }

    resolved_targets = _resolve_source_filters(parsed_output.targets, sources)
    final_filter     = filter_ or (resolved_targets[0] if len(resolved_targets) == 1 else None)
    entity_f         = (parsed_output.target_entity or "").strip().lower() or None

    log.append(log_entry(
        "analyze",
        f"Cibles : {parsed_output.targets or ['aucune']}. Entité : {entity_f or 'aucune'}. Requêtes : {parsed_output.sub_queries}",
        {
            "target":        final_filter,
            "targets":       resolved_targets,
            "entity_filter": entity_f,
            "sub_queries":   parsed_output.sub_queries,
            "reason":        parsed_output.reason,
        },
    ))

    return {
        "sub_queries":          parsed_output.sub_queries,
        "source_filter":        final_filter,
        "entity_filter":        entity_f,
        "target_sources":       resolved_targets,
        "reasoning":            parsed_output.reason,
        "current_branch":       "plan",
        "decision_history":     list(state.get("decision_history", [])) + ["plan.analyze"],
        "tree_depth":           state.get("tree_depth", 0) + 1,
        "decision_log":         log,
    }
