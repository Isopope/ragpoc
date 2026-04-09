"""Nœuds agent_reason, agent_action, consolidate_chunks et routeurs.

Port de rag_pipeline.py:477-851.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from ..config import RAGConfig
from ..state import UnifiedRAGState, log_entry
from ..tools.query import QueryTool, combine_chunks

# ── Schéma des outils OpenAI (constant) ───────────────────────────────────────

TOOLS_CFG: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Effectue une recherche hybride (sémantique + BM25) dans la base documentaire. "
                "Appeler plusieurs fois avec des formulations variées si les résultats sont insuffisants. "
                "Retourne les extraits (jusqu'à 10) avec un 'chunk_index' et un 'source_name'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Mots-clés de recherche."},
                    "source_name": {
                        "type": "string",
                        "description": (
                            "Nom du fichier cible si tu veux limiter cette recherche à un document précis. "
                            "Laisse vide pour chercher dans toute la base."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_neighboring_chunk",
            "description": (
                "Récupère le contexte exact entourant un chunk s'il semble coupé, "
                "en appelant l'index prev_chunk ou next_chunk (si >= 0)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_name":  {"type": "string",  "description": "Nom du fichier"},
                    "chunk_index":  {"type": "integer", "description": "Index précis du chunk manquant"},
                },
                "required": ["source_name", "chunk_index"],
            },
        },
    },
]


# ── Routeurs ───────────────────────────────────────────────────────────────────

def route_agent(state: UnifiedRAGState) -> str:
    """Routeur après agent_reason.

    Retourne :
      "agent_action"  → le modèle a émis des tool_calls
      "rerank_prep"   → l'agent a terminé ou max_iter atteint
    """
    messages   = state.get("messages", [])
    iterations = state.get("agent_iterations", 0)
    config_max = state.get("_max_agent_iter", 60)  # injecté par le graphe

    if not messages:
        return "agent_action"

    if iterations >= config_max:
        logger.warning("[{}] route_agent — max iterations ({})", state["question_id"], iterations)
        return "rerank_prep"

    last_msg = messages[-1]
    if isinstance(last_msg, dict) and last_msg.get("role") == "assistant":
        if last_msg.get("tool_calls"):
            return "agent_action"
        return "rerank_prep"

    return "agent_action"


def route_after_action(state: UnifiedRAGState, *, rag_config: RAGConfig) -> str:
    """Routeur après agent_action.

    Retourne "compress_context" si le budget token est dépassé, sinon "agent_reason".
    """
    messages  = state.get("messages", [])
    msg_chars = sum(
        len(str(m.get("content") or "")) + len(str(m.get("tool_calls") or ""))
        for m in messages
    )
    doc_chars = sum(len(doc.get("page_content", "")) for doc in state.get("all_docs", []))
    estimated = (msg_chars + doc_chars) // 4

    if not rag_config.enable_compression:
        if estimated > rag_config.token_threshold:
            logger.info(
                "[{}] Compression désactivée — seuil dépassé ({} tokens estimés)",
                state["question_id"], estimated,
            )
        return "agent_reason"

    if estimated > rag_config.token_threshold:
        logger.info(
            "[{}] Seuil de compression atteint ({} tokens estimés)",
            state["question_id"], estimated,
        )
        return "compress_context"
    return "agent_reason"


# ── Nœud 2 : agent_reason ─────────────────────────────────────────────────────

def _build_initial_prompt(state: UnifiedRAGState) -> str:
    """Construit le prompt inicial de la boucle ReAct."""
    sources_info = ""
    if state.get("available_sources"):
        names        = [Path(s).name for s in state["available_sources"]]
        sources_info = f"Documents indexés: {', '.join(names)}."
        if state.get("source_filter"):
            sources_info += f" (Filtré sur {Path(state['source_filter']).name})"
        elif state.get("target_sources"):
            target_names = [Path(s).name for s in state.get("target_sources", [])]
            if target_names:
                sources_info += f" Documents explicitement ciblés: {', '.join(target_names)}."

    plans = " - " + "\n - ".join(state.get("sub_queries", [state["question"]]))

    first_rule = (
        "1. Comble les lacunes du contexte compressé ci-dessous avec de nouvelles recherches ciblées.\n"
        if state.get("context_summary")
        else "1. Tu DOIS utiliser 'search_documents' lors de ta PREMIÈRE action — sans exception.\n"
    )

    context_injection = ""
    if state.get("context_summary"):
        context_injection = (
            "\n\n[CONTEXTE DE RECHERCHE COMPRESSÉ DEPUIS LES RECHERCHES PRÉCÉDENTES]\n"
            f"{state['context_summary']}\n"
            "[FIN DU CONTEXTE COMPRESSÉ]\n\n"
            "Poursuis la recherche en comblant les lacunes identifiées ci-dessus."
        )

    return (
        f"Tu es un agent de recherche documentaire expert.\n{sources_info}\n\n"
        f"Question de l'utilisateur : {state['question']}\n\n"
        "Le système d'analyse préconise d'essayer ces angles de recherche :\n"
        f"{plans}\n\n"
        "RÈGLES STRICTES :\n"
        f"{first_rule}"
        "2. Si un extrait semble coupé ou incomplet, utilise 'get_neighboring_chunk'.\n"
        "   Si la question compare plusieurs documents, appelle 'search_documents' plusieurs fois avec 'source_name' pour chaque document.\n"
        "3. Écris ton raisonnement AVANT chaque appel d'outil ou décision finale.\n"
        "4. Varie les formulations de recherche pour couvrir tous les aspects de la question.\n"
        "5. Quand les extraits récupérés suffisent à répondre, dis 'RECHERCHE_TERMINEE'.\n"
        "   IMPORTANT : Ne dis JAMAIS 'RECHERCHE_TERMINEE' si tu reçois des erreurs de la base documentaire.\n"
        "   En cas d'erreur d'un outil, essaie une autre formulation ou un autre angle — ne capitule pas.\n"
        f"6. Ne jamais inventer d'informations non présentes dans les extraits.{context_injection}"
    )


def agent_reason(state: UnifiedRAGState, *, llm_call: Callable, rag_config: RAGConfig) -> dict:
    """Nœud 2 : raisonnement ReAct, produit des tool_calls ou termine la boucle."""
    qid        = state["question_id"]
    log        = list(state.get("decision_log", []))
    messages   = list(state.get("messages", []))
    iterations = state.get("agent_iterations", 0)

    # Entrée dans la branche react (première itération de la boucle)
    history = list(state.get("decision_history", []))
    if not messages and "react.search" not in history:
        history = history + ["react.search"]

    if not messages:
        messages.append({"role": "user", "content": _build_initial_prompt(state)})

    try:
        resp = llm_call(
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
            tools=TOOLS_CFG,
            tool_choice="auto",
        )
    except Exception as exc:
        logger.error("[{}] agent_reason — erreur : {}", qid, exc)
        return {"error": str(exc)}

    choice  = resp.choices[0]
    message = choice.message

    if not message.content and not message.tool_calls:
        logger.warning("[{}] agent_reason content vide ({})", qid, choice.finish_reason or "UNKNOWN")
        messages.append({"role": "assistant", "content": "RECHERCHE_TERMINEE"})
        return {"messages": messages, "current_branch": "react", "decision_history": history}

    model_msg: dict[str, Any] = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        model_msg["tool_calls"] = [
            {
                "id":       tc.id,
                "type":     tc.type,
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ]
    messages.append(model_msg)

    if message.content:
        thought = message.content.strip()
        if thought:
            log.append(log_entry("agent.think", thought[:500], {"iteration": iterations}))

    return {
        "messages":         messages,
        "decision_log":     log,
        "agent_iterations": iterations + 1,
        "current_branch":   "react",
        "decision_history": history,
    }


# ── Nœud 3 : agent_action ─────────────────────────────────────────────────────

def agent_action(
    state: UnifiedRAGState,
    *,
    query_tool: QueryTool,
    rag_config: RAGConfig,
    weaviate_store: Any = None,
) -> dict:
    """Nœud 3 : exécute les tool_calls en parallèle (search_documents / get_neighboring_chunk).

    Architecture en 3 phases :
      1. Pré-traitement séquentiel  — déduplication, résolution des sources, validation des index
      2. Exécution parallèle         — ThreadPoolExecutor sur les appels Weaviate indépendants
      3. Fusion séquentielle          — déduplication seen_keys, construction fn_response_parts
    """
    qid            = state["question_id"]
    log            = list(state.get("decision_log", []))
    messages       = list(state.get("messages", []))
    all_docs       = list(state.get("all_docs", []))
    seen_keys      = set(state.get("seen_keys", set()))
    seen_queries   = list(state.get("seen_queries", []))
    filter_        = state.get("source_filter")
    target_sources = state.get("target_sources", [])

    model_content = messages[-1]
    fn_calls      = model_content.get("tool_calls") or []

    _IDX_MIN, _IDX_MAX = 0, 100_000

    # ── Phase 1 : pré-traitement séquentiel ───────────────────────────────────
    # Chaque élément de `prepared` décrit un tool_call et ses paramètres résolus.
    prepared: list[dict] = []

    for tc in fn_calls:
        fc_name = tc["function"]["name"]
        try:
            fc_args = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError:
            fc_args = {}

        item: dict = {"tc": tc, "fc_name": fc_name, "skip": False, "skip_result": None}

        if fc_name == "search_documents":
            query                 = fc_args.get("query", "")
            requested_source_name = str(fc_args.get("source_name") or "").strip()
            resolved_source       = (
                filter_
                or next((s for s in target_sources if Path(s).name.lower() == requested_source_name.lower()), None)
                or next((s for s in state.get("available_sources", []) if Path(s).name.lower() == requested_source_name.lower()), None)
            )
            query_signature = f"{requested_source_name.lower()}::{query.lower().strip()}"
            is_dup          = any(q.lower().strip() == query_signature for q, _ in seen_queries)
            seen_queries.append((query_signature, 1.0))

            item.update({
                "query":                  query,
                "requested_source_name":  requested_source_name,
                "resolved_source":        resolved_source,
                "query_signature":        query_signature,
            })

            if is_dup:
                item["skip"]        = True
                item["skip_result"] = {"found": 0, "results": [], "notice": "Requête déjà effectuée, essaie une formulation différente."}
                log.append(log_entry("agent.action", f"Skip query (duplicate): {query[:50]} / {requested_source_name or 'all'}"))

        elif fc_name == "get_neighboring_chunk":
            src_name = fc_args.get("source_name", "")
            idx      = int(fc_args.get("chunk_index", -1))

            if not (_IDX_MIN <= idx <= _IDX_MAX):
                item["skip"]        = True
                item["skip_result"] = {"error": f"Index invalide : {idx} (hors plage [{_IDX_MIN}, {_IDX_MAX}])"}
            else:
                source_full = filter_ or next(
                    (d["source"] for d in all_docs if Path(d.get("source", "")).name == src_name),
                    None,
                )
                source_full = source_full or next(
                    (s for s in state.get("available_sources", []) if Path(s).name == src_name),
                    src_name,
                )
                item.update({"src_name": src_name, "idx": idx, "source_full": source_full})

        else:
            item["skip"]        = True
            item["skip_result"] = {"error": "Outil inconnu."}

        prepared.append(item)

    # ── Phase 2 : exécution parallèle des appels Weaviate ─────────────────────

    def _run_search(it: dict) -> dict:
        try:
            chunks = query_tool.execute(
                it["query"],
                source_filter=it["resolved_source"],
                top_k=rag_config.top_k_retrieve,
                alpha=rag_config.hybrid_alpha,
            )
            return {"ok": True, "chunks": chunks}
        except Exception as exc:
            return {"ok": False, "error": exc}

    def _run_get_chunk(it: dict) -> dict:
        try:
            chunk = query_tool.get_chunk_by_index(it["source_full"], it["idx"])
            return {"ok": True, "chunk": chunk}
        except Exception as exc:
            return {"ok": False, "error": exc}

    active   = [p for p in prepared if not p["skip"]]
    max_w    = max(len(active), 1)
    futures  = {}

    with ThreadPoolExecutor(max_workers=min(max_w, 8)) as executor:
        for it in active:
            if it["fc_name"] == "search_documents":
                futures[id(it)] = executor.submit(_run_search, it)
            elif it["fc_name"] == "get_neighboring_chunk":
                futures[id(it)] = executor.submit(_run_get_chunk, it)

    # ── Phase 3 : fusion séquentielle dans l'ordre original ───────────────────
    fn_response_parts: list[dict] = []
    new_docs_total = 0

    for item in prepared:
        tc    = item["tc"]
        tc_id = tc["id"]

        if item["skip"]:
            fn_response_parts.append({
                "role": "tool", "tool_call_id": tc_id,
                "content": json.dumps(item["skip_result"], ensure_ascii=False),
            })
            continue

        fut = futures.get(id(item))
        out = fut.result() if fut else {"ok": False, "error": RuntimeError("future manquant")}

        if item["fc_name"] == "search_documents":
            if out["ok"]:
                merged      = out["chunks"]
                new_count   = 0
                chunks_info: list[dict] = []
                for doc in merged:
                    k = (doc.get("source", ""), int(doc.get("chunk_index", -1)))
                    if k not in seen_keys:
                        all_docs.append(doc)
                        seen_keys.add(k)
                        new_count += 1
                    chunks_info.append({
                        "chunk_index": doc.get("chunk_index"),
                        "source_name": Path(doc.get("source", "")).name,
                        "kind":        doc.get("kind", "text"),
                        "title_path":  doc.get("title_path", ""),
                        "content":     doc.get("page_content", ""),
                        "prev_chunk":  doc.get("prev_chunk", -1),
                        "next_chunk":  doc.get("next_chunk", -1),
                    })
                new_docs_total += new_count
                result = {"found": len(merged), "new_chunks": new_count, "results": chunks_info[:10]}
                log.append(log_entry(
                    "agent.action",
                    f"Recherche '{item['query'][:50]}' sur {item['requested_source_name'] or 'toute la base'} → {len(merged)} hits ({new_count} nouveaux)",
                    {
                        "query":            item["query"],
                        "source_name":      item["requested_source_name"] or None,
                        "resolved_source":  item["resolved_source"],
                        "found":            len(merged),
                        "new":              new_count,
                    },
                ))
            else:
                exc    = out["error"]
                result = {"error": f"Recherche échouée: {exc}"}
                logger.error("[{}] agent_action search '{}': {}", qid, item["query"][:50], exc)
                log.append(log_entry(
                    "agent.action",
                    f"ERREUR search '{item['query'][:50]}' sur {item['requested_source_name'] or 'toute la base'}: {exc}",
                    {"query": item["query"], "source_name": item["requested_source_name"] or None, "error": str(exc)},
                ))

        elif item["fc_name"] == "get_neighboring_chunk":
            if out["ok"]:
                chunk = out["chunk"]
                if chunk:
                    k = (chunk.get("source", ""), int(chunk.get("chunk_index", -1)))
                    if k not in seen_keys:
                        chunk["_expanded"] = True
                        all_docs.append(chunk)
                        seen_keys.add(k)
                        new_docs_total += 1
                    result = {"found": True, "chunk": {"chunk_index": item["idx"], "content": chunk.get("page_content", "")}}
                    log.append(log_entry("agent.action", f"Voisin {item['src_name']} idx {item['idx']}"))
                else:
                    result = {"found": False}
            else:
                exc    = out["error"]
                result = {"error": str(exc)}
                logger.error("[{}] get_neighboring_chunk {} idx {}: {}", qid, item["src_name"], item["idx"], exc)
                log.append(log_entry(
                    "agent.action",
                    f"ERREUR get_chunk {item['src_name']} idx {item['idx']}: {exc}",
                    {"error": str(exc)},
                ))
        else:
            result = {"error": "Outil inconnu."}

        fn_response_parts.append({
            "role":         "tool",
            "tool_call_id": tc_id,
            "content":      json.dumps(result, ensure_ascii=False),
        })

    messages.extend(fn_response_parts)
    return {
        "messages":             messages,
        "decision_log":         log,
        "all_docs":             all_docs,
        "seen_keys":            seen_keys,
        "seen_queries":         seen_queries,
        "last_action_new_docs": new_docs_total,
    }


# ── Nœud C : consolidate_chunks ───────────────────────────────────────────────

def consolidate_chunks(
    state: UnifiedRAGState,
    *,
    query_tool: QueryTool,
    rag_config: RAGConfig,
) -> dict:
    """Consolide et déduplique tous les chunks à la fin de la boucle ReAct."""
    docs = state.get("all_docs", [])
    log  = list(state.get("decision_log", []))

    if not docs:
        log.append(log_entry("agent.fallback", "Aucun doc trouvé. Fallback direct."))
        try:
            docs = query_tool.execute(
                state["question"],
                source_filter=state.get("source_filter"),
                top_k=rag_config.top_k_retrieve,
                alpha=0.5,
            )
        except Exception:
            pass

    retrieved_docs = combine_chunks([docs])
    log.append(log_entry(
        "agent.loop_end",
        f"{len(retrieved_docs)} chunks prêts pour rerank.",
        {"current_branch": "synthesize"},
    ))
    return {
        "retrieved_docs":   retrieved_docs,
        "current_branch":   "synthesize",
        "decision_history": list(state.get("decision_history", [])) + ["synthesize.rerank"],
        "tree_depth":       state.get("tree_depth", 0) + 1,
        "decision_log":     log,
    }
