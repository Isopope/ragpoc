"""Nœuds agent_reason, agent_action, consolidate_chunks et routeurs.

Port de rag_pipeline.py:477-851.
"""
from __future__ import annotations

import json
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
                    "query": {"type": "string", "description": "Mots-clés de recherche."}
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
    """Nœud 3 : exécute les tool_calls du modèle (search_documents / get_neighboring_chunk)."""
    qid          = state["question_id"]
    log          = list(state.get("decision_log", []))
    messages     = list(state.get("messages", []))
    all_docs     = list(state.get("all_docs", []))
    seen_keys    = set(state.get("seen_keys", set()))
    seen_queries = list(state.get("seen_queries", []))
    filter_      = state.get("source_filter")

    model_content    = messages[-1]
    fn_calls         = model_content.get("tool_calls") or []
    fn_response_parts: list[dict] = []

    for tc in fn_calls:
        fc_name = tc["function"]["name"]
        fc_args = json.loads(tc["function"]["arguments"])
        result: dict = {}

        # ── search_documents ──────────────────────────────────────────────────
        if fc_name == "search_documents":
            query = fc_args.get("query", "")
            is_dup = any(q.lower().strip() == query.lower().strip() for q, _ in seen_queries)
            seen_queries.append((query, 1.0))

            if is_dup:
                result = {"found": 0, "results": [], "notice": "Requête déjà effectuée, essaie une formulation différente."}
                log.append(log_entry("agent.action", f"Skip query (duplicate): {query[:50]}"))
            else:
                try:
                    merged     = query_tool.execute(query, source_filter=filter_, top_k=rag_config.top_k_retrieve, alpha=rag_config.hybrid_alpha)
                    new_count  = 0
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

                    result = {"found": len(merged), "new_chunks": new_count, "results": chunks_info[:10]}
                    log.append(log_entry(
                        "agent.action",
                        f"Recherche '{query[:50]}' → {len(merged)} hits ({new_count} nouveaux)",
                        {"query": query, "found": len(merged), "new": new_count},
                    ))
                except Exception as exc:
                    result = {"error": f"Recherche échouée: {exc}"}
                    logger.error("[{}] agent_action search '{}': {}", qid, query[:50], exc)
                    log.append(log_entry(
                        "agent.action",
                        f"ERREUR search '{query[:50]}': {exc}",
                        {"query": query, "error": str(exc)},
                    ))

        # ── get_neighboring_chunk ─────────────────────────────────────────────
        elif fc_name == "get_neighboring_chunk":
            src_name = fc_args.get("source_name", "")
            idx      = int(fc_args.get("chunk_index", -1))

            # Garde-fou anti-injection : valide la plage d'index (port de rag_pipeline.py:688)
            _IDX_MIN, _IDX_MAX = 0, 100_000
            if not (_IDX_MIN <= idx <= _IDX_MAX):
                result = {"error": f"Index invalide : {idx} (hors plage [{_IDX_MIN}, {_IDX_MAX}])"}
                fn_response_parts.append({
                    "role": "tool", "tool_call_id": tc["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                })
                continue

            source_full = filter_ or next(
                (d["source"] for d in all_docs if Path(d.get("source", "")).name == src_name),
                None,
            )
            source_full = source_full or next(
                (s for s in state.get("available_sources", []) if Path(s).name == src_name),
                src_name,
            )
            try:
                chunk = query_tool.get_chunk_by_index(source_full, idx)
                if chunk:
                    k = (chunk.get("source", ""), int(chunk.get("chunk_index", -1)))
                    if k not in seen_keys:
                        chunk["_expanded"] = True
                        all_docs.append(chunk)
                        seen_keys.add(k)
                    result = {"found": True, "chunk": {"chunk_index": idx, "content": chunk.get("page_content", "")}}
                    log.append(log_entry("agent.action", f"Voisin {src_name} idx {idx}"))
                else:
                    result = {"found": False}
            except Exception as exc:
                result = {"error": str(exc)}
                logger.error("[{}] get_neighboring_chunk {} idx {}: {}", qid, src_name, idx, exc)
                log.append(log_entry(
                    "agent.action",
                    f"ERREUR get_chunk {src_name} idx {idx}: {exc}",
                    {"error": str(exc)},
                ))
        else:
            result = {"error": "Outil inconnu."}

        fn_response_parts.append({
            "role":         "tool",
            "tool_call_id": tc["id"],
            "content":      json.dumps(result, ensure_ascii=False),
        })

    messages.extend(fn_response_parts)
    return {
        "messages":     messages,
        "decision_log": log,
        "all_docs":     all_docs,
        "seen_keys":    seen_keys,
        "seen_queries": seen_queries,
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
