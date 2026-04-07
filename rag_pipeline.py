"""Agent RAG basé sur LangGraph — pipeline décisionnel avec reranking Cohere et expansion.

Pipeline :
  1. analyze_question  — identifie le(s) document(s) ciblé(s) par la question
  2. agent_loop        — ReAct avec function calling natif OpenAI (search + get_chunk)
  3. rerank            — Cohere Rerank (fallback LLM si Cohere indisponible)
  4. generate          — OpenAI LLM génère la réponse finale

Améliorations v2 :
  - Cohere Rerank remplace le LLM dans le nœud rerank
  - deduplicate_queries appelée avant chaque tool call dans agent_loop
  - Timeout threading sur tous les appels LLM (avec fallback gracieux)
  - Retry + backoff exponentiel sur les appels Weaviate
  - question_id UUID propagé dans tous les logs
  - decision_log structuré : list[dict] avec step / ts / metadata
  - _SYSTEM_PROMPT et _build_context_entry internalisés (plus d'import rag_chain)
  - Validation du range de chunk_index côté agent
  - MAX_AGENT_ITER exposé comme paramètre du graphe
  - Fallback agent_loop respecte source_filter
  - Suppression du doublon all_docs / retrieved_docs dans RAGState
  - Exceptions generate propagées de façon structurée
  - Migration vers OpenAI (gpt-4.1, text-embedding-3-small) — v3
"""
from __future__ import annotations

import ast
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

from loguru import logger


# ── Constantes internes ───────────────────────────────────────────────────────

_SYSTEM_PROMPT = """Tu es un assistant expert, précis et bienveillant.

Ta tâche est de générer une réponse complète et structurée basée UNIQUEMENT sur les extraits fournis.

Règles strictes :
1. Utilise UNIQUEMENT les informations présentes dans les extraits fournis.
2. Si l'information demandée n'est pas dans les extraits, dis-le explicitement.
3. Préserve les chiffres, versions, termes techniques et détails exacts.
4. Rédige en français, dans un style clair et professionnel.
5. Ne conclus pas avec des remarques finales, notes, avis ou répétitions après la section Sources.
   La section Sources est toujours le dernier élément de ta réponse.

Mise en forme :
- Utilise le Markdown (titres, gras, listes) pour la lisibilité.
- Rédige en paragraphes fluides quand c'est possible.
- Conclus par une section Sources comme décrit ci-dessous.

Règles pour la section Sources :
- Inclure "---\\n**Sources :**\\n" à la fin, suivi d'une liste à puces des noms de fichiers.
- Lister UNIQUEMENT les entrées ayant une vraie extension de fichier (.pdf, .docx, .txt…).
- Dédupliquer : si le même fichier apparaît plusieurs fois, le lister une seule fois.
- Si aucun nom de fichier valide n'est présent, omettre la section Sources.
- LA SECTION SOURCES EST LA DERNIÈRE CHOSE QUE TU ÉCRIS."""

_COMPRESSION_PROMPT = """Tu es un expert en compression de contexte de recherche.

Ta tâche est de condenser le contenu récupéré en un résumé concis, axé sur la question, directement utilisable par un agent RAG pour continuer ou finaliser sa réponse.

Règles :
1. Conserve UNIQUEMENT les informations pertinentes pour répondre à la question de l'utilisateur.
2. Préserve les chiffres, noms, versions, termes techniques et configurations exacts.
3. Supprime les doublons, détails non pertinents ou administratifs.
4. N'inclus pas les requêtes de recherche, IDs de chunks ni identifiants internes.
5. Organise les résultats par fichier source. Chaque section DOIT commencer par : ### nom_fichier.pdf
6. Signale les informations manquantes dans une section « Lacunes ».
7. Limite à environ 400-600 mots. Priorité aux faits critiques et données structurées.
8. Produis uniquement du contenu Markdown structuré, sans explications.

Structure requise :
# Résumé du Contexte de Recherche

## Focalisation
[Reformulation technique brève de la question]

## Résultats Structurés

### nom_fichier.pdf
- Faits directement pertinents
- Contexte de soutien (si nécessaire)

## Lacunes
- Aspects manquants ou incomplets"""

_CHUNK_INDEX_MIN = 0
_CHUNK_INDEX_MAX = 100_000   # garde-fou anti-injection
BASE_TOKEN_THRESHOLD = 12_000  # ~48k chars → déclenche la compression du contexte
MAX_NO_PROGRESS_STEPS = 2


# ── Helpers : parsing & fusion ────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    """Retire les balises markdown ``` éventuellement ajoutées par le LLM."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_json_llm(text: str) -> object:
    """Parse du JSON potentiellement malformé produit par un LLM.

    Stratégies tentées dans l'ordre :
    1. json.loads standard (après nettoyage des fences)
    2. ast.literal_eval  (accepte guillemets simples, tuples Python, etc.)
    3. Extraction du premier bloc JSON par regex (objet {} ou tableau [])
    4. Nettoyage des virgules trailing avant de retenter json.loads
    """
    if not text:
        raise ValueError("Réponse LLM vide")

    cleaned = _strip_fences(text)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(cleaned)
    except (ValueError, SyntaxError):
        pass

    for pattern in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pattern, cleaned, re.DOTALL)
        if m:
            candidate = m.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
            try:
                return ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                pass

    no_trailing = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        return json.loads(no_trailing)
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Impossible de parser la réponse LLM : {cleaned[:200]!r}")


def _combine_chunks(chunk_sets: list[list[dict]]) -> list[dict]:
    """Fusionne plusieurs listes de chunks en dédupliquant par (source, chunk_index).

    En cas de doublon, conserve l'entrée avec le score hybride le plus élevé.
    Retourne les chunks triés par score décroissant (ordre stable).
    """
    unique: dict[tuple[str, int], dict] = {}
    for chunk in (c for cs in chunk_sets for c in cs):
        key = (chunk.get("source", ""), int(chunk.get("chunk_index", -1)))
        if key not in unique:
            unique[key] = chunk
        else:
            if (chunk.get("_score") or 0) > (unique[key].get("_score") or 0):
                merged = {**chunk}
                for k, v in unique[key].items():
                    if k.startswith("_") and k not in merged:
                        merged[k] = v
                unique[key] = merged

    return sorted(unique.values(), key=lambda d: d.get("_score") or 0, reverse=True)


def deduplicate_queries(
    queries_with_weights: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Déduplique les requêtes par comparaison insensible à la casse, somme les poids."""
    query_map: dict[str, tuple[str, float]] = {}
    for query, weight in queries_with_weights:
        key = query.lower().strip()
        if key in query_map:
            orig, w = query_map[key]
            query_map[key] = (orig, w + weight)
        else:
            query_map[key] = (query, weight)
    return list(query_map.values())


def _weighted_rrf(
    ranked_results: list[list[dict]],
    weights: list[float],
    k: int = 60,
) -> list[dict]:
    """Weighted Reciprocal Rank Fusion sur plusieurs listes ordonnées.

    Formule : score(doc) = Σ weight_i / (k + rank_i)
    """
    rrf_scores: dict[tuple[str, int], float] = {}
    best_doc:   dict[tuple[str, int], dict]  = {}

    for result_list, weight in zip(ranked_results, weights):
        for rank, doc in enumerate(result_list, start=1):
            key = (doc.get("source", ""), int(doc.get("chunk_index", -1)))
            rrf_scores[key] = rrf_scores.get(key, 0.0) + weight / (k + rank)
            if key not in best_doc:
                best_doc[key] = doc
            elif (doc.get("_score") or 0.0) > (best_doc[key].get("_score") or 0.0):
                best_doc[key] = doc

    return [
        {**best_doc[key], "_score": rrf_scores[key]}
        for key in sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    ]


# ── Helpers : logging structuré ───────────────────────────────────────────────

def _log_entry(
    step: str,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> dict:
    """Crée une entrée de log structurée."""
    return {
        "step":     step,
        "ts":       datetime.now(timezone.utc).isoformat(),
        "message":  message,
        "metadata": metadata or {},
    }


# ── Helpers : retry Weaviate ──────────────────────────────────────────────────

def _weaviate_with_retry(fn, *args, max_retries: int = 3, base_delay: float = 0.5, **kwargs):
    """Appelle fn(*args, **kwargs) avec retry exponentiel sur exception."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            delay = base_delay * (2 ** attempt)
            logger.warning("Weaviate retry {}/{} dans {:.1f}s : {}", attempt + 1, max_retries, delay, exc)
            time.sleep(delay)
    raise RuntimeError(f"Weaviate indisponible après {max_retries} tentatives") from last_exc


# ── Helpers : contexte de génération ─────────────────────────────────────────

def _build_context_entry(index: int, doc: dict) -> str:
    """Formate un chunk pour le prompt de génération."""
    source_name = Path(doc.get("source", "inconnu")).name
    title_path  = (doc.get("title_path") or "").strip()
    content     = (doc.get("page_content") or "").strip()
    kind        = doc.get("kind", "text")
    expanded    = " (contexte étendu)" if doc.get("_expanded") else ""

    header = f"[Source {index}] {source_name}"
    if title_path:
        header += f" — {title_path}"
    header += f" [{kind}{expanded}]"

    return f"{header}\n{content}"


# ── État partagé du graphe ────────────────────────────────────────────────────

class RAGState(TypedDict):
    """État mutable propagé entre les nœuds LangGraph."""
    question_id:        str             # UUID de corrélation pour les logs
    question:           str
    available_sources:  list[str]
    source_filter:      str | None
    target_sources:     list[str]
    sub_queries:        list[str]       # Sous-requêtes planifiées issues de l'analyse
    messages:           list[Any]       # Historique LLM (OpenAI) pour la boucle
    all_docs:           list[dict]      # Chunks bruts collectés
    seen_keys:          set[tuple[str, int]] # Déduplication
    seen_queries:       list[tuple[str, float]] # Déduplication des recherches
    agent_iterations:   int             # Compteur d'itérations
    consecutive_no_progress: int       # Nombre d'actions successives sans nouveau contexte utile
    last_action_new_docs: int          # Nombre de nouveaux chunks ajoutés à la dernière action
    conversation_summary: str          # Résumé des échanges précédents (passé par l'UI)
    context_summary:    str            # Résumé compressé du contexte de recherche interne

    retrieved_docs:     list[dict]      # Chunks finals unifiés après ReAct
    reranked_docs:      list[dict]
    answer:             str
    decision_log:       list[dict]      # entrées structurées {step, ts, message, metadata}
    error:              str | None      # erreur fatale propagée (None si succès)


# ── Construction du graphe ────────────────────────────────────────────────────

def build_rag_graph(
    weaviate_store,
    openai_key: str,
    cohere_key: str | None = None,
    embedding_model: str = "text-embedding-3-small",
    llm_model: str = "gpt-4.1",
    top_k_retrieve: int = 20,
    top_k_final: int = 5,
    hybrid_alpha: float = 0.5,
    max_tokens: int = 4000,
    max_agent_iter: int = 60,
    llm_timeout: float = 30.0,
    enable_compression: bool = True,
):
    """Compile et retourne le graphe LangGraph RAG.

    Args:
        weaviate_store:   Instance du store Weaviate (hybride BM25 + HNSW).
        openai_key:       Clé API OpenAI.
        cohere_key:       Clé API Cohere pour le reranking (facultatif, fallback LLM si None).
        embedding_model:  Modèle d'embedding OpenAI.
        llm_model:        Modèle LLM OpenAI pour analyze + agent + génération.
        top_k_retrieve:   Nombre de chunks récupérés par requête Weaviate.
        top_k_final:      Non utilisé directement (conservé pour compatibilité API).
        hybrid_alpha:     Balance sémantique/BM25 (0 = full BM25, 1 = full sémantique).
        max_tokens:       Budget de tokens pour la réponse finale.
        max_agent_iter:   Nombre maximal d'itérations de la boucle ReAct.
        llm_timeout:      Timeout en secondes sur chaque appel LLM OpenAI.
        enable_compression:
                         Si False, le nœud compress_context est conservé dans le graphe
                         mais n'est jamais emprunté.
    """
    import threading

    from openai import OpenAI
    from langgraph.graph import END, START, StateGraph

    client = OpenAI(api_key=openai_key)

    # Cohere client (optionnel)
    cohere_client = None
    if cohere_key:
        try:
            import cohere
            cohere_client = cohere.Client(api_key=cohere_key)
            logger.info("Cohere Rerank activé")
        except ImportError:
            logger.warning("Package 'cohere' non installé — fallback reranking LLM")

    # ── Helper : appel LLM avec timeout ──────────────────────────────────────
    def _llm_call_with_timeout(
        messages: list,
        timeout: float = llm_timeout,
        **kwargs,
    ):
        """Appelle client.chat.completions.create dans un thread avec timeout.

        Lève TimeoutError si le LLM ne répond pas dans `timeout` secondes.
        """
        result: dict = {"response": None, "error": None}

        def _run():
            try:
                result["response"] = client.chat.completions.create(
                    model=llm_model, messages=messages, **kwargs
                )
            except Exception as exc:
                result["error"] = exc

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            raise TimeoutError(f"LLM timeout après {timeout}s")
        if result["error"]:
            raise result["error"]
        return result["response"]

    # ── Helper : embedding avec timeout ──────────────────────────────────────
    def _embed_with_timeout(
        text: str,
        timeout: float = llm_timeout,
    ) -> list[float]:
        """Appelle client.embeddings.create dans un thread avec timeout.

        Retourne directement le vecteur (list[float]).
        """
        result: dict = {"vector": None, "error": None}

        def _run():
            try:
                resp = client.embeddings.create(model=embedding_model, input=text or " ")
                result["vector"] = resp.data[0].embedding
            except Exception as exc:
                result["error"] = exc

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            raise TimeoutError(f"Embedding timeout après {timeout}s")
        if result["error"]:
            raise result["error"]
        return result["vector"]

    def _resolve_sources_by_names(target_names: list[str], sources: list[str]) -> list[str]:
        """Résout une liste de noms de fichiers vers leurs chemins complets."""
        resolved: list[str] = []
        normalized = {Path(s).name.lower(): s for s in sources}
        for name in target_names:
            key = (name or "").strip().lower()
            if key and key in normalized and normalized[key] not in resolved:
                resolved.append(normalized[key])
        return resolved

    def _resolve_source_by_name(source_name: str | None, sources: list[str]) -> str | None:
        """Résout un nom de fichier vers son chemin complet."""
        if not source_name:
            return None
        target = source_name.strip().lower()
        for src in sources:
            if Path(src).name.lower() == target:
                return src
        return None

        # ── Nœud 1 : Planification & Reformulation ───────────────────────────────
    def analyze_and_plan(state: RAGState) -> dict:
        """S'inspire de 'rewrite_query' : décompose la question en sous-requêtes 
        optimisées et identifie d'éventuels filtres."""
        qid  = state["question_id"]
        log  = list(state.get("decision_log", []))
        question = state["question"]
        sources  = state.get("available_sources", [])
        filter_ = state.get("source_filter")

        if filter_ is not None:
            name = Path(filter_).name
            log.append(_log_entry("analyze", f"Filtre manuel → {name}", {"source": filter_}))
            # Même avec un filtre manuel, on peut toujours décomposer la requête.

        conv_ctx = ""
        if state.get("conversation_summary"):
            conv_ctx = f"\nContexte de la conversation précédente :\n{state['conversation_summary']}\n"

        prompt = f"""Tu es un expert en analyse de requêtes documentaires.{conv_ctx}
Question de l'utilisateur : {question}
Documents disponibles : {', '.join([Path(s).name for s in sources]) if sources else 'Aucun'}

RÈGLES DE REFORMULATION (strictes) :
1. La question DOIT être auto-suffisante — elle doit contenir toutes les informations nécessaires sans le contexte de conversation.
2. Ne générer que des questions pertinentes au domaine documentaire disponible.
3. Chaque sous-requête doit être grammaticalement correcte et en français.
4. Si la question est complexe, la décomposer en 2-3 aspects distincts. Sinon, générer 1 seule sous-requête.
5. Si la question fait référence à quelque chose mentionné dans la conversation précédente, l'intégrer explicitement dans la sous-requête.
6. Si un ou plusieurs noms de fichiers sont explicitement mentionnés parmi les documents disponibles, indique-les dans "targets". Sinon [].
7. En cas de comparaison entre plusieurs documents, conserve-les tous dans "targets" et n'en choisis pas un seul arbitrairement.

Réponds UNIQUEMENT en JSON (sans balise markdown) sous la forme :
{{
  "targets": ["<nom_fichier_1>", "<nom_fichier_2>"],
  "reason": "<explication courte>",
  "sub_queries": ["<requête_1>", "<requête_2>"]
}}"""

        target_names: list[str] = []
        sub_queries = []
        reason = "analyse standard"

        try:
            resp = _llm_call_with_timeout(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            parsed = _parse_json_llm(resp.choices[0].message.content or "{}")
            if isinstance(parsed, dict):
                raw_targets = parsed.get("targets")
                if isinstance(raw_targets, str):
                    target_names = [raw_targets]
                elif isinstance(raw_targets, list):
                    target_names = [str(name) for name in raw_targets if str(name).strip()]
                else:
                    legacy_target = parsed.get("target")
                    if isinstance(legacy_target, str) and legacy_target.lower() != "null":
                        target_names = [legacy_target]
                sub_queries = parsed.get("sub_queries", [])
                reason      = parsed.get("reason", reason)
                if isinstance(sub_queries, str):
                    sub_queries = [sub_queries]
        except Exception as exc:
            logger.warning("[{}] analyze_and_plan — erreur LLM : {}", qid, exc)

        resolved_targets = _resolve_sources_by_names(target_names, sources)
        target_filter = filter_
        if not filter_ and len(resolved_targets) == 1:
            target_filter = resolved_targets[0]

        if not sub_queries:
            sub_queries = [question]

        log.append(_log_entry(
            "analyze",
            f"Cibles : {target_names or ['aucune']}. Requêtes : {sub_queries}",
            {
                "target": target_filter,
                "targets": resolved_targets,
                "sub_queries": sub_queries,
                "reason": reason,
            },
        ))
        
        return {
            "source_filter": target_filter,
            "target_sources": resolved_targets,
            "sub_queries": sub_queries[:3],
            "decision_log": log,
        }

    # ── Initialisation des Outils  ────────────────────────────────────────────
    tools_cfg = [
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
                        "source_name": {"type": "string", "description": "Nom du fichier"},
                        "chunk_index": {"type": "integer", "description": "Index précis du chunk manquant"},
                    },
                    "required": ["source_name", "chunk_index"],
                },
            },
        },
    ]

    # ── Nœud 2 : Raisonnement de l'agent (Orchestrator) ──────────────────────
    def agent_reason(state: RAGState) -> dict:
        qid = state["question_id"]
        log = list(state.get("decision_log", []))
        messages = list(state.get("messages", []))
        iterations = state.get("agent_iterations", 0)
        
        # S'inspirer du "get_orchestrator_prompt" : Consignes strictes + suggestions de multi-queries
        if not messages:
            sources_info = ""
            if state.get("available_sources"):
                names = [Path(s).name for s in state["available_sources"]]
                sources_info = f"Documents indexés: {', '.join(names)}."
                if state.get("source_filter"):
                    sources_info += f" (Filtré sur {Path(state['source_filter']).name})"
                elif state.get("target_sources"):
                    target_names = [Path(s).name for s in state.get("target_sources", [])]
                    if target_names:
                        sources_info += f" Documents explicitement ciblés: {', '.join(target_names)}."

            current_docs = len(state.get("all_docs", []))
            no_progress = state.get("consecutive_no_progress", 0)
            progress_hint = (
                f"\nContexte collecté jusque-là: {current_docs} extrait(s) uniques. "
                f"Actions sans progrès consécutives: {no_progress}."
            )
            
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

            initial_prompt = (
                f"Tu es un agent de recherche documentaire expert.\n{sources_info}{progress_hint}\n\n"
                f"Question de l'utilisateur : {state['question']}\n\n"
                "Le système d'analyse préconise d'essayer ces angles de recherche :\n"
                f"{plans}\n\n"
                "RÈGLES STRICTES :\n"
                f"{first_rule}"
                "2. Si un extrait semble coupé ou incomplet, utilise 'get_neighboring_chunk'.\n"
                "   Si la question compare plusieurs documents, appelle 'search_documents' plusieurs fois avec 'source_name' pour chaque document.\n"
                "3. Avant un appel d'outil, écris AU PLUS une phrase courte de raisonnement.\n"
                "4. N'appelle PAS un outil si tu as déjà assez d'éléments pour répondre.\n"
                "5. Quand les extraits récupérés suffisent à répondre, dis 'RECHERCHE_TERMINEE'.\n"
                "   Si la dernière action n'a apporté aucun nouveau chunk utile, évite de relancer une recherche proche.\n"
                f"   Si tu disposes déjà d'environ {top_k_final} extraits utiles, termine la recherche.\n"
                "   IMPORTANT : Ne dis JAMAIS 'RECHERCHE_TERMINEE' si tu reçois des erreurs de la base documentaire.\n"
                "   En cas d'erreur d'un outil, essaie une autre formulation ou un autre angle — sans boucler inutilement.\n"
                f"6. Ne jamais inventer d'informations non présentes dans les extraits.{context_injection}"
            )
            messages.append({"role": "user", "content": initial_prompt})

        try:
            resp = _llm_call_with_timeout(
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
                tools=tools_cfg,
                tool_choice="auto",
            )
        except Exception as exc:
            logger.error("[{}] agent_reason — erreur : {}", qid, exc)
            return {"error": str(exc)}

        choice = resp.choices[0]
        message = choice.message

        if not message.content and not message.tool_calls:
            finish_reason = choice.finish_reason or "UNKNOWN"
            logger.warning("[{}] agent_reason content vide ({})", qid, finish_reason)
            messages.append({"role": "assistant", "content": "RECHERCHE_TERMINEE"})
            return {"messages": messages}

        model_msg: dict = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            model_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]
        messages.append(model_msg)

        if message.content:
            thought = message.content.strip()
            if thought:
                log.append(_log_entry("agent.think", thought[:500], {"iteration": iterations}))

        return {"messages": messages, "decision_log": log, "agent_iterations": iterations + 1}

    # ── Nœud 3 : Exécution des outils (Action) ───────────────────────────────
    def agent_action(state: RAGState) -> dict:
        qid = state["question_id"]
        log = list(state.get("decision_log", []))
        messages = list(state.get("messages", []))
        all_docs = list(state.get("all_docs", []))
        seen_keys = set(state.get("seen_keys", set()))
        seen_queries = list(state.get("seen_queries", []))
        consecutive_no_progress = state.get("consecutive_no_progress", 0)
        filter_ = state.get("source_filter")
        target_sources = state.get("target_sources", [])
        action_new_docs = 0
        action_made_progress = False
        
        # Le dernier message est celui du modèle avec potentiellement des appels de fonction
        model_content = messages[-1]

        fn_calls = model_content.get("tool_calls") or []
        fn_response_parts = []

        for tc in fn_calls:
            fc_name = tc["function"]["name"]
            fc_args = json.loads(tc["function"]["arguments"])
            result = {}
            if fc_name == "search_documents":
                query = fc_args.get("query", "")
                requested_source_name = (fc_args.get("source_name") or "").strip()
                resolved_source = (
                    filter_
                    or _resolve_source_by_name(requested_source_name, target_sources)
                    or _resolve_source_by_name(requested_source_name, state.get("available_sources", []))
                )
                query_signature = f"{requested_source_name.lower()}::{query.lower().strip()}"
                # Déduplication : vérifier AVANT d'ajouter à l'historique
                is_duplicate = any(
                    q.lower().strip() == query_signature for q, _ in seen_queries
                )
                seen_queries.append((query_signature, 1.0))

                if is_duplicate:
                    result = {"found": 0, "results": [], "notice": "Requête déjà effectuée, essaie une formulation différente."}
                    log.append(_log_entry("agent.action", f"Skip query (duplicate): {query[:50]} / {requested_source_name or 'all'}"))
                else:
                    try:
                        vector = _embed_with_timeout(query.strip() or " ", timeout=llm_timeout)
                        sem_docs = _weaviate_with_retry(
                            weaviate_store.hybrid_search, query=query, query_vector=vector,
                            top_k=top_k_retrieve, alpha=hybrid_alpha, source=resolved_source
                        )
                        kw_docs = _weaviate_with_retry(
                            weaviate_store.hybrid_search, query=query, query_vector=vector,
                            top_k=top_k_retrieve, alpha=max(0.0, round(hybrid_alpha - 0.3, 1)), source=resolved_source
                        )
                        merged = _weighted_rrf([sem_docs, kw_docs], [1.0, 0.5])

                        new_count = 0
                        chunks_info = []
                        for doc in merged:
                            k = (doc.get("source", ""), int(doc.get("chunk_index", -1)))
                            if k not in seen_keys:
                                all_docs.append(doc)
                                seen_keys.add(k)
                                new_count += 1

                            chunks_info.append({
                                "chunk_index": doc.get("chunk_index"),
                                "source_name": Path(doc.get("source", "")).name,
                                "kind": doc.get("kind", "text"),
                                "title_path": doc.get("title_path", ""),
                                "content": doc.get("page_content", ""),
                                "prev_chunk": doc.get("prev_chunk", -1),
                                "next_chunk": doc.get("next_chunk", -1),
                            })
                        result = {"found": len(merged), "new_chunks": new_count, "results": chunks_info[:10]}
                        action_new_docs += new_count
                        action_made_progress = action_made_progress or (new_count > 0)
                        log.append(_log_entry(
                            "agent.action",
                            f"Recherche '{query[:50]}' sur {requested_source_name or 'toute la base'} → {len(merged)} hits ({new_count} nouveaux)",
                            {
                                "query": query,
                                "source_name": requested_source_name or None,
                                "resolved_source": resolved_source,
                                "found": len(merged),
                                "new": new_count,
                            },
                        ))
                    except Exception as e:
                        error_msg = f"Recherche échouée: {e}"
                        result = {"error": error_msg}
                        logger.error("[{}] agent_action — search '{}': {}", qid, query[:50], e)
                        log.append(_log_entry(
                            "agent.action",
                            f"ERREUR search '{query[:50]}' sur {requested_source_name or 'toute la base'}: {e}",
                            {"query": query, "source_name": requested_source_name or None, "error": str(e)},
                        ))
            
            elif fc_name == "get_neighboring_chunk":
                src_name = fc_args.get("source_name", "")
                idx = int(fc_args.get("chunk_index", -1))
                if not (_CHUNK_INDEX_MIN <= idx <= _CHUNK_INDEX_MAX):
                    result = {"error": "Index invalide"}
                else:
                    # Résolution file full path
                    source_full = filter_ or next((d["source"] for d in all_docs if Path(d.get("source", "")).name == src_name), None)
                    source_full = source_full or next((s for s in state.get("available_sources", []) if Path(s).name == src_name), src_name)
                    
                    try:
                        chunk = _weaviate_with_retry(weaviate_store.get_chunk_by_index, source_full, idx)
                        if chunk:
                            k = (chunk.get("source", ""), int(chunk.get("chunk_index", -1)))
                            if k not in seen_keys:
                                chunk["_expanded"] = True
                                all_docs.append(chunk)
                                seen_keys.add(k)
                                action_new_docs += 1
                                action_made_progress = True
                            result = {"found": True, "chunk": {"chunk_index": idx, "content": chunk.get("page_content", "")}}
                            log.append(_log_entry("agent.action", f"Voisin {src_name} idx {idx}"))
                        else:
                            result = {"found": False}
                    except Exception as e:
                        result = {"error": str(e)}
                        logger.error("[{}] agent_action — get_neighboring_chunk {} idx {}: {}", qid, src_name, idx, e)
                        log.append(_log_entry(
                            "agent.action",
                            f"ERREUR get_chunk {src_name} idx {idx}: {e}",
                            {"error": str(e)},
                        ))
            
            else:
                result = {"error": "Outil inconnu."}

            fn_response_parts.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result, ensure_ascii=False),
            })

        messages.extend(fn_response_parts)
        if action_made_progress:
            consecutive_no_progress = 0
        else:
            consecutive_no_progress += 1
            log.append(_log_entry(
                "agent.guard",
                f"Aucune progression utile sur cette action ({consecutive_no_progress}/{MAX_NO_PROGRESS_STEPS})",
                {"new_docs": action_new_docs, "seen_docs": len(all_docs)},
            ))
        
        return {
            "messages": messages, 
            "decision_log": log,
            "all_docs": all_docs,
            "seen_keys": seen_keys,
            "seen_queries": seen_queries,
            "consecutive_no_progress": consecutive_no_progress,
            "last_action_new_docs": action_new_docs,
        }

    # ── Routeur après action : compression si budget token dépassé ───────────
    def route_after_action(state: RAGState) -> str:
        """Après agent_action : décide si le contexte doit être compressé."""
        messages = state.get("messages", [])
        msg_chars = sum(
            len(str(m.get("content") or "")) + len(str(m.get("tool_calls") or ""))
            for m in messages
        )
        doc_chars = sum(len(doc.get("page_content", "")) for doc in state.get("all_docs", []))
        estimated_tokens = (msg_chars + doc_chars) // 4
        no_progress = state.get("consecutive_no_progress", 0)
        doc_count = len(state.get("all_docs", []))

        if no_progress >= MAX_NO_PROGRESS_STEPS:
            logger.info(
                "[{}] Arrêt anticipé de la boucle : {} action(s) sans progrès",
                state["question_id"], no_progress,
            )
            return "consolidate"

        if doc_count >= top_k_final and state.get("last_action_new_docs", 0) == 0:
            logger.info(
                "[{}] Arrêt anticipé : {} chunks collectés et dernière action sans nouveauté",
                state["question_id"], doc_count,
            )
            return "consolidate"

        if not enable_compression:
            if estimated_tokens > BASE_TOKEN_THRESHOLD:
                logger.info(
                    "[{}] Compression désactivée — seuil dépassé ({} tokens estimés), retour à agent_reason",
                    state["question_id"], estimated_tokens,
                )
            return "agent_reason"

        if estimated_tokens > BASE_TOKEN_THRESHOLD:
            logger.info(
                "[{}] Seuil de compression atteint ({} tokens estimés) → compress_context",
                state["question_id"], estimated_tokens,
            )
            return "compress_context"
        return "agent_reason"

    # ── Nœud : Compression du contexte ───────────────────────────────────────
    def compress_context(state: RAGState) -> dict:
        """Compresse les docs récupérés en résumé structuré quand le budget token est dépassé.

        Après compression, remet messages à [] pour que agent_reason reparte avec
        le contexte compressé injecté dans son prompt initial.
        """
        qid = state["question_id"]
        log = list(state.get("decision_log", []))
        all_docs = state.get("all_docs", [])
        question = state["question"]
        existing_summary = state.get("context_summary", "")

        content_parts = [
            _build_context_entry(i, doc)
            for i, doc in enumerate(all_docs, start=1)
        ]
        raw_content = "\n\n".join(content_parts)

        compress_input = (
            f"RÉSUMÉ EXISTANT :\n{existing_summary}\n\nNOUVEAU CONTENU :\n{raw_content}"
            if existing_summary
            else raw_content
        )

        context_summary = existing_summary
        try:
            resp = _llm_call_with_timeout(
                messages=[
                    {"role": "system", "content": _COMPRESSION_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Question : {question}\n\n"
                            f"Contenu à compresser :\n{compress_input[:40_000]}"
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=1200,
            )
            context_summary = resp.choices[0].message.content or existing_summary
            log.append(_log_entry(
                "compress",
                f"Contexte compressé : {len(raw_content)} → {len(context_summary)} chars",
                {"raw_chars": len(raw_content), "summary_chars": len(context_summary)},
            ))
        except Exception as exc:
            logger.warning("[{}] compress_context — erreur : {}", qid, exc)
            log.append(_log_entry("compress", f"Compression échouée : {exc}"))

        # Réinitialise les messages : agent_reason repartira avec la summary injectée
        return {"context_summary": context_summary, "messages": [], "decision_log": log}

    # ── Nœud C : Fin de la boucle  ───────────────────────────────────────────
    def _combine_all_docs(docs: list[dict]) -> list[dict]:
        return _combine_chunks([docs])

    # ── Routeur pour la logique conditionnelle ────────────────────────────────
    def route_agent(state: RAGState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return "agent_reason"
            
        # Si la limite d'itérations est atteinte
        if state.get("agent_iterations", 0) >= max_agent_iter:
            logger.warning("[{}] Routeur — Fin de boucle (max itérations)", state["question_id"])
            return "rerank_prep"
        
        last_msg = messages[-1]

        # L'assistant a émis des appels d'outils → exécuter
        if isinstance(last_msg, dict) and last_msg.get("role") == "assistant":
            if last_msg.get("tool_calls"):
                return "agent_action"
            # Pas d'appel d'outil → l'agent a terminé
            return "rerank_prep"

        # Dernier message = réponse d'outil → retour au raisonnement
        return "agent_reason"
        
    def consolidate_chunks(state: RAGState) -> dict:
        """Consolide la liste complète en chunks retrievés (uniquement à la fin)."""
        docs = state.get("all_docs", [])
        log = list(state.get("decision_log", []))
        
        # Fallback de ségrégation inspiré de 'fallback_response' du nouveau module
        if not docs:
            log.append(_log_entry("agent.fallback", "Aucun doc trouvé. Fallback direct."))
            try:
                vector = _embed_with_timeout(state["question"], timeout=llm_timeout)
                docs = _weaviate_with_retry(weaviate_store.hybrid_search, query=state["question"], query_vector=vector, top_k=top_k_retrieve, alpha=0.5, source=state.get("source_filter"))
            except Exception:
                pass
                
        retrieved_docs = _combine_all_docs(docs)
        log.append(_log_entry("agent.loop_end", f"{len(retrieved_docs)} chunks prêts pour rerank."))
        return {"retrieved_docs": retrieved_docs, "decision_log": log}


    # ── Nœud 4 : reranking (Cohere si dispo, fallback LLM) ───────────────────
    def rerank(state: RAGState) -> dict:
        qid      = state["question_id"]
        docs     = state.get("retrieved_docs", [])
        question = state["question"]
        log      = list(state.get("decision_log", []))

        if not docs:
            log.append(_log_entry("rerank", "Aucun document à reranker"))
            return {"reranked_docs": [], "decision_log": log}

        # ── Cohere Rerank ─────────────────────────────────────────────────────
        if cohere_client is not None:
            try:
                docs_texts = [
                    (doc.get("page_content") or "")[:2048]
                    for doc in docs
                ]
                response = cohere_client.rerank(
                    query=question,
                    documents=docs_texts,
                    model="rerank-multilingual-v3.0",
                    top_n=len(docs),
                    return_documents=False,
                )
                scored_docs = [{**doc} for doc in docs]
                for result in response.results:
                    scored_docs[result.index]["_rerank_score"] = result.relevance_score

                reranked = sorted(
                    scored_docs,
                    key=lambda d: float(d.get("_rerank_score", 0.0)),
                    reverse=True,
                )[:20]

                log.append(_log_entry(
                    "rerank.cohere",
                    f"Cohere Rerank : {len(docs)} → {len(reranked)} chunks pertinents",
                    {"n_input": len(docs), "n_output": len(reranked)},
                ))
                return {"reranked_docs": reranked, "decision_log": log}

            except Exception as exc:
                logger.warning("[{}] Cohere Rerank échoué, fallback LLM : {}", qid, exc)
                log.append(_log_entry(
                    "rerank.cohere",
                    f"Cohere Rerank échoué : {exc} — fallback LLM",
                    {"error": str(exc)},
                ))

        # ── Fallback LLM reranking ────────────────────────────────────────────
        summaries = []
        for i, doc in enumerate(docs):
            kind  = doc.get("kind", "text")
            title = (doc.get("title_path") or "").strip()
            text_ = (doc.get("page_content") or "")[:400].replace("\n", " ")
            extra = " *(contexte étendu)*" if doc.get("_expanded") else ""
            summaries.append(f"[{i}] {kind}{extra} | {title} | {text_}…")

        prompt = (
            f"Question : {question}\n\n"
            "Note la pertinence de chaque extrait de 0 à 10 "
            "(10 = répond parfaitement à la question, 0 = hors sujet).\n"
            "Réponds UNIQUEMENT avec un tableau JSON compact d'entiers sur UNE SEULE LIGNE, "
            "dans l'ordre exact des extraits (sans balise markdown, sans indentation, sans saut de ligne).\n"
            f"Exemple pour {len(docs)} extraits : [{', '.join(['8'] * min(len(docs), 4))}"
            f"{'...' if len(docs) > 4 else ''}]\n\n"
            + "\n".join(summaries)
        )

        scores: list[int] = []
        try:
            resp = _llm_call_with_timeout(
                messages=[
                    {"role": "system", "content": "Tu es un expert en pertinence documentaire."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=max(600, len(docs) * 15),
                timeout=llm_timeout,
            )
            text_resp = resp.choices[0].message.content or "[]"
            try:
                parsed = _parse_json_llm(text_resp)
                if not isinstance(parsed, list):
                    raise ValueError(f"Pas une liste JSON : {text_resp[:50]}")
                scores = [int(s) for s in parsed]
            except Exception:
                nums = re.findall(r"\b(?:10|[0-9])\b", text_resp)
                scores = [int(n) for n in nums] if nums else []
        except TimeoutError:
            logger.warning("[{}] rerank LLM — timeout", qid)
        except Exception as exc:
            logger.warning("[{}] rerank LLM — erreur : {}", qid, exc)

        if not scores:
            scores = [5] * len(docs)

        if len(scores) < len(docs):
            scores += [0] * (len(docs) - len(scores))
        elif len(scores) > len(docs):
            scores = scores[: len(docs)]

        scored_docs = [{**doc} for doc in docs]
        for i, doc in enumerate(scored_docs):
            doc["_rerank_score"] = scores[i]

        reranked = sorted(
            scored_docs,
            key=lambda d: float(d.get("_rerank_score", 0.0)),
            reverse=True,
        )[:20]

        log.append(_log_entry(
            "rerank.llm",
            f"LLM Rerank : {len(docs)} → {len(reranked)} chunks",
            {"n_input": len(docs), "n_output": len(reranked)},
        ))
        return {"reranked_docs": reranked, "decision_log": log}


    # ── Nœud 5 : génération ──────────────────────────────────────────────────
    def generate(state: RAGState) -> dict:
        qid      = state["question_id"]
        docs     = state.get("reranked_docs", [])
        question = state["question"]
        log      = list(state.get("decision_log", []))

        if not docs:
            answer = "Aucun extrait pertinent n'a été trouvé pour répondre à votre question."
            log.append(_log_entry("generate", "Aucun document disponible"))
            return {"answer": answer, "error": None, "decision_log": log}

        context = "\n\n".join(
            _build_context_entry(i, doc) for i, doc in enumerate(docs, start=1)
        )

        user_content = f"Contexte :\n{context}\n\nQuestion : {question}"
        if state.get("conversation_summary"):
            user_content = (
                f"Contexte de la conversation précédente :\n{state['conversation_summary']}\n\n"
                + user_content
            )

        try:
            resp = _llm_call_with_timeout(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
                timeout=llm_timeout * 2,
            )
            answer = resp.choices[0].message.content or ""
            if not answer:
                reason = resp.choices[0].finish_reason or "UNKNOWN"
                raise RuntimeError(f"Réponse vide (finish_reason: {reason})")
        except Exception as exc:
            msg = f"Erreur de génération : {exc}"
            logger.error("[{}] generate — {}", qid, exc)
            log.append(_log_entry("generate", msg, {"error": str(exc)}))
            return {"answer": msg, "error": msg, "decision_log": log}

        log.append(_log_entry(
            "generate",
            f"Réponse produite ({len(answer)} caractères)",
            {"n_chars": len(answer), "n_sources": len(docs)},
        ))
        return {"answer": answer, "error": None, "decision_log": log}

    builder = StateGraph(RAGState)
    builder.add_node("analyze_and_plan", analyze_and_plan)
    builder.add_node("agent_reason",     agent_reason)
    builder.add_node("agent_action",     agent_action)
    builder.add_node("compress_context", compress_context)
    builder.add_node("consolidate",      consolidate_chunks)
    builder.add_node("rerank",           rerank)
    builder.add_node("generate",         generate)

    builder.add_edge(START,              "analyze_and_plan")
    builder.add_edge("analyze_and_plan", "agent_reason")

    # Boucle ReAct : Reason -> Action ou Sortie
    builder.add_conditional_edges("agent_reason", route_agent, {
        "agent_action": "agent_action",
        "rerank_prep": "consolidate"
    })

    # Action -> compression si budget dépassé, sinon retour au raisonnement
    builder.add_conditional_edges("agent_action", route_after_action, {
        "compress_context": "compress_context",
        "agent_reason":     "agent_reason",
        "consolidate":      "consolidate",
    })
    builder.add_edge("compress_context", "agent_reason")
    
    builder.add_edge("consolidate", "rerank")
    builder.add_edge("rerank",      "generate")
    builder.add_edge("generate",    END)

    return builder.compile()


# ── Classe wrapper ────────────────────────────────────────────────────────────

class RAGAgent:
    """Agent RAG basé sur LangGraph.

    Interface publique compatible avec RAGChain.query().
    """

    def __init__(
        self,
        weaviate_store,
        openai_key: str,
        cohere_key: str | None = None,
        *,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4.1",
        top_k_retrieve: int = 20,
        top_k_final: int = 5,
        hybrid_alpha: float = 0.5,
        max_tokens: int = 4000,
        max_agent_iter: int = 60,
        llm_timeout: float = 30.0,
        enable_compression: bool = True,
    ) -> None:
        self._store = weaviate_store
        self._graph = build_rag_graph(
            weaviate_store  = weaviate_store,
            openai_key      = openai_key,
            cohere_key      = cohere_key,
            embedding_model = embedding_model,
            llm_model       = llm_model,
            top_k_retrieve  = top_k_retrieve,
            top_k_final     = top_k_final,
            hybrid_alpha    = hybrid_alpha,
            max_tokens      = max_tokens,
            max_agent_iter  = max_agent_iter,
            llm_timeout     = llm_timeout,
            enable_compression = enable_compression,
        )

    def stream_query(
        self,
        question: str,
        source: str | None = None,
        conversation_summary: str = "",
    ):
        """Exécute le pipeline agentique et yields les événements au fil de l'eau.

        Args:
            question:             Question de l'utilisateur.
            source:               Filtre optionnel sur un document précis.
            conversation_summary: Résumé des échanges précédents (fourni par l'UI).
        """
        try:
            available_sources = _weaviate_with_retry(self._store.list_sources)
        except Exception:
            available_sources = []

        question_id = str(uuid.uuid4())

        initial_state: RAGState = {
            "question_id":          question_id,
            "question":             question,
            "available_sources":    available_sources,
            "source_filter":        source,
            "target_sources":       [source] if source else [],
            "sub_queries":          [],
            "messages":             [],
            "all_docs":             [],
            "seen_keys":            set(),
            "seen_queries":         [],
            "agent_iterations":     0,
            "consecutive_no_progress": 0,
            "last_action_new_docs": 0,
            "conversation_summary": conversation_summary,
            "context_summary":      "",
            "retrieved_docs":       [],
            "reranked_docs":        [],
            "answer":               "",
            "decision_log":         [],
            "error":                None,
        }

        # Yield les events en temps réel
        for event in self._graph.stream(initial_state, stream_mode="updates"):
            yield event


    def query(
        self,
        question: str,
        source: str | None = None,
    ) -> dict:
        """Exécute le pipeline agentique et retourne la réponse.

        Retourne un dict :

            - ``answer``        (str)         — réponse finale
            - ``sources``       (list[dict])  — chunks rerankés (du plus au moins pertinent)
            - ``question``      (str)
            - ``question_id``   (str)         — UUID de corrélation pour les logs
            - ``n_retrieved``   (int)         — chunks avant reranking
            - ``decision_log``  (list[dict])  — trace structurée {step, ts, message, metadata}
            - ``error``         (str|None)    — message d'erreur fatale, None si succès
        """
        try:
            available_sources = _weaviate_with_retry(self._store.list_sources)
        except Exception:
            available_sources = []

        question_id = str(uuid.uuid4())

        initial_state: RAGState = {
            "question_id":          question_id,
            "question":             question,
            "available_sources":    available_sources,
            "source_filter":        source,
            "target_sources":       [source] if source else [],
            "sub_queries":          [],
            "messages":             [],
            "all_docs":             [],
            "seen_keys":            set(),
            "seen_queries":         [],
            "agent_iterations":     0,
            "consecutive_no_progress": 0,
            "last_action_new_docs": 0,
            "conversation_summary": "",
            "context_summary":      "",
            "retrieved_docs":       [],
            "reranked_docs":        [],
            "answer":               "",
            "decision_log":         [],
            "error":                None,
        }

        final = self._graph.invoke(initial_state)

        return {
            "answer":       final.get("answer", ""),
            "sources":      final.get("reranked_docs", []),
            "question":     question,
            "question_id":  question_id,
            "n_retrieved":  len(final.get("retrieved_docs", [])),
            "decision_log": final.get("decision_log", []),
            "error":        final.get("error"),
        }
