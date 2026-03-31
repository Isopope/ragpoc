"""Interface Streamlit pour le POC RAG — Weaviate hybride + OpenAI."""
from __future__ import annotations

import atexit
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from loguru import logger

# Charger .env si présent
load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
WEAVIATE_HOST   = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT   = int(os.getenv("WEAVIATE_PORT", "8080"))
LLM_MODEL       = os.getenv("LLM_MODEL", "gpt-4.1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
TOP_K_RETRIEVE  = int(os.getenv("TOP_K_RETRIEVE", "20"))
TOP_K_FINAL     = int(os.getenv("TOP_K_FINAL", "5"))
HYBRID_ALPHA    = float(os.getenv("HYBRID_ALPHA", "0.5"))
MAX_TOKENS      = int(os.getenv("MAX_TOKENS", "1000"))
AGENT_BACKEND   = os.getenv("AGENT_BACKEND", "rag_pipeline")
UPLOADS_DIR     = Path(__file__).parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG POC",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Ressources partagées (cached à la session) ─────────────────────────────────

@st.cache_resource
def _get_store():
    from weaviate_store import WeaviateStore
    store = WeaviateStore(host=WEAVIATE_HOST, port=WEAVIATE_PORT)
    try:
        store.connect()
        atexit.register(store.close)
        return store
    except Exception as exc:
        logger.error("Connexion Weaviate échouée : {}", exc)
        return None


def _close_store() -> None:
    """Ferme proprement la connexion Weaviate avant de vider le cache."""
    try:
        s = _get_store()
        if s is not None:
            s.close()
    except Exception:
        pass


@st.cache_resource
def _get_openai():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None
    return api_key


@st.cache_resource
def _get_agent():
    store      = _get_store()
    openai_key = _get_openai()
    if store is None or openai_key is None:
        return None
    if AGENT_BACKEND == "elysia":
        from langgraph_implementation.rag_agent import ElysiaRAGAgent
        return ElysiaRAGAgent(
            weaviate_store  = store,
            openai_key      = openai_key,
            embedding_model = EMBEDDING_MODEL,
            llm_model       = LLM_MODEL,
        )

    from rag_agent import RAGAgent
    return RAGAgent(
        weaviate_store  = store,
        openai_key      = openai_key,
        embedding_model = EMBEDDING_MODEL,
        llm_model       = LLM_MODEL,
        top_k_retrieve  = TOP_K_RETRIEVE,
        top_k_final     = TOP_K_FINAL,
        hybrid_alpha    = HYBRID_ALPHA,
        max_tokens      = MAX_TOKENS,
    )


# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []         # historique du chat
if "selected_source" not in st.session_state:
    st.session_state.selected_source = None


# ════════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🔍 RAG POC")
    st.caption("Streamlit · Weaviate hybrid · OpenAI")
    st.divider()

    # ── Clés API ───────────────────────────────────────────────────────────────
    st.subheader("⚙️ Configuration")
    api_key_input = st.text_input(
        "OpenAI API Key (LLM + embeddings)",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        placeholder="sk-…",
    )
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
        if api_key_input != os.getenv("_LAST_OPENAI_KEY", ""):
            os.environ["_LAST_OPENAI_KEY"] = api_key_input
            _get_openai.clear()
            _get_agent.clear()  # le store reste ouvert, seul l'agent change

    st.divider()

    # ── Documents indexés ──────────────────────────────────────────────────
    store      = _get_store()
    openai_key = _get_openai()
    chain      = _get_agent()

    st.subheader("📚 Documents indexés")
    if store and store.is_ready():
        try:
            sources = store.list_sources()
        except Exception:
            sources = []

        if sources:
            source_names = {Path(s).name: s for s in sources}
            all_choice   = "— Tous les documents —"
            choice = st.selectbox(
                "Filtrer la recherche",
                [all_choice] + list(source_names.keys()),
            )
            st.session_state.selected_source = (
                source_names[choice] if choice != all_choice else None
            )

            if choice != all_choice:
                n_chunks = store.count(source_names[choice])
                st.caption(f"{n_chunks} chunks pour ce document.")
                if st.button("🗑️ Supprimer ce document", use_container_width=True):
                    store.delete_source(source_names[choice])
                    st.success("Document supprimé.")
                    st.rerun()
        else:
            st.caption("Aucun document indexé.")
            st.session_state.selected_source = None

        total = store.count()
        st.caption(f"Total : {total} chunks dans la base.")

        if total > 0:
            if st.button(
                "🗑️ Vider toute la base",
                use_container_width=True,
                type="secondary",
            ):
                store.reset_collection()
                _close_store()
                _get_store.clear()
                _get_agent.clear()
                st.success("Base vidée.")
                st.rerun()
    else:
        st.error("❌ Weaviate non disponible.")

    st.divider()

    # ── Upload & ingestion ────────────────────────────────────────────────
    st.subheader("📤 Ajouter un document")
    uploaded = st.file_uploader("Choisir un PDF ou un JSONL", type=["pdf", "jsonl"])

    is_jsonl = uploaded is not None and uploaded.name.lower().endswith(".jsonl")

    if not is_jsonl:
        col1, col2 = st.columns(2)
        with col1:
            parser = st.selectbox("Parser", ["docling", "mineru", "simple"])
        with col2:
            strategy = st.selectbox(
                "Découpage",
                ["by_token", "by_sentence", "by_block"],
            )
    else:
        parser = "docling"
        strategy = "by_token"

        # Champ optionnel pour remplacer le source path présent dans le JSONL
        source_override_input = st.text_input(
            "Source (chemin PDF original)",
            placeholder="Laisser vide pour utiliser la valeur du JSONL",
            help=(
                "Si le chemin stocké dans le JSONL ne correspond plus à l'emplacement "
                "réel du PDF, renseignez ici le chemin absolu correct."
            ),
        )

    if st.button(
        "▶️ Ingérer le document",
        disabled=(uploaded is None or store is None or openai_key is None),
        use_container_width=True,
        type="primary",
    ):
        # Sauvegarde dans uploads/
        dest = UPLOADS_DIR / uploaded.name
        dest.write_bytes(uploaded.getbuffer())

        status_placeholder = st.empty()
        log_lines: list[str] = []

        def _progress(msg: str) -> None:
            log_lines.append(msg)
            status_placeholder.text("\n".join(log_lines[-5:]))

        try:
            if is_jsonl:
                from ingestor import ingest_jsonl

                n = ingest_jsonl(
                    jsonl_path=dest,
                    weaviate_store=store,
                    openai_key=openai_key,
                    embedding_model=EMBEDDING_MODEL,
                    progress_cb=_progress,
                    source_override=source_override_input.strip() or None,
                )
            else:
                from ingestor import ingest_pdf

                n = ingest_pdf(
                    pdf_path=dest,
                    weaviate_store=store,
                    openai_key=openai_key,
                    embedding_model=EMBEDDING_MODEL,
                    chunking_strategy=strategy,
                    parser=parser if parser != "simple" else "docling",
                    force_simple=(parser == "simple"),
                    progress_cb=_progress,
                )
            st.success(f"✅ {n} chunks indexés pour « {uploaded.name} »")
            _close_store()
            _get_store.clear()
            _get_agent.clear()
            st.rerun()
        except Exception as exc:
            st.error(f"Erreur : {exc}")
            logger.exception("Erreur d'ingestion")

    st.divider()

    # ── Statuts ───────────────────────────────────────────────────────────
    weaviate_ok = store is not None and store.is_ready()
    openai_ok   = openai_key is not None
    st.caption(f"Weaviate : {'🟢 connecté'  if weaviate_ok else '🔴 déconnecté'}")
    st.caption(f"OpenAI   : {'🟢 configuré' if openai_ok   else '🔴 clé manquante'}")

    if st.button("🔄 Recharger", use_container_width=True):
        _close_store()
        _get_store.clear()
        _get_openai.clear()
        _get_agent.clear()
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
#  ZONE DE CHAT PRINCIPALE
# ════════════════════════════════════════════════════════════════════════════════

def _show_sources(docs: list[dict]) -> None:
    """Affiche les chunks source dans un expander."""
    with st.expander(f"📎 Sources utilisées ({len(docs)} extraits)", expanded=False):
        for i, doc in enumerate(docs, start=1):
            doc_name     = Path(doc.get("source", "?")).name
            page         = doc.get("page_idx", 0) + 1
            kind         = doc.get("kind", "text")
            rerank_score = doc.get("_rerank_score")
            hybrid_score = doc.get("_score")
            preview      = doc.get("page_content", "")[:300].strip()

            score_str = (
                f"rerank : `{rerank_score}/10`"
                if rerank_score is not None
                else f"score hybride : `{hybrid_score:.4f}`"
                if hybrid_score is not None
                else ""
            )
            expanded_badge = " · *(contexte étendu)*" if doc.get("_expanded") else ""
            st.markdown(
                f"**#{i}** — `{doc_name}` · page {page} · *{kind}*"
                + (f" · {score_str}" if score_str else "")
                + expanded_badge
            )
            st.text(preview + ("…" if len(doc.get("page_content", "")) > 300 else ""))
            if i < len(docs):
                st.divider()


def _show_decision_log(log: list[str]) -> None:
    """Affiche la trace des décisions de l'agent dans un expander."""
    if not log:
        return
    with st.expander("🧠 Décisions de l'agent", expanded=False):
        for step in log:
            st.caption(step)


def _build_conversation_summary() -> str:
    """Formate les derniers échanges comme résumé de conversation pour le pipeline RAG.

    Ne renvoie rien si la conversation est vide ou contient moins de 2 messages.
    Prend au maximum les 6 derniers messages (3 échanges Q/R) pour éviter
    un contexte trop verbeux.
    """
    msgs = st.session_state.get("messages", [])
    if len(msgs) < 2:
        return ""
    recent = msgs[-6:]
    lines = []
    for m in recent:
        role = "Utilisateur" if m["role"] == "user" else "Assistant"
        text = (m.get("content") or "")[:400].replace("\n", " ")
        lines.append(f"{role} : {text}")
    return "\n".join(lines)


st.title("💬 Chat RAG")

# Bannières d'état
if not api_key_input and not os.getenv("OPENAI_API_KEY"):
    st.info(
        "\U0001f448 Renseignez votre clé OpenAI dans la barre latérale pour commencer.",
        icon="ℹ️",
    )
elif not (store and store.is_ready()):
    st.warning(
        "Weaviate n'est pas accessible. "
        "Lancez `docker compose up -d` puis rechargez la page.",
        icon="⚠️",
    )
elif store and store.count() == 0:
    st.info(
        "👈 Aucun document indexé. Uploadez un PDF dans la barre latérale.",
        icon="📄",
    )

# Affichage de l'historique du chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            _show_sources(msg["sources"])
        if msg["role"] == "assistant" and msg.get("decision_log"):
            _show_decision_log(msg["decision_log"])

# Champ de saisie
prompt = st.chat_input(
    placeholder="Posez une question sur vos documents…",
    disabled=(chain is None),
)

if prompt:
    # Message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Réponse de l'assistant
    with st.chat_message("assistant"):
        answer = ""
        sources = []
        decision_log = []
        
        # Interface de chat inspirée par core/chat_interface.py
        with st.status("🔍 Initialisation de la recherche...", expanded=True) as status_box:
            try:
                # On itère via stream au lieu de query pour une UI dynamique
                # (Assurez-vous que rag_pipeline.stream_query existe)
                for event in chain.stream_query(
                    question=prompt,
                    source=st.session_state.selected_source,
                    conversation_summary=_build_conversation_summary(),
                ):
                    for node_name, state_update in event.items():
                        logs = state_update.get("decision_log", [])
                        if logs:
                            last_log = logs[-1]
                            message = last_log.get("message", "")
                            
                            if node_name == "analyze_and_plan":
                                status_box.update(label="📋 Analyse et Planification en cours...")
                                st.markdown(f"**Plan :** {message}")
                            elif node_name == "agent_reason":
                                status_box.update(label="🤔 Raisonnement de l'agent...")
                                st.markdown(f"*{message}*")
                            elif node_name == "agent_action":
                                status_box.update(label="🛠️ Exécution des actions...")
                                st.markdown(f"**Action :** {message}")
                            elif node_name == "rerank":
                                status_box.update(label="📊 Classement (Reranking) des résultats...")
                                st.markdown(f"**Rerank :** {message}")
                            elif node_name == "generate":
                                status_box.update(label="💡 Génération de la réponse finale...")
                                
                        if "answer" in state_update and state_update["answer"]:
                            answer = state_update["answer"]
                        if "reranked_docs" in state_update:
                            sources = state_update["reranked_docs"]
                        if "decision_log" in state_update:
                            decision_log = state_update["decision_log"]
                            
                status_box.update(label="✅ Recherche terminée !", state="complete", expanded=False)
            except Exception as exc:
                answer       = f"❌ Erreur : {exc}"
                status_box.update(label="❌ Erreur critique", state="error", expanded=True)
                logger.exception("Erreur lors de la requête RAG")

        st.markdown(answer)
        if sources:
            _show_sources(sources)
        if decision_log:
            _show_decision_log(decision_log)

    st.session_state.messages.append({
        "role":         "assistant",
        "content":      answer,
        "sources":      sources,
        "decision_log": decision_log,
    })
