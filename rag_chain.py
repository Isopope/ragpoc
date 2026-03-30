"""RAG chain : hybride BM25+dense → génération OpenAI.

Pipeline complet :
  1. Embed question avec OpenAI Embeddings
  2. Hybrid search dans Weaviate (BM25 + HNSW, top_k résultats)
  3. Génération avec OpenAI Chat Completions (gpt-4.1, etc.)
"""
from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

_SYSTEM_PROMPT = """\
Tu es un assistant expert. Tu réponds aux questions en t'appuyant UNIQUEMENT \
sur les extraits de documents fournis dans le contexte.
- Exploite attentivement la structure et les métadonnées fournies dans chaque extrait (ex: chemins de titres/chapitres, libellés de tableaux ou données JSON en métadonnées) pour comprendre le contexte global.
- Si la réponse n'est pas dans le contexte, dis-le clairement sans inventer.
- Cite les passages pertinents si utile, y compris les numéros de page des documents si l'info est pertinente.
- Réponds dans la même langue que la question.
"""

# Le contexte est construit dynamiquement par _build_context_entry()


def _build_context_entry(i: int, doc: dict) -> str:
    """Construit une entrée de contexte riche à partir d'un chunk Weaviate."""
    page        = doc.get("page_idx", 0) + 1
    kind        = doc.get("kind", "text")
    score       = doc.get("_score", 0.0)
    title_path  = (doc.get("title_path") or "").strip()
    title_level = doc.get("title_level", 0)
    tok_count   = doc.get("token_count", 0)
    text        = (doc.get("page_content") or "").strip()
    html        = (doc.get("html") or "").strip()

    # Légendes et notes de bas de page (stockées en JSON)
    try:
        captions = json.loads(doc.get("captions_json") or "[]")
    except (json.JSONDecodeError, TypeError):
        captions = []
    try:
        footnotes = json.loads(doc.get("footnotes_json") or "[]")
    except (json.JSONDecodeError, TypeError):
        footnotes = []

    lines: list[str] = []

    # ── En-tête ────────────────────────────────────────────────────────────
    header = f"=== Extrait {i} | page {page} | {kind}"
    if score:
        header += f" | pertinence {score:.3f}"
    if tok_count:
        header += f" | ~{tok_count} tokens"
    lines.append(header)

    # ── Fil de section ─────────────────────────────────────────────────────
    if title_path:
        prefix = "#" * max(1, title_level) if title_level else "§"
        lines.append(f"{prefix} {title_path}")

    lines.append("")  # ligne vide avant le contenu

    # ── Contenu : HTML pour tableaux/équations, texte sinon ────────────────
    if kind in ("table", "equation") and html:
        lines.append(html)
    else:
        lines.append(text)

    # ── Légendes associées ─────────────────────────────────────────────────
    if captions:
        lines.append("")
        lines.append("[Légendes] " + " | ".join(captions))

    # ── Notes de bas de page ───────────────────────────────────────────────
    if footnotes:
        lines.append("[Notes]    " + " | ".join(footnotes))

    lines.append("===")
    return "\n".join(lines)


class RAGChain:
    """Pipeline RAG : embed → hybrid retrieve → rerank → generate."""

    def __init__(
        self,
        weaviate_store,
        openai_key: str,
        *,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4.1",
        top_k_final: int = 5,
        hybrid_alpha: float = 0.5,
        max_tokens: int = 1000,
    ) -> None:
        from openai import OpenAI

        self._store = weaviate_store
        self._client = OpenAI(api_key=openai_key)
        self._emb_model = embedding_model
        self._llm_model = llm_model
        self._top_k_final = top_k_final
        self._alpha = hybrid_alpha
        self._max_tokens = max_tokens

    def query(
        self,
        question: str,
        source: str | None = None,
    ) -> dict:
        """Répond à une question en s'appuyant sur les documents indexés.

        Parameters
        ----------
        question:
            La question posée par l'utilisateur.
        source:
            Chemin absolu du document sur lequel filtrer la recherche.
            Si ``None``, cherche dans tous les documents.

        Returns
        -------
        dict avec les clés :
            - ``answer``       (str)
            - ``sources``      (list[dict]) — chunks rerankés, du plus pertinent au moins
            - ``question``     (str)
            - ``n_retrieved``  (int) — nombre de chunks avant reranking
        """
        # 1. Embedding de la question (OpenAI Embeddings)
        q_text = question.strip() or " "
        emb_result = self._client.embeddings.create(
            model=self._emb_model,
            input=q_text,
        )
        q_vector: list[float] = emb_result.data[0].embedding

        # 2. Hybrid search (BM25 + HNSW) — top_k_final résultats
        docs = self._store.hybrid_search(
            query=question,
            query_vector=q_vector,
            top_k=self._top_k_final,
            alpha=self._alpha,
            source=source,
        )
        n_retrieved = len(docs)
        logger.debug(
            "Hybrid search : {} docs récupérés (alpha={}) pour : {!r}",
            n_retrieved,
            self._alpha,
            question[:60],
        )

        if not docs:
            return {
                "answer": (
                    "Aucun document n'est encore indexé. "
                    "Veuillez d'abord ingérer un PDF via la barre latérale."
                ),
                "sources": [],
                "question": question,
                "n_retrieved": 0,
            }

        # 3. Construction du contexte enrichi
        context_parts = [
            _build_context_entry(i, doc)
            for i, doc in enumerate(docs, start=1)
        ]
        context = "\n".join(context_parts)

        # 4. Génération
        resp = self._client.chat.completions.create(
            model=self._llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Contexte :\n{context}\n\nQuestion : {question}"},
            ],
            max_tokens=self._max_tokens,
            temperature=0.1,
        )
        answer = resp.choices[0].message.content or ""

        return {
            "answer": answer,
            "sources": docs,
            "question": question,
            "n_retrieved": n_retrieved,
        }
