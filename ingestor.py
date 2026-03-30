"""Ingesteur PDF → chunks → embeddings OpenAI → Weaviate.

Deux modes selon ce qui est installé :
  • openingestion  (défaut) — pipeline complet DoclingChef / MinerUChef + chunker
  • simple         (fallback) — extraction page par page avec PyMuPDF + découpage naïf

Embeddings : OpenAI Embeddings (text-embedding-3-small, text-embedding-3-large, etc.)
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from loguru import logger

# ── embedding helper (OpenAI) ────────────────────────────────────────────────

def _embed_texts(
    texts: list[str],
    openai_key: str,
    model: str,
    batch_size: int = 2048,
    progress_cb: Callable[[str], None] | None = None,
) -> list[list[float]]:
    """Encode une liste de textes en vecteurs via l'API OpenAI Embeddings.

    openai.embeddings.create accepte jusqu'à 2048 textes par appel.
    Les textes vides sont remplacés par un espace pour éviter les erreurs.
    """
    from openai import OpenAI

    client = OpenAI(api_key=openai_key)
    # OpenAI rejette les chaînes vides — on les remplace par un espace
    sanitized = [t if t and t.strip() else " " for t in texts]
    vectors: list[list[float]] = []
    total = len(sanitized)

    for i in range(0, total, batch_size):
        batch = sanitized[i : i + batch_size]
        result = client.embeddings.create(model=model, input=batch)
        # Les embeddings sont retournés dans l'ordre des entrées (triés par index)
        vectors.extend([e.embedding for e in sorted(result.data, key=lambda e: e.index)])
        if progress_cb:
            progress_cb(f"  Embedding {min(i + batch_size, total)}/{total}…")

    return vectors


# ── mode openingestion ────────────────────────────────────────────────────────

def _ingest_with_openingestion(
    pdf_path: Path,
    source: str,
    parser: str,
    strategy: str,
    openai_key: str,
    embedding_model: str,
    progress_cb: Callable[[str], None],
) -> tuple[list[dict], list[list[float]]]:
    from openingestion import ingest

    progress_cb(f"Parsing '{pdf_path.name}' avec {parser} / {strategy}…")
    chunks = ingest(pdf_path, parser=parser, strategy=strategy, image_mode="path")
    progress_cb(f"{len(chunks)} chunks extraits.")

    import json as _json

    chunk_dicts = []
    for c in chunks:
        page_idx = c.position_int[0][0] if c.position_int else 0
        extras   = c.extras or {}

        # Légendes et notes de bas de page (listes de strings → JSON)
        captions  = extras.get("captions",  getattr(c, "captions",  [])) or []
        footnotes = extras.get("footnotes", getattr(c, "footnotes", [])) or []

        # HTML enrichi pour tableaux / équations
        html = extras.get("html", getattr(c, "html", "")) or ""

        chunk_dicts.append({
            "page_content":  c.page_content,
            "source":        source,
            "kind":          c.kind.value if hasattr(c.kind, "value") else str(c.kind),
            "title_path":    c.title_path or "",
            "title_level":   c.title_level,
            "chunk_index":   c.chunk_index,
            "reading_order": c.reading_order,
            "prev_chunk":    c.prev_chunk_index if c.prev_chunk_index is not None else -1,
            "next_chunk":    c.next_chunk_index if c.next_chunk_index is not None else -1,
            "page_idx":      page_idx,
            "token_count":   c.token_count,
            "html":          html,
            "captions_json": _json.dumps(captions,  ensure_ascii=False),
            "footnotes_json":_json.dumps(footnotes, ensure_ascii=False),
        })

    texts = [d["page_content"] for d in chunk_dicts]
    progress_cb("Embedding des chunks (OpenAI)…")
    vectors = _embed_texts(texts, openai_key, embedding_model, progress_cb=progress_cb)

    return chunk_dicts, vectors


# ── mode fallback (PyMuPDF) ───────────────────────────────────────────────────

def _ingest_simple(
    pdf_path: Path,
    source: str,
    openai_key: str,
    embedding_model: str,
    chunk_size: int = 800,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[list[dict], list[list[float]]]:
    try:
        import fitz
    except ImportError:
        raise ImportError("pymupdf est requis en mode fallback : pip install pymupdf")

    _cb = progress_cb or (lambda m: None)
    _cb(f"Extraction texte de '{pdf_path.name}' (mode simple)…")

    doc = fitz.open(str(pdf_path))
    raw_pages: list[tuple[int, str]] = []
    for page in doc:
        text = page.get_text("text").strip()
        if text:
            raw_pages.append((page.number, text))
    doc.close()

    _cb(f"{len(raw_pages)} pages avec du texte.")

    # Découpage naïf par blocs de ~chunk_size caractères
    chunk_dicts: list[dict] = []
    idx = 0
    for page_idx, text in raw_pages:
        for start in range(0, len(text), chunk_size):
            block = text[start : start + chunk_size].strip()
            if not block:
                continue
            chunk_dicts.append({
                "page_content":   block,
                "source":        source,
                "kind":          "text",
                "title_path":    "",
                "title_level":   0,
                "chunk_index":   idx,
                "reading_order": idx,
                "prev_chunk":    idx - 1,
                "next_chunk":    -1,   # inconnu avant la fin
                "page_idx":      page_idx,
                "token_count":   len(block) // 4,
                "html":          "",
                "captions_json": "[]",
                "footnotes_json":"[]",
            })
            idx += 1

    _cb(f"{len(chunk_dicts)} chunks créés (mode simple).")
    texts = [d["page_content"] for d in chunk_dicts]
    _cb("Embedding des chunks (OpenAI)…")
    vectors = _embed_texts(texts, openai_key, embedding_model, progress_cb=_cb)

    return chunk_dicts, vectors


# ── point d'entrée public ─────────────────────────────────────────────────────

def ingest_pdf(
    pdf_path: Path,
    weaviate_store,
    openai_key: str,
    embedding_model: str = "text-embedding-3-small",
    chunking_strategy: str = "by_token",
    parser: str = "docling",
    progress_cb: Callable[[str], None] | None = None,
    force_simple: bool = False,
) -> int:
    """Parse un PDF, embed ses chunks via OpenAI Embeddings et les stocke dans Weaviate.

    Parameters
    ----------
    pdf_path:
        Chemin vers le fichier PDF.
    weaviate_store:
        Instance connectée de ``WeaviateStore``.
    openai_key:
        Clé API OpenAI.
    embedding_model:
        Modèle d'embedding OpenAI (défaut : text-embedding-3-small).
    chunking_strategy:
        Stratégie openingestion : by_token, by_sentence, by_block…
    parser:
        Parser openingestion : docling ou mineru.
    progress_cb:
        Callback appelé avec des messages de progression (pour l'UI).
    force_simple:
        Si True, utilise le mode PyMuPDF même si openingestion est dispo.

    Returns
    -------
    int
        Nombre de chunks stockés.
    """
    _cb = progress_cb or (lambda msg: logger.info(msg))
    source = str(pdf_path.resolve())

    # Supprimer les éventuels chunks existants pour ce fichier
    weaviate_store.delete_source(source)

    if force_simple:
        chunk_dicts, vectors = _ingest_simple(
            pdf_path, source, openai_key, embedding_model, progress_cb=_cb
        )
    else:
        try:
            chunk_dicts, vectors = _ingest_with_openingestion(
                pdf_path, source, parser, chunking_strategy,
                openai_key, embedding_model, _cb,
            )
        except ImportError:
            logger.warning(
                "openingestion introuvable — passage en mode simple (PyMuPDF)."
            )
            _cb("⚠️ openingestion non installé, mode simple activé.")
            chunk_dicts, vectors = _ingest_simple(
                pdf_path, source, openai_key, embedding_model, progress_cb=_cb
            )

    _cb("Stockage dans Weaviate…")
    n = weaviate_store.insert_chunks(chunk_dicts, vectors)
    _cb(f"✅ {n} chunks indexés pour '{pdf_path.name}'.")
    return n


# ── ingestion depuis un fichier JSONL pré-découpé ────────────────────────────

def ingest_jsonl(
    jsonl_path: Path,
    weaviate_store,
    openai_key: str,
    embedding_model: str = "text-embedding-3-small",
    progress_cb: Callable[[str], None] | None = None,
    source_override: str | None = None,
) -> int:
    """Ingère un fichier JSONL de chunks pré-découpés (format openingestion) dans Weaviate.

    Chaque ligne JSON doit contenir au minimum ``page_content`` et ``source``.
    Les champs supplémentaires du format openingestion sont mappés vers le
    schéma Weaviate (prev_chunk_index → prev_chunk, position_int → page_idx, etc.).

    Parameters
    ----------
    jsonl_path:
        Chemin vers le fichier ``.jsonl``.
    weaviate_store:
        Instance connectée de ``WeaviateStore``.
    openai_key:
        Clé API OpenAI (pour les embeddings).
    embedding_model:
        Modèle OpenAI Embeddings.
    progress_cb:
        Callback de progression (UI).
    source_override:
        Remplace le champ ``source`` présent dans le JSONL.
        Utile quand le chemin stocké dans le fichier ne correspond plus
        à l'emplacement réel du PDF.
    """
    import json as _json

    _cb = progress_cb or (lambda msg: logger.info(msg))

    _cb(f"Lecture de '{jsonl_path.name}'…")
    raw_lines: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                raw_lines.append(_json.loads(line))

    if not raw_lines:
        raise ValueError(f"Fichier vide : {jsonl_path}")

    _cb(f"{len(raw_lines)} lignes lues.")

    # Détermine la source canonique
    first_source = raw_lines[0].get("source", str(jsonl_path))
    source = source_override or first_source

    # Supprime les éventuels chunks existants pour cette source
    weaviate_store.delete_source(source)

    chunk_dicts: list[dict] = []
    for raw in raw_lines:
        extras = raw.get("extras") or {}

        # page_idx depuis position_int (liste de tuples [page, x0, y0, x1, y1])
        pos = raw.get("position_int") or []
        page_idx = int(pos[0][0]) if pos and pos[0] else 0

        # Légendes : inferred_caption stocké dans extras
        inferred = extras.get("inferred_caption", "")
        captions: list[str] = [inferred] if inferred else []

        # HTML enrichi (tableaux/équations)
        html: str = extras.get("html", "") or ""

        chunk_dicts.append({
            "page_content":  raw.get("page_content") or "",
            "source":        source,
            "kind":          raw.get("kind") or "text",
            "title_path":    raw.get("title_path") or "",
            "title_level":   int(raw.get("title_level") or 0),
            "chunk_index":   int(raw.get("chunk_index") or 0),
            "reading_order": int(raw.get("reading_order") or 0),
            "prev_chunk":    int(raw["prev_chunk_index"]) if raw.get("prev_chunk_index") is not None else -1,
            "next_chunk":    int(raw["next_chunk_index"]) if raw.get("next_chunk_index") is not None else -1,
            "page_idx":      page_idx,
            "token_count":   int(raw.get("token_count") or 0),
            "html":          html,
            "captions_json": _json.dumps(captions, ensure_ascii=False),
            "footnotes_json": "[]",
        })

    texts = [d["page_content"] for d in chunk_dicts]
    _cb("Embedding des chunks (OpenAI)…")
    vectors = _embed_texts(texts, openai_key, embedding_model, progress_cb=_cb)

    _cb("Stockage dans Weaviate…")
    n = weaviate_store.insert_chunks(chunk_dicts, vectors)
    _cb(f"✅ {n} chunks indexés pour '{jsonl_path.name}'.")
    return n
