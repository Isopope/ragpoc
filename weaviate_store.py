"""Weaviate vector store — wrapper minimaliste pour le POC RAG.

Collection unique : RagChunk
Vectorizer       : none (vecteurs Voyage AI fournis manuellement)
Index            : HNSW cosinus + BM25 (activé par défaut sur les propriétés TEXT)
Recherche        : hybrid (BM25 + dense), paramétrable via alpha
"""
from __future__ import annotations

from loguru import logger

import weaviate
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    Tokenization,
    VectorDistances,
)
from weaviate.classes.query import Filter, HybridFusion, MetadataQuery

COLLECTION_NAME = "RagChunk"


class WeaviateStore:
    """Client Weaviate encapsulant les opérations CRUD pour les RagChunks."""

    def __init__(self, host: str = "localhost", port: int = 8080) -> None:
        self._host = host
        self._port = port
        self._client: weaviate.WeaviateClient | None = None

    # ── connexion ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        self._client = weaviate.connect_to_local(host=self._host, port=self._port)
        logger.info("Connecté à Weaviate sur {}:{}", self._host, self._port)
        self._ensure_schema()

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def is_ready(self) -> bool:
        try:
            return self._client is not None and self._client.is_ready()
        except Exception:
            return False

    def _ensure_connected(self) -> None:
        """Reconnecte automatiquement si la connexion Weaviate est perdue."""
        try:
            if self._client and self._client.is_ready():
                return
        except Exception:
            pass
        logger.warning("Connexion Weaviate perdue — reconnexion sur {}:{}", self._host, self._port)
        try:
            if self._client:
                try:
                    self._client.close()
                except Exception:
                    pass
            self._client = weaviate.connect_to_local(host=self._host, port=self._port)
            logger.info("Reconnexion Weaviate réussie.")
        except Exception as exc:
            raise RuntimeError(f"Impossible de se reconnecter à Weaviate : {exc}") from exc

    # ── schéma ────────────────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        if self._client.collections.exists(COLLECTION_NAME):
            logger.debug("Collection {} déjà présente.", COLLECTION_NAME)
            return

        self._client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE
            ),
            properties=[
                # ── champs indexés BM25 (Tokenization.WORD) ───────────────
                Property(
                    name="page_content",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.WORD,
                ),
                Property(
                    name="title_path",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.WORD,
                ),
                # ── métadonnées structurelles (filtrage / affichage) ──────
                Property(name="source",        data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="kind",          data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="chunk_index",   data_type=DataType.INT),
                Property(name="page_idx",      data_type=DataType.INT),
                Property(name="token_count",   data_type=DataType.INT),
                Property(name="title_level",   data_type=DataType.INT),
                Property(name="reading_order", data_type=DataType.INT),
                Property(name="prev_chunk",    data_type=DataType.INT),
                Property(name="next_chunk",    data_type=DataType.INT),
                # ── contenus enrichis (pas de BM25 — markup/JSON) ─────────
                # index_searchable=False désactive l'index inversé BM25
                Property(
                    name="html",
                    data_type=DataType.TEXT,
                    index_searchable=False,
                    skip_vectorization=True,
                ),
                Property(
                    name="captions_json",
                    data_type=DataType.TEXT,
                    index_searchable=False,
                    skip_vectorization=True,
                ),
                Property(
                    name="footnotes_json",
                    data_type=DataType.TEXT,
                    index_searchable=False,
                    skip_vectorization=True,
                ),
            ],
        )
        logger.info("Collection {} créée.", COLLECTION_NAME)

    def reset_collection(self) -> None:
        """Supprime et recrée la collection (dev / tests)."""
        if self._client.collections.exists(COLLECTION_NAME):
            self._client.collections.delete(COLLECTION_NAME)
            logger.warning("Collection {} supprimée.", COLLECTION_NAME)
        self._ensure_schema()

    # ── écriture ──────────────────────────────────────────────────────────────

    def insert_chunks(
        self,
        chunks: list[dict],
        vectors: list[list[float]],
    ) -> int:
        """Insère des chunks avec leurs vecteurs dans Weaviate.

        Returns le nombre d'objets insérés avec succès.
        """
        collection = self._client.collections.get(COLLECTION_NAME)
        inserted = 0
        failed = 0

        with collection.batch.dynamic() as batch:
            for chunk, vector in zip(chunks, vectors):
                batch.add_object(properties=chunk, vector=vector)
                inserted += 1

        # Le batch peut avoir des erreurs silencieuses → on les log
        if hasattr(batch, "number_errors") and batch.number_errors:
            failed = batch.number_errors
            logger.warning("{} erreurs lors de l'insertion.", failed)

        logger.info("Inséré {} chunk(s) dans Weaviate.", inserted - failed)
        return inserted - failed

    def delete_source(self, source: str) -> int:
        """Supprime tous les chunks associés à une source donnée."""
        collection = self._client.collections.get(COLLECTION_NAME)
        result = collection.data.delete_many(
            where=Filter.by_property("source").equal(source)
        )
        count = result.successful if result is not None else 0
        logger.info("Supprimé {} chunk(s) pour la source : {}", count, source)
        return count

    # ── lecture ───────────────────────────────────────────────────────────────

    def hybrid_search(
        self,
        query: str,
        query_vector: list[float],
        top_k: int = 20,
        alpha: float = 0.5,
        source: str | None = None,
    ) -> list[dict]:
        """Recherche hybride BM25 + dense (HNSW).

        Parameters
        ----------
        query:
            Texte brut de la question (utilisé par BM25).
        query_vector:
            Vecteur dense de la question (utilisé par HNSW).
        top_k:
            Nombre de résultats à retourner avant reranking (recommandé : 20).
        alpha:
            Pondération : 0.0 = BM25 pur, 1.0 = dense pur, 0.5 = équilibré.
        source:
            Filtre optionnel sur le chemin absolu du document.
        """
        self._ensure_connected()
        collection = self._client.collections.get(COLLECTION_NAME)
        filters = Filter.by_property("source").equal(source) if source else None

        result = collection.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=alpha,
            limit=top_k,
            fusion_type=HybridFusion.RELATIVE_SCORE,
            return_metadata=MetadataQuery(score=True, explain_score=False),
            filters=filters,
        )

        docs = []
        for obj in result.objects:
            docs.append({
                **obj.properties,
                "_score": obj.metadata.score,
                "_id": str(obj.uuid),
            })
        return docs

    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        source: str | None = None,
    ) -> list[dict]:
        """Recherche dense pure (conservée pour compatibilité / tests)."""
        collection = self._client.collections.get(COLLECTION_NAME)
        filters = Filter.by_property("source").equal(source) if source else None

        result = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
            filters=filters,
        )

        docs = []
        for obj in result.objects:
            docs.append({
                **obj.properties,
                "_distance": obj.metadata.distance,
                "_id": str(obj.uuid),
            })
        return docs

    def list_sources(self) -> list[str]:
        """Retourne la liste des sources uniques indexées."""
        self._ensure_connected()
        collection = self._client.collections.get(COLLECTION_NAME)
        # On récupère uniquement la propriété source pour économiser de la mémoire
        result = collection.query.fetch_objects(
            limit=10_000,
            return_properties=["source"],
        )
        sources = sorted(
            set(
                obj.properties.get("source", "")
                for obj in result.objects
                if obj.properties.get("source")
            )
        )
        return sources

    def count(self, source: str | None = None) -> int:
        """Retourne le nombre total de chunks (ou filtré par source)."""
        self._ensure_connected()
        collection = self._client.collections.get(COLLECTION_NAME)
        if source:
            resp = collection.aggregate.over_all(
                total_count=True,
                filters=Filter.by_property("source").equal(source),
            )
        else:
            resp = collection.aggregate.over_all(total_count=True)
        return resp.total_count or 0

    def get_chunk_by_index(self, source: str, chunk_index: int) -> dict | None:
        """Récupère un chunk spécifique par source + chunk_index.

        Utilisé par l'agent pour l'expansion des chunks voisins (prev/next).
        """
        self._ensure_connected()
        collection = self._client.collections.get(COLLECTION_NAME)
        result = collection.query.fetch_objects(
            limit=1,
            filters=(
                Filter.by_property("source").equal(source)
                & Filter.by_property("chunk_index").equal(chunk_index)
            ),
        )
        if result.objects:
            obj = result.objects[0]
            return {**obj.properties, "_id": str(obj.uuid)}
        return None
