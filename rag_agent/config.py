"""Configuration centralisée du package rag_agent.

Remplace tous les os.getenv() éparpillés dans rag_pipeline.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv


@dataclass
class RAGConfig:
    """Configuration complète du pipeline RAG unifié."""

    # ── LLM ───────────────────────────────────────────────────────────────────
    openai_key: str = ""
    llm_model: str = "gpt-4.1"
    embedding_model: str = "text-embedding-3-small"
    max_tokens: int = 4000
    llm_timeout: float = 30.0

    # ── Reranking ─────────────────────────────────────────────────────────────
    cohere_key: Optional[str] = None

    # ── Récupération ──────────────────────────────────────────────────────────
    top_k_retrieve: int = 20
    top_k_final: int = 5
    hybrid_alpha: float = 0.5

    # ── Boucle ReAct ──────────────────────────────────────────────────────────
    max_agent_iter: int = 60
    # Seuil (en tokens estimés) déclenchant la compression du contexte
    token_threshold: int = 12_000

    # ── Arbre de décision ─────────────────────────────────────────────────────
    max_tree_depth: int = 5
    tree_mode: str = "rag"  # "rag" | "multibranch" | "onebranch"

    # ── Feature flags ─────────────────────────────────────────────────────────
    use_cohere_rerank: bool = True
    enable_compression: bool = False
    use_follow_up: bool = True
    use_title_generation: bool = True
    debug: bool = False

    # ── Weaviate (pour les outils) ────────────────────────────────────────────
    weaviate_host: str = "localhost"
    weaviate_port: int = 8080

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Crée une RAGConfig en lisant les variables d'environnement."""
        load_dotenv()
        return cls(
            openai_key=os.getenv("OPENAI_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", "gpt-4.1"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            max_tokens=int(os.getenv("MAX_TOKENS", "4000")),
            llm_timeout=float(os.getenv("LLM_TIMEOUT", "30.0")),
            cohere_key=os.getenv("COHERE_API_KEY") or None,
            top_k_retrieve=int(os.getenv("TOP_K_RETRIEVE", "20")),
            top_k_final=int(os.getenv("TOP_K_FINAL", "10")),
            hybrid_alpha=float(os.getenv("HYBRID_ALPHA", "0.5")),
            max_agent_iter=int(os.getenv("MAX_AGENT_ITER", "60")),
            token_threshold=int(os.getenv("TOKEN_THRESHOLD", "12000")),
            max_tree_depth=int(os.getenv("MAX_TREE_DEPTH", "5")),
            tree_mode=os.getenv("LANGGRAPH_MODE", "rag"),
            use_cohere_rerank=os.getenv("USE_COHERE_RERANK", "true").lower() == "true",
            enable_compression=os.getenv("ENABLE_COMPRESSION", "false").lower() == "true",
            use_follow_up=os.getenv("USE_FOLLOW_UP", "true").lower() == "true",
            use_title_generation=os.getenv("USE_TITLE_GENERATION", "true").lower() == "true",
            debug=os.getenv("DEBUG", "false").lower() == "true",
            weaviate_host=os.getenv("WEAVIATE_HOST", "localhost"),
            weaviate_port=int(os.getenv("WEAVIATE_PORT", "8080")),
        )

    def validate(self) -> None:
        """Lève ValueError si la configuration est invalide."""
        if not self.openai_key:
            raise ValueError(
                "OPENAI_API_KEY est requis. "
                "Définissez la variable d'environnement ou passez openai_key au constructeur."
            )
        if not 0.0 <= self.hybrid_alpha <= 1.0:
            raise ValueError(f"hybrid_alpha doit être entre 0 et 1, reçu : {self.hybrid_alpha}")
        if self.max_agent_iter < 1:
            raise ValueError(f"max_agent_iter doit être ≥ 1, reçu : {self.max_agent_iter}")

    def to_dict(self) -> dict:
        """Exporte les paramètres publics (sans les clés API)."""
        return {
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "max_tokens": self.max_tokens,
            "llm_timeout": self.llm_timeout,
            "top_k_retrieve": self.top_k_retrieve,
            "hybrid_alpha": self.hybrid_alpha,
            "max_agent_iter": self.max_agent_iter,
            "token_threshold": self.token_threshold,
            "tree_mode": self.tree_mode,
            "use_cohere_rerank": self.use_cohere_rerank,
            "enable_compression": self.enable_compression,
        }
