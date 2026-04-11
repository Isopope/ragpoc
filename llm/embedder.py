"""
Centralized embedder functions and classes via LiteLLM.
Inspired by Onyx's modular NLP architecture.
"""
import threading
import time
from typing import Any, Callable, Optional, Union

from loguru import logger

from .constants import (
    DEFAULT_EMBEDDING_MODELS,
    EmbeddingModelTextType,
    EmbeddingProvider,
    EmbedTextType,
    PROVIDER_BATCH_SIZE,
)

def _batch_list(items: list[Any], batch_size: int):
    """Slices a list into batches of a given size."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]

class EmbeddingModel:
    """
    Robust embedding model interface.
    Handles batching, timeouts, and provider-specific text types.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = timeout
        self.provider = self._detect_provider(model)

    def _detect_provider(self, model: str) -> EmbeddingProvider:
        """Detect provider from model string prefix."""
        model_lower = model.lower()
        if model_lower.startswith("openai/"): return EmbeddingProvider.OPENAI
        if model_lower.startswith("cohere/"): return EmbeddingProvider.COHERE
        if model_lower.startswith("voyage/"): return EmbeddingProvider.VOYAGE
        if model_lower.startswith("google/") or model_lower.startswith("vertex/"): return EmbeddingProvider.GOOGLE
        if model_lower.startswith("mistral/"): return EmbeddingProvider.MISTRAL
        if model_lower.startswith("ollama/"): return EmbeddingProvider.OLLAMA
        
        # Fallback based on model name substrings
        if "gpt-3.5" in model_lower or "text-embedding" in model_lower: return EmbeddingProvider.OPENAI
        return EmbeddingProvider.OPENAI # Default fallback

    def embed_text(self, text: str, text_type: EmbedTextType = EmbedTextType.PASSAGE) -> list[float]:
        """Embeds a single string."""
        vectors = self.embed_batch([text], text_type=text_type)
        return vectors[0]

    def embed_batch(
        self, 
        texts: list[str], 
        text_type: EmbedTextType = EmbedTextType.PASSAGE
    ) -> list[list[float]]:
        """
        Embeds a list of strings efficiently by handling provider batch limits.
        """
        if not texts:
            return []

        try:
            import litellm
        except ImportError:
            raise ImportError("litellm must be installed to use EmbeddingModel.")

        batch_size = PROVIDER_BATCH_SIZE.get(self.provider, 128)
        text_type_str = EmbeddingModelTextType.get_type(self.provider, text_type)
        
        all_vectors: list[list[float]] = []
        
        for batch in _batch_list(texts, batch_size):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "input": [t or " " for t in batch],
                "api_key": self.api_key,
                "api_base": self.api_base,
            }
            
            # Application des paramètres de type de texte spécifiques
            if self.provider == EmbeddingProvider.GOOGLE:
                kwargs["task_type"] = text_type_str
            elif self.provider == EmbeddingProvider.COHERE:
                kwargs["input_type"] = text_type_str
            elif self.provider == EmbeddingProvider.VOYAGE:
                kwargs["input_type"] = text_type_str

            result: dict = {"vectors": None, "error": None}
            start_time = time.monotonic()

            def _run():
                try:
                    resp = litellm.embedding(**kwargs)
                    result["vectors"] = [item["embedding"] for item in resp.data]
                except Exception as exc:
                    result["error"] = exc

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(timeout=self.timeout)

            if t.is_alive():
                logger.error(f"Embedding timeout for model {self.model} after {self.timeout}s")
                raise TimeoutError(f"Embedding timeout après {self.timeout}s")
                
            if result["error"]:
                logger.error(f"Embedding error with {self.model}: {result['error']}")
                raise result["error"]
                
            all_vectors.extend(result["vectors"])
            
            elapsed = time.monotonic() - start_time
            logger.debug(f"Embedded {len(batch)} items with {self.model} in {elapsed:.2f}s")
        
        return all_vectors

def make_embedder(client: Optional[Any], model: str, timeout: float = 60.0) -> Callable[[str], list[float]]:
    """
    Backward compatible factory function.
    Returns a function that embeds a single string as a PASSAGE.
    """
    api_key = getattr(client, "api_key", None) if client else None
    api_base = getattr(client, "base_url", None) if client else None
    
    model_instance = EmbeddingModel(
        model=model,
        api_key=api_key,
        api_base=api_base,
        timeout=timeout
    )
    
    def _embed(text: str) -> list[float]:
        return model_instance.embed_text(text)
        
    return _embed
