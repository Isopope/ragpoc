"""Constants and configuration structures for LLM providers."""

from enum import Enum

class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    COHERE = "cohere"
    VOYAGE = "voyage"
    GOOGLE = "google"
    MISTRAL = "mistral"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"

class EmbedTextType(str, Enum):
    QUERY = "query"
    PASSAGE = "passage"

# Dictionary mapping provider names to lists of their supported/available models
PROVIDER_MODELS: dict[str, list[str]] = {
    "openai":  ["gpt-4.1", "gpt-4o", "gpt-4o-mini", "o3-mini"],
    "mistral": ["mistral/mistral-large-latest", "mistral/mistral-small-latest", "mistral/open-mistral-7b"],
    "claude":  ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"],
    "ollama":  ["ollama/gemma4", "ollama/llama3.2", "ollama/mistral", "ollama/phi4"],
}

# Default embedding models per provider
DEFAULT_EMBEDDING_MODELS: dict[str, str] = {
    "openai": "text-embedding-3-small",
    "mistral": "mistral/mistral-embed",
    "claude": "voyage/voyage-2",  # Standard choice for Anthropic ecosystem
    "ollama": "ollama/nomic-embed-text",
}

# Max batch sizes for embeddings (inspired by Onyx)
PROVIDER_BATCH_SIZE: dict[EmbeddingProvider, int] = {
    EmbeddingProvider.OPENAI: 2048,
    EmbeddingProvider.COHERE: 96,
    EmbeddingProvider.VOYAGE: 128,
    EmbeddingProvider.GOOGLE: 250,
    EmbeddingProvider.MISTRAL: 512,
    EmbeddingProvider.OLLAMA: 128,
}

class EmbeddingModelTextType:
    """Mapping of custom text types to provider-specific text types."""
    PROVIDER_TEXT_TYPE_MAP = {
        EmbeddingProvider.COHERE: {
            EmbedTextType.QUERY: "search_query",
            EmbedTextType.PASSAGE: "search_document",
        },
        EmbeddingProvider.VOYAGE: {
            EmbedTextType.QUERY: "query",
            EmbedTextType.PASSAGE: "document",
        },
        EmbeddingProvider.GOOGLE: {
            EmbedTextType.QUERY: "RETRIEVAL_QUERY",
            EmbedTextType.PASSAGE: "RETRIEVAL_DOCUMENT",
        },
    }

    @staticmethod
    def get_type(provider: EmbeddingProvider, text_type: EmbedTextType) -> str:
        """Get provider-specific text type string."""
        if provider not in EmbeddingModelTextType.PROVIDER_TEXT_TYPE_MAP:
            return text_type.value
        return EmbeddingModelTextType.PROVIDER_TEXT_TYPE_MAP[provider].get(text_type, text_type.value)

# Pre-computed list of available providers
PROVIDER_LIST: list[str] = [p.value for p in EmbeddingProvider]
