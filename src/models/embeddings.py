"""Embedding model setup and management."""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.utils.config import Config

# Global embedding model instance
_embed_model = None


def get_embedding_model() -> HuggingFaceEmbedding:
    """
    Get or create the embedding model singleton.

    Returns:
        HuggingFaceEmbedding: The embedding model instance.
    """
    global _embed_model

    if _embed_model is None:
        print(f"Initializing embedding model: {Config.EMBEDDING_MODEL}")
        _embed_model = HuggingFaceEmbedding(
            model_name=Config.EMBEDDING_MODEL,
            # Cache directory will be set to default HuggingFace cache
        )
        print("Embedding model initialized successfully")

    return _embed_model


def reset_embedding_model():
    """Reset the embedding model singleton (useful for testing)."""
    global _embed_model
    _embed_model = None

