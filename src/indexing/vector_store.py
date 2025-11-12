"""ChromaDB vector store setup and management."""

import chromadb
from chromadb import Collection
from llama_index.vector_stores.chroma import ChromaVectorStore
from src.utils.config import Config

# Global ChromaDB client
_chroma_client = None


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Get or create the ChromaDB client singleton.

    Returns:
        chromadb.PersistentClient: The ChromaDB client instance.
    """
    global _chroma_client

    if _chroma_client is None:
        print(f"Initializing ChromaDB client at: {Config.CHROMA_PATH}")
        _chroma_client = chromadb.PersistentClient(path=Config.CHROMA_PATH)
        print("ChromaDB client initialized successfully")

    return _chroma_client


def get_or_create_collection(
    collection_name: str = None,
    reset: bool = False
) -> Collection:
    """
    Get or create a ChromaDB collection.

    Args:
        collection_name: Name of the collection. Defaults to Config.COLLECTION_NAME.
        reset: If True, delete existing collection and create new one.

    Returns:
        Collection: The ChromaDB collection instance.
    """
    client = get_chroma_client()
    name = collection_name or Config.COLLECTION_NAME

    if reset:
        try:
            client.delete_collection(name=name)
            print(f"Deleted existing collection: {name}")
        except Exception as e:
            print(f"No existing collection to delete: {e}")

    collection = client.get_or_create_collection(name=name)
    print(f"Collection '{name}' ready (items: {collection.count()})")

    return collection


def get_vector_store(
    collection_name: str = None,
    reset: bool = False
) -> ChromaVectorStore:
    """
    Get a ChromaVectorStore instance for LlamaIndex integration.

    Args:
        collection_name: Name of the collection. Defaults to Config.COLLECTION_NAME.
        reset: If True, delete existing collection and create new one.

    Returns:
        ChromaVectorStore: The vector store instance.
    """
    collection = get_or_create_collection(collection_name, reset)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return vector_store


def delete_collection(collection_name: str = None):
    """
    Delete a ChromaDB collection.

    Args:
        collection_name: Name of the collection to delete. Defaults to Config.COLLECTION_NAME.
    """
    client = get_chroma_client()
    name = collection_name or Config.COLLECTION_NAME

    try:
        client.delete_collection(name=name)
        print(f"Successfully deleted collection: {name}")
    except Exception as e:
        print(f"Error deleting collection {name}: {e}")


def list_collections():
    """List all collections in the ChromaDB instance."""
    client = get_chroma_client()
    collections = client.list_collections()

    if not collections:
        print("No collections found")
    else:
        print(f"Found {len(collections)} collection(s):")
        for col in collections:
            count = col.count()
            print(f"  - {col.name}: {count} items")

    return collections


def get_collection_stats(collection_name: str = None) -> dict:
    """
    Get statistics about a collection.

    Args:
        collection_name: Name of the collection. Defaults to Config.COLLECTION_NAME.

    Returns:
        dict: Statistics about the collection.
    """
    collection = get_or_create_collection(collection_name)

    return {
        "name": collection.name,
        "count": collection.count(),
        "metadata": collection.metadata,
    }


def reset_chroma_client():
    """Reset the ChromaDB client singleton (useful for testing)."""
    global _chroma_client
    _chroma_client = None

