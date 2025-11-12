"""Index builder for creating and managing VectorStoreIndex."""

from typing import Optional
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from src.models.embeddings import get_embedding_model
from src.models.llm import get_llm
from src.indexing.vector_store import get_vector_store
from src.utils.config import Config


def build_index(collection_name: str = None) -> VectorStoreIndex:
    """
    Build a VectorStoreIndex from an existing ChromaDB collection.

    Args:
        collection_name: Name of the collection. Defaults to Config.COLLECTION_NAME.

    Returns:
        VectorStoreIndex: The index built from the vector store.
    """
    print(f"Building index from collection: {collection_name or Config.COLLECTION_NAME}")

    # Get vector store
    vector_store = get_vector_store(collection_name=collection_name, reset=False)

    # Get embedding model
    embed_model = get_embedding_model()

    # Build index from existing vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    print("✓ Index built successfully")
    return index


def create_query_engine(
    index: Optional[VectorStoreIndex] = None,
    response_mode: str = "compact",
    similarity_top_k: int = None,
    llm=None,
):
    """
    Create a query engine from an index.

    Args:
        index: VectorStoreIndex to use. If None, builds from default collection.
        response_mode: Response synthesis mode ("compact", "tree_summarize", "simple_summarize", etc.)
        similarity_top_k: Number of top results to retrieve. Defaults to Config.TOP_K_RESULTS.
        llm: LLM to use. If None, uses default from config.

    Returns:
        Query engine for semantic search and question answering.
    """
    # Build index if not provided
    if index is None:
        index = build_index()

    # Get LLM if not provided
    if llm is None:
        llm = get_llm()

    # Set default top_k
    top_k = similarity_top_k if similarity_top_k is not None else Config.TOP_K_RESULTS

    print(f"Creating query engine (mode={response_mode}, top_k={top_k})")

    # Create query engine
    query_engine = index.as_query_engine(
        llm=llm,
        response_mode=response_mode,
        similarity_top_k=top_k,
    )

    print("✓ Query engine created")
    return query_engine


def create_retriever(
    index: Optional[VectorStoreIndex] = None,
    similarity_top_k: int = None,
) -> VectorIndexRetriever:
    """
    Create a retriever for semantic search without LLM synthesis.

    Args:
        index: VectorStoreIndex to use. If None, builds from default collection.
        similarity_top_k: Number of top results to retrieve. Defaults to Config.TOP_K_RESULTS.

    Returns:
        VectorIndexRetriever: Retriever for semantic search.
    """
    # Build index if not provided
    if index is None:
        index = build_index()

    # Set default top_k
    top_k = similarity_top_k if similarity_top_k is not None else Config.TOP_K_RESULTS

    print(f"Creating retriever (top_k={top_k})")

    # Create retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )

    print("✓ Retriever created")
    return retriever


def create_custom_query_engine(
    index: Optional[VectorStoreIndex] = None,
    retriever: Optional[VectorIndexRetriever] = None,
    response_mode: str = "compact",
    llm=None,
):
    """
    Create a custom query engine with a specific retriever.

    Args:
        index: VectorStoreIndex to use. If None, builds from default collection.
        retriever: Custom retriever. If None, creates default retriever.
        response_mode: Response synthesis mode.
        llm: LLM to use. If None, uses default from config.

    Returns:
        RetrieverQueryEngine: Custom query engine.
    """
    # Build index if not provided
    if index is None:
        index = build_index()

    # Create retriever if not provided
    if retriever is None:
        retriever = create_retriever(index)

    # Get LLM if not provided
    if llm is None:
        llm = get_llm()

    print(f"Creating custom query engine (mode={response_mode})")

    # Create query engine from retriever
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        response_mode=response_mode,
    )

    print("✓ Custom query engine created")
    return query_engine


class IndexManager:
    """Manager class for handling index operations."""

    def __init__(self, collection_name: str = None):
        """
        Initialize IndexManager.

        Args:
            collection_name: Name of the collection. Defaults to Config.COLLECTION_NAME.
        """
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self._index = None
        self._query_engine = None
        self._retriever = None

    @property
    def index(self) -> VectorStoreIndex:
        """Get or build the index."""
        if self._index is None:
            self._index = build_index(self.collection_name)
        return self._index

    @property
    def query_engine(self):
        """Get or create the default query engine."""
        if self._query_engine is None:
            self._query_engine = create_query_engine(self.index)
        return self._query_engine

    @property
    def retriever(self) -> VectorIndexRetriever:
        """Get or create the default retriever."""
        if self._retriever is None:
            self._retriever = create_retriever(self.index)
        return self._retriever

    def query(self, query_text: str):
        """
        Query the index.

        Args:
            query_text: The query string.

        Returns:
            Query response.
        """
        return self.query_engine.query(query_text)

    def retrieve(self, query_text: str):
        """
        Retrieve relevant nodes without LLM synthesis.

        Args:
            query_text: The query string.

        Returns:
            List of retrieved nodes.
        """
        return self.retriever.retrieve(query_text)

    def reset(self):
        """Reset cached components."""
        self._index = None
        self._query_engine = None
        self._retriever = None

