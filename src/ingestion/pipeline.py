"""Main ingestion pipeline for combining CSV and PDF data."""

from typing import List, Optional
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from src.utils.config import Config
from src.models.embeddings import get_embedding_model
from src.indexing.vector_store import get_vector_store
from src.ingestion.csv_loader import load_faculty_csv
from src.ingestion.pdf_loader import load_pdfs_from_directory


async def run_ingestion_pipeline(
    csv_path: Optional[str] = None,
    pdf_dir: Optional[str] = None,
    reset_collection: bool = False,
) -> int:
    """
    Run the complete ingestion pipeline: load CSV + PDFs, chunk, embed, and store.

    Args:
        csv_path: Path to faculty CSV file. Defaults to Config.CSV_PATH.
        pdf_dir: Path to PDF directory. Defaults to Config.PDF_DIR.
        reset_collection: If True, delete existing collection and start fresh.

    Returns:
        int: Number of nodes ingested.
    """
    print("=" * 60)
    print("Starting Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Load all documents
    print("\n[Step 1] Loading documents...")
    documents = []

    # Load CSV
    try:
        csv_docs = load_faculty_csv(csv_path)
        documents.extend(csv_docs)
        print(f"✓ Loaded {len(csv_docs)} documents from CSV")
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")

    # Load PDFs
    try:
        pdf_docs = load_pdfs_from_directory(pdf_dir)
        documents.extend(pdf_docs)
        print(f"✓ Loaded {len(pdf_docs)} documents from PDFs")
    except Exception as e:
        print(f"✗ Error loading PDFs: {e}")

    if not documents:
        print("\n⚠ No documents loaded. Please check your data directory.")
        return 0

    print(f"\n📚 Total documents loaded: {len(documents)}")

    # Step 2: Initialize components
    print("\n[Step 2] Initializing components...")
    embed_model = get_embedding_model()
    vector_store = get_vector_store(reset=reset_collection)
    print("✓ Components initialized")

    # Step 3: Build ingestion pipeline
    print("\n[Step 3] Building ingestion pipeline...")
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP,
            ),
            embed_model,
        ],
        vector_store=vector_store,
    )
    print("✓ Pipeline built")

    # Step 4: Run pipeline
    print("\n[Step 4] Running pipeline (this may take a few minutes)...")
    try:
        nodes = await pipeline.arun(documents=documents)
        print(f"✓ Successfully ingested {len(nodes)} nodes")
    except Exception as e:
        print(f"✗ Error during ingestion: {e}")
        raise

    # Step 5: Summary
    print("\n" + "=" * 60)
    print("Ingestion Complete!")
    print("=" * 60)
    print(f"Documents processed: {len(documents)}")
    print(f"Nodes created: {len(nodes)}")
    print(f"Collection: {Config.COLLECTION_NAME}")
    print(f"Storage: {Config.CHROMA_PATH}")
    print("=" * 60)

    return len(nodes)


def run_ingestion_pipeline_sync(
    csv_path: Optional[str] = None,
    pdf_dir: Optional[str] = None,
    reset_collection: bool = False,
) -> int:
    """
    Synchronous wrapper for the ingestion pipeline.

    Args:
        csv_path: Path to faculty CSV file. Defaults to Config.CSV_PATH.
        pdf_dir: Path to PDF directory. Defaults to Config.PDF_DIR.
        reset_collection: If True, delete existing collection and start fresh.

    Returns:
        int: Number of nodes ingested.
    """
    import asyncio
    import nest_asyncio

    # Allow nested event loops (useful in Jupyter notebooks)
    nest_asyncio.apply()

    return asyncio.run(
        run_ingestion_pipeline(csv_path, pdf_dir, reset_collection)
    )


def preview_documents(csv_path: Optional[str] = None, pdf_dir: Optional[str] = None) -> dict:
    """
    Preview what documents would be loaded without actually ingesting them.

    Args:
        csv_path: Path to faculty CSV file. Defaults to Config.CSV_PATH.
        pdf_dir: Path to PDF directory. Defaults to Config.PDF_DIR.

    Returns:
        dict: Preview information about the documents.
    """
    print("Previewing documents...")

    preview = {
        "csv": {"count": 0, "sample": None},
        "pdf": {"count": 0, "types": {}, "sample": None},
    }

    # Preview CSV
    try:
        csv_docs = load_faculty_csv(csv_path)
        preview["csv"]["count"] = len(csv_docs)
        if csv_docs:
            preview["csv"]["sample"] = {
                "text": csv_docs[0].text[:200] + "...",
                "metadata": csv_docs[0].metadata,
            }
    except Exception as e:
        preview["csv"]["error"] = str(e)

    # Preview PDFs
    try:
        pdf_docs = load_pdfs_from_directory(pdf_dir)
        preview["pdf"]["count"] = len(pdf_docs)

        # Count by type
        for doc in pdf_docs:
            doc_type = doc.metadata.get("type", "unknown")
            preview["pdf"]["types"][doc_type] = preview["pdf"]["types"].get(doc_type, 0) + 1

        if pdf_docs:
            preview["pdf"]["sample"] = {
                "text": pdf_docs[0].text[:200] + "...",
                "metadata": pdf_docs[0].metadata,
            }
    except Exception as e:
        preview["pdf"]["error"] = str(e)

    return preview

