"""Main ingestion pipeline for combining CSV and PDF data."""

from typing import List, Optional, Tuple
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from src.utils.config import Config
from src.models.embeddings import get_embedding_model
from src.models.llm import get_llm
from src.indexing.vector_store import get_vector_store, get_or_create_collection
from src.ingestion.csv_loader import load_faculty_csv, validate_csv_format
from src.ingestion.pdf_loader import load_pdfs_from_directory
from src.ingestion.enrichment import enrich_csv_documents, get_pdf_nodes_by_faculty
from src.ingestion.website_crawler import crawl_faculty_websites


async def _ingest_documents_to_collection(
    documents: List[Document],
    collection_name: str,
    reset: bool = False
) -> Tuple[int, List]:
    """
    Helper function to ingest documents into a specific collection.

    Args:
        documents: List of documents to ingest.
        collection_name: Name of the collection.
        reset: Whether to reset the collection.

    Returns:
        Tuple of (number of nodes ingested, list of nodes)
    """
    print(f"\n[Ingestion] Processing {len(documents)} documents for '{collection_name}'...")

    # Initialize components
    embed_model = get_embedding_model()
    vector_store = get_vector_store(collection_name=collection_name, reset=reset)

    # Build ingestion pipeline
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP,
            ),
            embed_model,
        ],
    )

    # Process documents through transformations
    print(f"   Chunking and embedding documents...")
    nodes = await pipeline.arun(documents=documents)
    print(f"   ✓ Generated {len(nodes)} nodes from {len(documents)} documents")

    # Add nodes to vector store in batches
    batch_size = Config.INGESTION_BATCH_SIZE
    print(f"   Adding nodes to vector store in batches of {batch_size}...")
    total_batches = (len(nodes) + batch_size - 1) // batch_size

    i = 0
    batch_num = 0
    while i < len(nodes):
        batch = nodes[i:i + batch_size]
        batch_num += 1
        print(f"   Batch {batch_num}/{total_batches} ({len(batch)} nodes)...", end=" ")

        try:
            await vector_store.async_add(batch)
            print("✓")
            i += batch_size
        except Exception as batch_error:
            # If batch is too large, try with smaller batches
            if "batch size" in str(batch_error).lower() or "max batch" in str(batch_error).lower():
                print(f"⚠ Batch too large, retrying with smaller size...")
                new_batch_size = min(batch_size // 2, len(batch))
                if new_batch_size < 1:
                    raise batch_error
                batch_size = new_batch_size
                print(f"   Using reduced batch size: {batch_size}")
                continue
            else:
                raise

    print(f"✓ Successfully ingested {len(nodes)} nodes into '{collection_name}'")

    return len(nodes), nodes


async def run_ingestion_pipeline(
    csv_path: Optional[str] = None,
    pdf_dir: Optional[str] = None,
    reset_collection: bool = False,
    enable_enrichment: bool = True,
    enable_website_crawling: bool = True,
) -> int:
    """
    Run the complete dual-store ingestion pipeline with enrichment.

    This pipeline:
    1. Loads CSV and PDF documents separately
    2. Ingests PDFs into faculty_pdfs collection
    3. Ingests CSV docs into faculty_profiles collection
    4. (Optional) Enriches CSV docs with PDF information and re-ingests
    5. (Optional) Crawls faculty websites and ingests into faculty_websites collection

    Args:
        csv_path: Path to faculty CSV file. Defaults to Config.CSV_PATH.
        pdf_dir: Path to PDF directory. Defaults to Config.PDF_DIR.
        reset_collection: If True, delete existing collections and start fresh.
        enable_enrichment: If True, enrich faculty profiles with PDF data.
        enable_website_crawling: If True, crawl faculty websites and lab websites.

    Returns:
        int: Total number of nodes ingested across all collections.
    """
    print("=" * 80)
    print("Starting Dual-Store Ingestion Pipeline with Enrichment")
    print("=" * 80)

    # Step 1: Load documents
    print("\n[Step 1] Loading documents...")

    csv_docs = []
    pdf_docs = []

    # Validate CSV format before loading
    try:
        print("\n[Validation] Validating CSV format...")
        validation = validate_csv_format(csv_path)

        if not validation.get("valid"):
            print(f"✗ CSV validation failed: {validation.get('error', 'Unknown error')}")
            raise ValueError(f"CSV validation failed: {validation.get('error', 'Unknown error')}")

        print(f"✓ CSV validation passed")
        print(f"  Path: {validation['path']}")
        print(f"  Rows: {validation['rows']}")
        print(f"  Columns: {len(validation['columns'])}")

        if validation.get('missing_recommended_columns'):
            missing = validation['missing_recommended_columns']
            print(f"  ⚠ Warning: Missing recommended columns: {', '.join(missing)}")
            print(f"    Recommended columns: {', '.join(validation['recommended_columns'])}")
        else:
            print(f"  ✓ All recommended columns present")
    except FileNotFoundError:
        # CSV file doesn't exist, will be handled below
        pass

    try:
        csv_docs = load_faculty_csv(csv_path)
        print(f"✓ Loaded {len(csv_docs)} faculty profiles from CSV")
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")

    try:
        pdf_docs = load_pdfs_from_directory(pdf_dir)
        print(f"✓ Loaded {len(pdf_docs)} PDF documents")
    except Exception as e:
        print(f"✗ Error loading PDFs: {e}")

    if not csv_docs and not pdf_docs:
        print("\n⚠ No documents loaded. Please check your data directory.")
        return 0

    total_nodes = 0

    # Step 2: Ingest PDFs into faculty_pdfs collection
    if pdf_docs:
        print("\n[Step 2] Ingesting PDFs into faculty_pdfs collection...")
        pdf_node_count, pdf_nodes = await _ingest_documents_to_collection(
            pdf_docs,
            Config.FACULTY_PDFS_COLLECTION,
            reset=reset_collection
        )
        total_nodes += pdf_node_count
    else:
        print("\n[Step 2] No PDFs to ingest, skipping...")

    # Step 3: Enrich CSV documents with PDF information (if enabled and PDFs exist)
    enriched_csv_docs = csv_docs
    if enable_enrichment and csv_docs and pdf_docs:
        print("\n[Step 3] Enriching faculty profiles with PDF information...")
        try:
            # Get the PDF collection
            pdf_collection = get_or_create_collection(Config.FACULTY_PDFS_COLLECTION)

            # Get list of faculty IDs
            faculty_ids = [doc.metadata.get("faculty_id") for doc in csv_docs if doc.metadata.get("faculty_id")]

            if faculty_ids:
                # Query PDF collection for each faculty_id
                pdf_nodes_by_faculty = get_pdf_nodes_by_faculty(pdf_collection, faculty_ids)

                # Enrich CSV documents
                llm = get_llm()
                enriched_csv_docs = enrich_csv_documents(csv_docs, pdf_nodes_by_faculty, llm)
            else:
                print("  ⚠ No faculty_ids found in CSV, skipping enrichment")
        except Exception as e:
            print(f"  ⚠ Error during enrichment: {e}")
            print(f"  Continuing with non-enriched CSV documents...")
            enriched_csv_docs = csv_docs
    else:
        print("\n[Step 3] Enrichment disabled or no data to enrich, skipping...")

    # Step 4: Ingest (enriched) CSV documents into faculty_profiles collection
    if enriched_csv_docs:
        print("\n[Step 4] Ingesting faculty profiles into faculty_profiles collection...")
        csv_node_count, csv_nodes = await _ingest_documents_to_collection(
            enriched_csv_docs,
            Config.FACULTY_PROFILES_COLLECTION,
            reset=reset_collection
        )
        total_nodes += csv_node_count
    else:
        print("\n[Step 4] No CSV documents to ingest, skipping...")

    # Step 5: Crawl and ingest faculty websites
    website_docs = []
    website_node_count = 0
    if enable_website_crawling and enriched_csv_docs:
        print("\n[Step 5] Crawling faculty websites and lab websites...")
        try:
            website_docs = crawl_faculty_websites(
                enriched_csv_docs,
                max_pages_per_site=20,
                max_depth=3,
                timeout=30,
                rate_limit_delay=1.0
            )

            if website_docs:
                print(f"\n[Step 5] Ingesting {len(website_docs)} website pages into {Config.FACULTY_WEBSITES_COLLECTION} collection...")
                website_node_count, website_nodes = await _ingest_documents_to_collection(
                    website_docs,
                    Config.FACULTY_WEBSITES_COLLECTION,
                    reset=reset_collection
                )
                total_nodes += website_node_count
                print(f"  ✓ Ingested {website_node_count} nodes from websites")
            else:
                print("  ⚠ No website pages crawled")
        except Exception as e:
            print(f"  ⚠ Error during website crawling: {e}")
            print(f"  Continuing without website data...")
    else:
        print("\n[Step 5] Website crawling disabled or no CSV data, skipping...")

    # Step 6: Summary
    print("\n" + "=" * 80)
    print("Ingestion Complete!")
    print("=" * 80)
    print(f"Faculty profiles (CSV): {len(csv_docs)} documents")
    print(f"PDF documents: {len(pdf_docs)} documents")
    print(f"Website pages: {len(website_docs)} documents")
    print(f"Total nodes created: {total_nodes}")
    print(f"Collections:")
    print(f"  - {Config.FACULTY_PROFILES_COLLECTION} (recommendations)")
    print(f"  - {Config.FACULTY_PDFS_COLLECTION} (evidence)")
    if website_node_count > 0:
        print(f"  - {Config.FACULTY_WEBSITES_COLLECTION} (websites)")
    print(f"Storage: {Config.CHROMA_PATH}")
    print("=" * 80)

    return total_nodes


def run_ingestion_pipeline_sync(
    csv_path: Optional[str] = None,
    pdf_dir: Optional[str] = None,
    reset_collection: bool = False,
    enable_enrichment: bool = True,
    enable_website_crawling: bool = True,
) -> int:
    """
    Synchronous wrapper for the ingestion pipeline.

    Args:
        csv_path: Path to faculty CSV file. Defaults to Config.CSV_PATH.
        pdf_dir: Path to PDF directory. Defaults to Config.PDF_DIR.
        reset_collection: If True, delete existing collections and start fresh.
        enable_enrichment: If True, enrich faculty profiles with PDF data.
        enable_website_crawling: If True, crawl faculty websites and lab websites.

    Returns:
        int: Total number of nodes ingested.
    """
    import asyncio
    import nest_asyncio

    # Allow nested event loops (useful in Jupyter notebooks)
    nest_asyncio.apply()

    return asyncio.run(
        run_ingestion_pipeline(csv_path, pdf_dir, reset_collection, enable_enrichment, enable_website_crawling)
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
        # Validate CSV format first
        validation = validate_csv_format(csv_path)
        preview["csv"]["validation"] = validation

        if validation.get("valid"):
            csv_docs = load_faculty_csv(csv_path)
            preview["csv"]["count"] = len(csv_docs)
            if csv_docs:
                preview["csv"]["sample"] = {
                    "text": csv_docs[0].text[:200] + "...",
                    "metadata": csv_docs[0].metadata,
                }
        else:
            preview["csv"]["error"] = validation.get("error", "Validation failed")
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

