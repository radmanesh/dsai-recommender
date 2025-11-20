"""Main ingestion pipeline for combining CSV and PDF data."""

from typing import List, Optional, Tuple
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from src.utils.config import Config
from src.utils.logger import get_logger, debug, info, warning, error, verbose
from src.models.embeddings import get_embedding_model
from src.models.llm import get_llm
from src.indexing.vector_store import get_vector_store, get_or_create_collection
from src.ingestion.csv_loader import load_faculty_csv, validate_csv_format
from src.ingestion.pdf_loader import load_pdfs_from_directory
from src.ingestion.enrichment import enrich_csv_documents, get_pdf_nodes_by_faculty
from src.ingestion.website_crawler import crawl_faculty_websites

logger = get_logger(__name__)


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
    info(f"Processing {len(documents)} documents for collection '{collection_name}'")
    verbose(f"Collection reset flag: {reset}")
    verbose(f"Document metadata preview: {[doc.metadata.get('type', 'unknown') for doc in documents[:5]]}")

    # Initialize components
    debug("Initializing embedding model...")
    embed_model = get_embedding_model()
    debug(f"Embedding model: {Config.EMBEDDING_MODEL}")

    debug(f"Getting vector store for collection: {collection_name}")
    vector_store = get_vector_store(collection_name=collection_name, reset=reset)
    verbose(f"Vector store initialized for collection: {collection_name}")

    # Build ingestion pipeline
    debug(f"Building ingestion pipeline with chunk_size={Config.CHUNK_SIZE}, chunk_overlap={Config.CHUNK_OVERLAP}")
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
    info("Chunking and embedding documents...")
    verbose(f"Starting transformation pipeline for {len(documents)} documents")
    nodes = await pipeline.arun(documents=documents)
    info(f"Generated {len(nodes)} nodes from {len(documents)} documents (avg {len(nodes)/len(documents):.1f} nodes/doc)")

    debug(f"Node count breakdown: {len(documents)} docs → {len(nodes)} nodes")
    verbose(f"Sample node IDs: {[node.node_id[:16] + '...' for node in nodes[:3]]}")

    # Add nodes to vector store in batches
    batch_size = Config.INGESTION_BATCH_SIZE
    info(f"Adding nodes to vector store in batches of {batch_size}...")
    total_batches = (len(nodes) + batch_size - 1) // batch_size
    verbose(f"Total batches to process: {total_batches}")

    i = 0
    batch_num = 0
    while i < len(nodes):
        batch = nodes[i:i + batch_size]
        batch_num += 1
        verbose(f"Processing batch {batch_num}/{total_batches}: {len(batch)} nodes (indices {i}-{min(i+batch_size, len(nodes))-1})")

        try:
            await vector_store.async_add(batch)
            debug(f"Batch {batch_num}/{total_batches} added successfully ({len(batch)} nodes)")
            i += batch_size
        except Exception as batch_error:
            # If batch is too large, try with smaller batches
            if "batch size" in str(batch_error).lower() or "max batch" in str(batch_error).lower():
                warning(f"Batch {batch_num} too large ({len(batch)} nodes), retrying with smaller size...")
                new_batch_size = min(batch_size // 2, len(batch))
                if new_batch_size < 1:
                    error(f"Cannot reduce batch size further. Error: {batch_error}")
                    raise batch_error
                batch_size = new_batch_size
                debug(f"Reduced batch size to: {batch_size}")
                verbose(f"Retrying batch {batch_num} with size {batch_size}")
                continue
            else:
                error(f"Error adding batch {batch_num}: {batch_error}")
                raise

    info(f"Successfully ingested {len(nodes)} nodes into '{collection_name}'")
    verbose(f"Ingestion complete for collection '{collection_name}': {len(nodes)} total nodes")

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
    info("=" * 80)
    info("Starting Dual-Store Ingestion Pipeline with Enrichment")
    info("=" * 80)
    debug(f"Pipeline configuration:")
    debug(f"  CSV path: {csv_path or Config.CSV_PATH}")
    debug(f"  PDF directory: {pdf_dir or Config.PDF_DIR}")
    debug(f"  Reset collection: {reset_collection}")
    debug(f"  Enable enrichment: {enable_enrichment}")
    debug(f"  Enable website crawling: {enable_website_crawling}")

    # Step 1: Load documents
    info("\n[Step 1] Loading documents...")

    csv_docs = []
    pdf_docs = []

    # Validate CSV format before loading
    try:
        info("[Validation] Validating CSV format...")
        validation = validate_csv_format(csv_path)
        verbose(f"CSV validation result: {validation}")

        if not validation.get("valid"):
            error(f"CSV validation failed: {validation.get('error', 'Unknown error')}")
            raise ValueError(f"CSV validation failed: {validation.get('error', 'Unknown error')}")

        info(f"CSV validation passed")
        debug(f"  Path: {validation['path']}")
        debug(f"  Rows: {validation['rows']}")
        debug(f"  Columns: {len(validation['columns'])}")
        verbose(f"  Column names: {validation['columns']}")

        if validation.get('missing_recommended_columns'):
            missing = validation['missing_recommended_columns']
            warning(f"Missing recommended columns: {', '.join(missing)}")
            debug(f"  Recommended columns: {', '.join(validation['recommended_columns'])}")
        else:
            debug("All recommended columns present")
    except FileNotFoundError as e:
        debug(f"CSV file not found: {e}, will be handled below")
        # CSV file doesn't exist, will be handled below
        pass

    try:
        debug("Loading faculty CSV...")
        csv_docs = load_faculty_csv(csv_path)
        info(f"Loaded {len(csv_docs)} faculty profiles from CSV")
        verbose(f"CSV documents metadata: {[doc.metadata.get('faculty_name', 'Unknown') for doc in csv_docs[:5]]}")
        if csv_docs:
            verbose(f"Sample CSV document keys: {list(csv_docs[0].metadata.keys())}")
    except Exception as e:
        error(f"Error loading CSV: {e}")
        verbose(f"CSV loading exception details: {type(e).__name__}: {str(e)}")

    try:
        debug(f"Loading PDFs from directory: {pdf_dir or Config.PDF_DIR}")
        pdf_docs = load_pdfs_from_directory(pdf_dir)
        info(f"Loaded {len(pdf_docs)} PDF documents")
        if pdf_docs:
            verbose(f"PDF types distribution: {[doc.metadata.get('type', 'unknown') for doc in pdf_docs]}")
            verbose(f"Sample PDF metadata: {pdf_docs[0].metadata if pdf_docs else 'None'}")
    except Exception as e:
        error(f"Error loading PDFs: {e}")
        verbose(f"PDF loading exception details: {type(e).__name__}: {str(e)}")

    if not csv_docs and not pdf_docs:
        warning("No documents loaded. Please check your data directory.")
        return 0

    total_nodes = 0
    pdf_node_count = 0
    csv_node_count = 0
    debug(f"Starting ingestion: {len(csv_docs)} CSV docs, {len(pdf_docs)} PDF docs")

    # Step 2: Ingest PDFs into faculty_pdfs collection
    if pdf_docs:
        info("\n[Step 2] Ingesting PDFs into faculty_pdfs collection...")
        debug(f"Collection: {Config.FACULTY_PDFS_COLLECTION}")
        pdf_node_count, pdf_nodes = await _ingest_documents_to_collection(
            pdf_docs,
            Config.FACULTY_PDFS_COLLECTION,
            reset=reset_collection
        )
        total_nodes += pdf_node_count
        verbose(f"PDF ingestion complete: {pdf_node_count} nodes from {len(pdf_docs)} documents")
    else:
        info("\n[Step 2] No PDFs to ingest, skipping...")
        debug("PDF directory empty or no PDF files found")

    # Step 3: Enrich CSV documents with PDF information (if enabled and PDFs exist)
    enriched_csv_docs = csv_docs
    if enable_enrichment and csv_docs and pdf_docs:
        info("\n[Step 3] Enriching faculty profiles with PDF information...")
        try:
            debug("Getting PDF collection for enrichment...")
            pdf_collection = get_or_create_collection(Config.FACULTY_PDFS_COLLECTION)
            verbose(f"PDF collection ready: {pdf_collection.count()} items")

            # Get list of faculty IDs
            faculty_ids = [doc.metadata.get("faculty_id") for doc in csv_docs if doc.metadata.get("faculty_id")]
            debug(f"Found {len(faculty_ids)} faculty IDs for enrichment")
            verbose(f"Faculty IDs: {faculty_ids[:10]}..." if len(faculty_ids) > 10 else f"Faculty IDs: {faculty_ids}")

            if faculty_ids:
                debug("Querying PDF collection for faculty documents...")
                pdf_nodes_by_faculty = get_pdf_nodes_by_faculty(pdf_collection, faculty_ids)
                verbose(f"PDF nodes by faculty: {[(fid, len(nodes)) for fid, nodes in list(pdf_nodes_by_faculty.items())[:5]]}")

                debug("Initializing LLM for enrichment...")
                llm = get_llm()
                debug(f"Enriching {len(csv_docs)} CSV documents with PDF data...")
                enriched_csv_docs = enrich_csv_documents(csv_docs, pdf_nodes_by_faculty, llm)
                info(f"Enriched {len(enriched_csv_docs)} faculty profiles with PDF information")
                verbose(f"Enrichment metadata added: {[k for k in enriched_csv_docs[0].metadata.keys() if k.startswith('pdf_')][:5]}")
            else:
                warning("No faculty_ids found in CSV, skipping enrichment")
        except Exception as e:
            error(f"Error during enrichment: {e}")
            warning("Continuing with non-enriched CSV documents...")
            verbose(f"Enrichment exception: {type(e).__name__}: {str(e)}")
            enriched_csv_docs = csv_docs
    else:
        debug(f"\n[Step 3] Enrichment disabled ({enable_enrichment}) or no data to enrich (CSV: {len(csv_docs)}, PDF: {len(pdf_docs)})")
        info("Enrichment disabled or no data to enrich, skipping...")

    # Step 4: Ingest (enriched) CSV documents into faculty_profiles collection
    if enriched_csv_docs:
        info("\n[Step 4] Ingesting faculty profiles into faculty_profiles collection...")
        debug(f"Collection: {Config.FACULTY_PROFILES_COLLECTION}")
        debug(f"Documents to ingest: {len(enriched_csv_docs)} (enriched: {len(enriched_csv_docs) != len(csv_docs)})")
        csv_node_count, csv_nodes = await _ingest_documents_to_collection(
            enriched_csv_docs,
            Config.FACULTY_PROFILES_COLLECTION,
            reset=reset_collection
        )
        total_nodes += csv_node_count
        verbose(f"CSV ingestion complete: {csv_node_count} nodes from {len(enriched_csv_docs)} documents")
    else:
        warning("\n[Step 4] No CSV documents to ingest, skipping...")
        debug("No enriched CSV documents available")

    # Step 5: Crawl and ingest faculty websites
    website_docs = []
    website_node_count = 0
    if enable_website_crawling and enriched_csv_docs:
        info("\n[Step 5] Crawling faculty websites and lab websites...")
        debug("Website crawling parameters: max_pages=20, max_depth=3, timeout=30s")
        try:
            website_docs = crawl_faculty_websites(
                enriched_csv_docs,
                max_pages_per_site=20,
                max_depth=3,
                timeout=30,
                rate_limit_delay=1.0
            )

            if website_docs:
                info(f"\n[Step 5] Ingesting {len(website_docs)} website pages into {Config.FACULTY_WEBSITES_COLLECTION} collection...")
                debug(f"Collection: {Config.FACULTY_WEBSITES_COLLECTION}")
                verbose(f"Website pages by source: {[doc.metadata.get('source_type', 'unknown') for doc in website_docs[:10]]}")
                website_node_count, website_nodes = await _ingest_documents_to_collection(
                    website_docs,
                    Config.FACULTY_WEBSITES_COLLECTION,
                    reset=reset_collection
                )
                total_nodes += website_node_count
                info(f"Ingested {website_node_count} nodes from websites")
                verbose(f"Website ingestion complete: {website_node_count} nodes from {len(website_docs)} pages")
            else:
                warning("No website pages crawled")
                debug("No websites found in CSV or all websites failed to crawl")
        except Exception as e:
            error(f"Error during website crawling: {e}")
            warning("Continuing without website data...")
            verbose(f"Website crawling exception: {type(e).__name__}: {str(e)}")
    else:
        debug(f"\n[Step 5] Website crawling disabled ({enable_website_crawling}) or no CSV data ({len(enriched_csv_docs)})")
        info("Website crawling disabled or no CSV data, skipping...")

    # Step 6: Summary
    info("\n" + "=" * 80)
    info("Ingestion Complete!")
    info("=" * 80)
    info(f"Faculty profiles (CSV): {len(csv_docs)} documents")
    info(f"PDF documents: {len(pdf_docs)} documents")
    info(f"Website pages: {len(website_docs)} documents")
    info(f"Total nodes created: {total_nodes}")
    debug(f"Node breakdown: CSV={csv_node_count}, PDF={pdf_node_count}, Websites={website_node_count}")
    info(f"Collections:")
    info(f"  - {Config.FACULTY_PROFILES_COLLECTION} (recommendations)")
    info(f"  - {Config.FACULTY_PDFS_COLLECTION} (evidence)")
    if website_node_count > 0:
        info(f"  - {Config.FACULTY_WEBSITES_COLLECTION} (websites)")
    debug(f"Storage path: {Config.CHROMA_PATH}")
    verbose(f"Full storage path: {Config.CHROMA_PATH}")
    info("=" * 80)

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

