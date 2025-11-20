"""Enrichment module for extracting PDF information and enriching faculty profile documents."""

from typing import List, Dict, Optional
from llama_index.core import Document
from llama_index.core.schema import NodeWithScore
from src.models.llm import get_llm
from src.utils.logger import get_logger, debug, info, warning, error, verbose

logger = get_logger(__name__)


def enrich_faculty_profile(
    faculty_id: str,
    pdf_nodes: List[NodeWithScore],
    llm=None
) -> Dict[str, str]:
    """
    Extract summary, research_interests, and publications from PDF nodes for a faculty member.

    Args:
        faculty_id: The faculty_id to enrich.
        pdf_nodes: List of PDF nodes (with scores) for this faculty member.
        llm: LLM instance. If None, uses default from config.

    Returns:
        Dict with keys: summary, research_interests, publications
    """
    debug(f"Enriching faculty profile: {faculty_id} with {len(pdf_nodes)} PDF nodes")
    verbose(f"Faculty ID: {faculty_id}, PDF nodes count: {len(pdf_nodes)}")

    if not pdf_nodes:
        debug(f"No PDF nodes found for faculty {faculty_id}")
        return {
            "summary": None,
            "research_interests": None,
            "publications": None
        }

    if llm is None:
        debug("Initializing LLM for enrichment...")
        llm = get_llm()

    # Combine text from all PDF nodes (limit to avoid token limits)
    combined_text = ""
    max_chars = 8000  # Leave room for prompt
    debug(f"Combining text from {len(pdf_nodes)} PDF nodes (max {max_chars} chars)...")

    for i, node in enumerate(pdf_nodes):
        node_text = node.node.text if hasattr(node, 'node') else node.text
        verbose(f"Processing node {i+1}/{len(pdf_nodes)}: {len(node_text)} chars")
        if len(combined_text) + len(node_text) < max_chars:
            combined_text += node_text + "\n\n"
        else:
            debug(f"Text limit reached at node {i+1}/{len(pdf_nodes)}, truncating...")
            break

    debug(f"Combined text length: {len(combined_text)} characters")
    if not combined_text.strip():
        warning(f"No text extracted from PDF nodes for faculty {faculty_id}")
        return {
            "summary": None,
            "research_interests": None,
            "publications": None
        }

    # Create prompt for extraction
    debug(f"Creating extraction prompt for faculty {faculty_id}...")
    prompt = f"""You are an expert at analyzing academic documents (CVs, papers, proposals). Analyze the following documents for faculty member with ID '{faculty_id}' and extract key information.

DOCUMENTS:
{combined_text}

Please extract the following information:

1. SUMMARY: Write a concise 2-3 sentence summary highlighting the faculty member's key expertise and contributions.

2. RESEARCH_INTERESTS: List the main research interests, topics, and areas (comma-separated). Focus on technical topics and methodologies.

3. PUBLICATIONS: List 3-5 notable publications or projects mentioned (if any), with brief descriptions. If no publications are explicitly mentioned, write "N/A".

Format your response EXACTLY as follows:

SUMMARY: [your summary here]
RESEARCH_INTERESTS: [interest1, interest2, interest3, ...]
PUBLICATIONS: [publication1; publication2; publication3; ... or N/A]

Be specific and use technical terminology where appropriate."""

    verbose(f"Prompt length: {len(prompt)} characters")
    debug(f"Querying LLM for enrichment extraction (faculty: {faculty_id})...")

    try:
        response = llm.complete(prompt)
        response_text = str(response).strip()
        debug(f"LLM response received: {len(response_text)} characters")
        verbose(f"LLM response preview: {response_text[:300]}...")

        # Parse response
        debug("Parsing LLM response...")
        result = {
            "summary": None,
            "research_interests": None,
            "publications": None
        }

        lines = response_text.split("\n")
        current_field = None
        current_text = []

        for line in lines:
            line = line.strip()

            if line.startswith("SUMMARY:"):
                if current_field and current_text:
                    result[current_field] = " ".join(current_text)
                current_field = "summary"
                content = line.replace("SUMMARY:", "").strip()
                current_text = [content] if content else []
            elif line.startswith("RESEARCH_INTERESTS:"):
                if current_field and current_text:
                    result[current_field] = " ".join(current_text)
                current_field = "research_interests"
                content = line.replace("RESEARCH_INTERESTS:", "").strip()
                current_text = [content] if content else []
            elif line.startswith("PUBLICATIONS:"):
                if current_field and current_text:
                    result[current_field] = " ".join(current_text)
                current_field = "publications"
                content = line.replace("PUBLICATIONS:", "").strip()
                current_text = [content] if content else []
            elif line and current_field:
                current_text.append(line)

        # Save the last field
        if current_field and current_text:
            result[current_field] = " ".join(current_text)

        # Clean up "N/A" entries
        for key in result:
            if result[key] and result[key].upper() in ["N/A", "N/A.", "NONE", "NONE."]:
                result[key] = None

        debug(f"Enrichment extraction complete for {faculty_id}")
        verbose(f"Extracted: summary={bool(result['summary'])}, research_interests={bool(result['research_interests'])}, publications={bool(result['publications'])}")
        if result['summary']:
            verbose(f"Summary preview: {result['summary'][:100]}...")
        if result['research_interests']:
            verbose(f"Research interests: {result['research_interests'][:100]}...")

        return result

    except Exception as e:
        error(f"Error extracting enrichment data for {faculty_id}: {e}")
        verbose(f"Enrichment exception: {type(e).__name__}: {str(e)}")
        return {
            "summary": None,
            "research_interests": None,
            "publications": None
        }


def enrich_csv_documents(
    csv_docs: List[Document],
    pdf_nodes_by_faculty: Dict[str, List[NodeWithScore]],
    llm=None
) -> List[Document]:
    """
    Enrich CSV documents with information extracted from PDF nodes.

    Args:
        csv_docs: List of CSV Document objects to enrich.
        pdf_nodes_by_faculty: Dict mapping faculty_id to list of PDF nodes.
        llm: LLM instance. If None, uses default from config.

    Returns:
        List[Document]: Enriched CSV documents.
    """
    if llm is None:
        llm = get_llm()

    enriched_docs = []

    info(f"Processing {len(csv_docs)} faculty profiles for enrichment...")
    debug(f"PDF nodes available for {len(pdf_nodes_by_faculty)} faculty members")
    verbose(f"Faculty IDs with PDFs: {list(pdf_nodes_by_faculty.keys())[:10]}...")

    for i, doc in enumerate(csv_docs, 1):
        faculty_id = doc.metadata.get("faculty_id")
        faculty_name = doc.metadata.get("faculty_name", "Unknown")
        debug(f"Processing profile {i}/{len(csv_docs)}: {faculty_name} ({faculty_id})")

        if not faculty_id:
            warning(f"Skipping {faculty_name}: no faculty_id")
            verbose(f"Document metadata keys: {list(doc.metadata.keys())}")
            enriched_docs.append(doc)
            continue

        # Get PDF nodes for this faculty
        pdf_nodes = pdf_nodes_by_faculty.get(faculty_id, [])
        verbose(f"PDF nodes for {faculty_id}: {len(pdf_nodes)} nodes")

        if not pdf_nodes:
            debug(f"{faculty_name} ({faculty_id}): no PDFs found")
            enriched_docs.append(doc)
            continue

        info(f"Enriching {faculty_name} ({faculty_id}) with {len(pdf_nodes)} PDF nodes...")
        verbose(f"PDF nodes scores: {[getattr(n, 'score', 'N/A') for n in pdf_nodes[:3]]}")

        # Extract enrichment data
        debug(f"Extracting enrichment data for {faculty_name}...")
        enrichment = enrich_faculty_profile(faculty_id, pdf_nodes, llm)

        # Build enriched text
        enriched_text_parts = [doc.text]  # Start with original CSV text
        original_text_length = len(doc.text)
        debug(f"Original document text length: {original_text_length} characters")

        if enrichment["summary"] or enrichment["research_interests"] or enrichment["publications"]:
            enriched_text_parts.append("\n--- Additional Information from CVs/Publications ---")
            debug(f"Adding enrichment fields: summary={bool(enrichment['summary'])}, research_interests={bool(enrichment['research_interests'])}, publications={bool(enrichment['publications'])}")

            if enrichment["summary"]:
                enriched_text_parts.append(f"\nSummary: {enrichment['summary']}")
                # Also add to metadata
                doc.metadata["pdf_summary"] = enrichment["summary"]
                verbose(f"Added pdf_summary: {enrichment['summary'][:100]}...")

            if enrichment["research_interests"]:
                enriched_text_parts.append(f"\nAdditional Research Interests: {enrichment['research_interests']}")
                doc.metadata["pdf_research_interests"] = enrichment["research_interests"]
                verbose(f"Added pdf_research_interests: {enrichment['research_interests'][:100]}...")

            if enrichment["publications"]:
                enriched_text_parts.append(f"\nNotable Publications: {enrichment['publications']}")
                doc.metadata["pdf_publications"] = enrichment["publications"]
                verbose(f"Added pdf_publications: {enrichment['publications'][:100]}...")
        else:
            debug(f"No enrichment data extracted for {faculty_name}")

        # Create enriched document
        enriched_text = "\n".join(enriched_text_parts)
        enriched_text_length = len(enriched_text)
        debug(f"Enriched document text length: {enriched_text_length} characters (added {enriched_text_length - original_text_length} chars)")

        enriched_doc = Document(
            text=enriched_text,
            metadata=doc.metadata.copy()
        )
        verbose(f"Enriched document metadata keys: {list(enriched_doc.metadata.keys())}")

        enriched_docs.append(enriched_doc)

    info(f"Enrichment complete: {len(enriched_docs)} profiles processed")
    enriched_count = sum(1 for doc in enriched_docs if any(k.startswith('pdf_') for k in doc.metadata.keys()))
    debug(f"Profiles successfully enriched: {enriched_count}/{len(enriched_docs)}")

    return enriched_docs


def get_pdf_nodes_by_faculty(
    pdf_collection,
    faculty_ids: List[str],
    top_k_per_faculty: int = 10
) -> Dict[str, List]:
    """
    Query PDF collection to get nodes for each faculty_id.

    Args:
        pdf_collection: ChromaDB collection containing PDF documents.
        faculty_ids: List of faculty IDs to query.
        top_k_per_faculty: Number of top nodes to retrieve per faculty.

    Returns:
        Dict mapping faculty_id to list of nodes.
    """
    from llama_index.core import VectorStoreIndex
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
    from src.models.embeddings import get_embedding_model

    # Create vector store and index for PDFs
    pdf_vector_store = ChromaVectorStore(chroma_collection=pdf_collection)
    embed_model = get_embedding_model()

    pdf_index = VectorStoreIndex.from_vector_store(
        vector_store=pdf_vector_store,
        embed_model=embed_model
    )

    pdf_nodes_by_faculty = {}

    info(f"Querying PDF store for {len(faculty_ids)} faculty members...")
    debug(f"PDF collection: {pdf_collection.name}, top_k_per_faculty: {top_k_per_faculty}")
    verbose(f"Faculty IDs to query: {faculty_ids[:10]}...")

    for i, faculty_id in enumerate(faculty_ids, 1):
        debug(f"Querying PDFs for faculty {i}/{len(faculty_ids)}: {faculty_id}")

        # Create a retriever with metadata filter using proper format
        verbose(f"Creating retriever with filter: faculty_id={faculty_id}, top_k={top_k_per_faculty}")
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="faculty_id", value=faculty_id)]
        )
        retriever = pdf_index.as_retriever(
            similarity_top_k=top_k_per_faculty,
            filters=filters
        )

        # Query with a generic prompt (the filter does the real work)
        query = f"Research, publications, and expertise of faculty member {faculty_id}"
        verbose(f"Query: {query}")

        try:
            nodes = retriever.retrieve(query)
            debug(f"Retrieved {len(nodes)} PDF nodes for faculty {faculty_id}")
            if nodes:
                pdf_nodes_by_faculty[faculty_id] = nodes
                verbose(f"PDF nodes scores: {[getattr(n, 'score', 'N/A') for n in nodes[:3]]}")
            else:
                debug(f"No PDF nodes found for faculty {faculty_id}")
        except Exception as e:
            warning(f"Error retrieving PDFs for {faculty_id}: {e}")
            verbose(f"PDF retrieval exception: {type(e).__name__}: {str(e)}")
            continue

    info(f"Retrieved PDF nodes for {len(pdf_nodes_by_faculty)} faculty members")
    debug(f"PDF nodes distribution: {[(fid, len(nodes)) for fid, nodes in list(pdf_nodes_by_faculty.items())[:5]]}")
    verbose(f"Total PDF nodes retrieved: {sum(len(nodes) for nodes in pdf_nodes_by_faculty.values())}")

    return pdf_nodes_by_faculty

