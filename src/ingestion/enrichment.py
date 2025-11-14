"""Enrichment module for extracting PDF information and enriching faculty profile documents."""

from typing import List, Dict, Optional
from llama_index.core import Document
from llama_index.core.schema import NodeWithScore
from src.models.llm import get_llm


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
    if not pdf_nodes:
        return {
            "summary": None,
            "research_interests": None,
            "publications": None
        }

    if llm is None:
        llm = get_llm()

    # Combine text from all PDF nodes (limit to avoid token limits)
    combined_text = ""
    max_chars = 8000  # Leave room for prompt

    for node in pdf_nodes:
        node_text = node.node.text if hasattr(node, 'node') else node.text
        if len(combined_text) + len(node_text) < max_chars:
            combined_text += node_text + "\n\n"
        else:
            break

    if not combined_text.strip():
        return {
            "summary": None,
            "research_interests": None,
            "publications": None
        }

    # Create prompt for extraction
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

    try:
        response = llm.complete(prompt)
        response_text = str(response).strip()

        # Parse response
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

        return result

    except Exception as e:
        print(f"  ⚠ Error extracting enrichment data for {faculty_id}: {e}")
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

    print(f"\n[Enrichment] Processing {len(csv_docs)} faculty profiles...")

    for i, doc in enumerate(csv_docs, 1):
        faculty_id = doc.metadata.get("faculty_id")
        faculty_name = doc.metadata.get("faculty_name", "Unknown")

        if not faculty_id:
            print(f"  [{i}/{len(csv_docs)}] Skipping {faculty_name}: no faculty_id")
            enriched_docs.append(doc)
            continue

        # Get PDF nodes for this faculty
        pdf_nodes = pdf_nodes_by_faculty.get(faculty_id, [])

        if not pdf_nodes:
            print(f"  [{i}/{len(csv_docs)}] {faculty_name} ({faculty_id}): no PDFs found")
            enriched_docs.append(doc)
            continue

        print(f"  [{i}/{len(csv_docs)}] Enriching {faculty_name} ({faculty_id}) with {len(pdf_nodes)} PDF nodes...")

        # Extract enrichment data
        enrichment = enrich_faculty_profile(faculty_id, pdf_nodes, llm)

        # Build enriched text
        enriched_text_parts = [doc.text]  # Start with original CSV text

        if enrichment["summary"] or enrichment["research_interests"] or enrichment["publications"]:
            enriched_text_parts.append("\n--- Additional Information from CVs/Publications ---")

            if enrichment["summary"]:
                enriched_text_parts.append(f"\nSummary: {enrichment['summary']}")
                # Also add to metadata
                doc.metadata["pdf_summary"] = enrichment["summary"]

            if enrichment["research_interests"]:
                enriched_text_parts.append(f"\nAdditional Research Interests: {enrichment['research_interests']}")
                doc.metadata["pdf_research_interests"] = enrichment["research_interests"]

            if enrichment["publications"]:
                enriched_text_parts.append(f"\nNotable Publications: {enrichment['publications']}")
                doc.metadata["pdf_publications"] = enrichment["publications"]

        # Create enriched document
        enriched_doc = Document(
            text="\n".join(enriched_text_parts),
            metadata=doc.metadata.copy()
        )

        enriched_docs.append(enriched_doc)

    print(f"✓ Enrichment complete: {len(enriched_docs)} profiles enriched")

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
    from src.models.embeddings import get_embedding_model

    # Create vector store and index for PDFs
    pdf_vector_store = ChromaVectorStore(chroma_collection=pdf_collection)
    embed_model = get_embedding_model()

    pdf_index = VectorStoreIndex.from_vector_store(
        vector_store=pdf_vector_store,
        embed_model=embed_model
    )

    pdf_nodes_by_faculty = {}

    print(f"\n[Enrichment] Querying PDF store for {len(faculty_ids)} faculty members...")

    for faculty_id in faculty_ids:
        # Create a retriever with metadata filter
        retriever = pdf_index.as_retriever(
            similarity_top_k=top_k_per_faculty,
            filters={"faculty_id": faculty_id}
        )

        # Query with a generic prompt (the filter does the real work)
        query = f"Research, publications, and expertise of faculty member {faculty_id}"

        try:
            nodes = retriever.retrieve(query)
            if nodes:
                pdf_nodes_by_faculty[faculty_id] = nodes
        except Exception as e:
            print(f"  ⚠ Error retrieving PDFs for {faculty_id}: {e}")
            continue

    print(f"✓ Retrieved PDF nodes for {len(pdf_nodes_by_faculty)} faculty members")

    return pdf_nodes_by_faculty

