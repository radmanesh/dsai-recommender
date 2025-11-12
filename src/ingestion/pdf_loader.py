"""PDF loader for faculty documents (CVs, papers, proposals)."""

from pathlib import Path
from typing import List, Optional, Dict
from llama_index.core import SimpleDirectoryReader, Document
from src.utils.config import Config
from src.models.llm import get_llm


def _extract_metadata_with_llm(text: str, llm=None) -> Dict[str, Optional[str]]:
    """
    Extract summary, research interests, and faculty name from PDF text using LLM.

    Args:
        text: The PDF text content.
        llm: LLM instance. If None, uses default from config.

    Returns:
        Dict with keys: summary, research_interests, faculty_name
    """
    if llm is None:
        llm = get_llm()

    # Truncate text if too long (keep first portion which usually has key info)
    max_chars = 10000
    if len(text) > max_chars:
        text_sample = text[:max_chars] + "\n\n[... text truncated ...]"
    else:
        text_sample = text

    prompt = f"""You are an expert academic document analyzer. Analyze the following document and extract key information.

DOCUMENT TEXT:
{text_sample}

Please extract the following information:

1. FACULTY_NAME: Extract the faculty member's full name (if mentioned). If not found, return "N/A"
2. SUMMARY: Write a concise 2-3 sentence summary of the document's main content
3. RESEARCH_INTERESTS: List the main research interests, areas, or topics mentioned (comma-separated). If not found, return "N/A"

Format your response EXACTLY as follows:

FACULTY_NAME: [name or N/A]
SUMMARY: [your summary here]
RESEARCH_INTERESTS: [interest1, interest2, interest3, ... or N/A]

Be specific and use technical terminology where appropriate."""

    try:
        response = llm.complete(prompt)
        response_text = str(response)

        # Parse response
        result = {
            "faculty_name": None,
            "summary": None,
            "research_interests": None
        }

        lines = response_text.strip().split("\n")
        current_field = None
        summary_lines = []

        for line in lines:
            line = line.strip()

            if line.startswith("FACULTY_NAME:"):
                name = line.replace("FACULTY_NAME:", "").strip()
                if name and name.upper() != "N/A":
                    result["faculty_name"] = name
            elif line.startswith("SUMMARY:"):
                current_field = "SUMMARY"
                summary_part = line.replace("SUMMARY:", "").strip()
                if summary_part:
                    summary_lines.append(summary_part)
            elif line.startswith("RESEARCH_INTERESTS:"):
                current_field = None
                interests = line.replace("RESEARCH_INTERESTS:", "").strip()
                if interests and interests.upper() != "N/A":
                    # Remove brackets if present and split by comma
                    interests = interests.strip("[]")
                    result["research_interests"] = [item.strip() for item in interests.split(",") if item.strip()]
            elif current_field == "SUMMARY" and line:
                summary_lines.append(line)

        result["summary"] = " ".join(summary_lines) if summary_lines else None

        return result
    except Exception as e:
        print(f"  ⚠ Error extracting metadata with LLM: {e}")
        return {
            "faculty_name": None,
            "summary": None,
            "research_interests": None
        }


def load_pdfs_from_directory(pdf_dir: str = None, extract_metadata: bool = None) -> List[Document]:
    """
    Load all PDFs from a directory using LlamaIndex SimpleDirectoryReader.

    Args:
        pdf_dir: Path to the directory containing PDFs. Defaults to Config.PDF_DIR.
        extract_metadata: If True, use LLM to extract summary, research interests, and faculty name.
                         Defaults to Config.EXTRACT_PDF_METADATA_WITH_LLM.

    Returns:
        List[Document]: List of LlamaIndex Document objects.
    """
    dir_path = Path(pdf_dir) if pdf_dir else Config.PDF_DIR

    if not dir_path.exists():
        print(f"Warning: PDF directory not found: {dir_path}")
        print("Creating directory...")
        dir_path.mkdir(parents=True, exist_ok=True)
        return []

    print(f"Loading PDFs from: {dir_path}")

    # Use SimpleDirectoryReader to load PDFs
    reader = SimpleDirectoryReader(
        input_dir=str(dir_path),
        recursive=True,
        required_exts=[".pdf"],
    )

    try:
        documents = reader.load_data()
        print(f"Loaded {len(documents)} documents from PDFs")

        # Use config default if extract_metadata not specified
        if extract_metadata is None:
            extract_metadata = Config.EXTRACT_PDF_METADATA_WITH_LLM

        # Group documents by file (PDFs may be split into multiple pages)
        documents_by_file = {}
        for doc in documents:
            file_name = doc.metadata.get("file_name", "unknown")
            if file_name not in documents_by_file:
                documents_by_file[file_name] = []
            documents_by_file[file_name].append(doc)

        # Enhance metadata for each document group
        llm = get_llm() if extract_metadata else None
        enhanced_documents = []

        for file_name, doc_group in documents_by_file.items():
            # Combine all pages for this PDF
            full_text = "\n\n".join([doc.text for doc in doc_group])

            # Extract metadata from filename first
            doc_type = _infer_pdf_type(file_name)
            faculty_name_from_filename = _extract_faculty_name_from_filename(file_name)

            # Extract metadata with LLM if enabled
            llm_metadata = {}
            if extract_metadata and full_text.strip():
                print(f"  Extracting metadata from: {file_name}...")
                llm_metadata = _extract_metadata_with_llm(full_text, llm)

            # Apply metadata to all documents in this group
            for doc in doc_group:
                doc.metadata["source"] = "pdf"
                doc.metadata["type"] = doc_type

                # Faculty name: prefer LLM extraction, fallback to filename
                if llm_metadata.get("faculty_name"):
                    doc.metadata["faculty_name"] = llm_metadata["faculty_name"]
                elif faculty_name_from_filename:
                    doc.metadata["faculty_name"] = faculty_name_from_filename

                # Add LLM-extracted metadata
                if llm_metadata.get("summary"):
                    doc.metadata["summary"] = llm_metadata["summary"]
                if llm_metadata.get("research_interests"):
                    doc.metadata["research_interests"] = ", ".join(llm_metadata["research_interests"])

            enhanced_documents.extend(doc_group)

        return enhanced_documents

    except Exception as e:
        print(f"Error loading PDFs: {e}")
        return []


def load_single_pdf(pdf_path: str, extract_metadata: bool = None) -> List[Document]:
    """
    Load a single PDF file.

    Args:
        pdf_path: Path to the PDF file.
        extract_metadata: If True, use LLM to extract summary, research interests, and faculty name.
                         Defaults to Config.EXTRACT_PDF_METADATA_WITH_LLM.

    Returns:
        List[Document]: List of Document objects (may be multiple pages).
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    print(f"Loading PDF: {path}")

    reader = SimpleDirectoryReader(input_files=[str(path)])

    try:
        documents = reader.load_data()
        print(f"Loaded {len(documents)} documents from PDF")

        # Combine all pages
        full_text = "\n\n".join([doc.text for doc in documents])

        # Extract metadata from filename
        doc_type = _infer_pdf_type(path.name)
        faculty_name_from_filename = _extract_faculty_name_from_filename(path.name)

        # Use config default if extract_metadata not specified
        if extract_metadata is None:
            extract_metadata = Config.EXTRACT_PDF_METADATA_WITH_LLM

        # Extract metadata with LLM if enabled
        llm_metadata = {}
        if extract_metadata and full_text.strip():
            print(f"  Extracting metadata from: {path.name}...")
            llm = get_llm()
            llm_metadata = _extract_metadata_with_llm(full_text, llm)

        # Enhance metadata
        for doc in documents:
            doc.metadata["source"] = "pdf"
            doc.metadata["type"] = doc_type

            # Faculty name: prefer LLM extraction, fallback to filename
            if llm_metadata.get("faculty_name"):
                doc.metadata["faculty_name"] = llm_metadata["faculty_name"]
            elif faculty_name_from_filename:
                doc.metadata["faculty_name"] = faculty_name_from_filename

            # Add LLM-extracted metadata
            if llm_metadata.get("summary"):
                doc.metadata["summary"] = llm_metadata["summary"]
            if llm_metadata.get("research_interests"):
                doc.metadata["research_interests"] = ", ".join(llm_metadata["research_interests"])

        return documents

    except Exception as e:
        print(f"Error loading PDF {path}: {e}")
        return []


def _infer_pdf_type(filename: str) -> str:
    """
    Infer the type of PDF from its filename.

    Args:
        filename: The PDF filename.

    Returns:
        str: Inferred type (cv, paper, proposal, or document).
    """
    filename_lower = filename.lower()

    if "cv" in filename_lower or "resume" in filename_lower:
        return "cv"
    elif "paper" in filename_lower or "publication" in filename_lower:
        return "paper"
    elif "proposal" in filename_lower:
        return "proposal"
    else:
        return "document"


def _extract_faculty_name_from_filename(filename: str) -> Optional[str]:
    """
    Try to extract faculty name from filename.

    Expected formats:
    - "John_Doe_CV.pdf"
    - "Jane_Smith_paper_title.pdf"
    - "john.doe_CV.pdf"

    Args:
        filename: The PDF filename.

    Returns:
        Optional[str]: Extracted faculty name or None.
    """
    # Remove extension
    name_part = Path(filename).stem

    # Look for common separators before type indicators
    for separator in ["_CV", "_cv", "_Paper", "_paper", "_Proposal", "_proposal"]:
        if separator in name_part:
            faculty_part = name_part.split(separator)[0]
            # Replace underscores/dots with spaces and title case
            faculty_name = faculty_part.replace("_", " ").replace(".", " ").title()
            return faculty_name

    # If no separator found, assume the entire filename is the faculty name
    if name_part:  # Make sure it's not empty
        faculty_name = name_part.replace("_", " ").replace(".", " ").title()
        return faculty_name

    return None

def get_pdf_stats(pdf_dir: str = None) -> dict:
    """
    Get statistics about PDFs in a directory.

    Args:
        pdf_dir: Path to the directory. Defaults to Config.PDF_DIR.

    Returns:
        dict: Statistics about the PDFs.
    """
    dir_path = Path(pdf_dir) if pdf_dir else Config.PDF_DIR

    if not dir_path.exists():
        return {
            "exists": False,
            "path": str(dir_path),
            "pdf_count": 0,
        }

    pdf_files = list(dir_path.rglob("*.pdf"))

    stats = {
        "exists": True,
        "path": str(dir_path),
        "pdf_count": len(pdf_files),
        "pdf_files": [f.name for f in pdf_files],
    }

    # Count by type
    type_counts = {}
    for pdf in pdf_files:
        pdf_type = _infer_pdf_type(pdf.name)
        type_counts[pdf_type] = type_counts.get(pdf_type, 0) + 1

    stats["type_counts"] = type_counts

    return stats

