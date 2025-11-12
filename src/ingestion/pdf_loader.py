"""PDF loader for faculty documents (CVs, papers, proposals)."""

from pathlib import Path
from typing import List, Optional
from llama_index.core import SimpleDirectoryReader, Document
from src.utils.config import Config


def load_pdfs_from_directory(pdf_dir: str = None) -> List[Document]:
    """
    Load all PDFs from a directory using LlamaIndex SimpleDirectoryReader.

    Args:
        pdf_dir: Path to the directory containing PDFs. Defaults to Config.PDF_DIR.

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

        # Enhance metadata for each document
        for doc in documents:
            # Add source type
            doc.metadata["source"] = "pdf"
            doc.metadata["type"] = _infer_pdf_type(doc.metadata.get("file_name", ""))

            # Try to extract faculty name from filename
            faculty_name = _extract_faculty_name_from_filename(
                doc.metadata.get("file_name", "")
            )
            if faculty_name:
                doc.metadata["faculty_name"] = faculty_name

        return documents

    except Exception as e:
        print(f"Error loading PDFs: {e}")
        return []


def load_single_pdf(pdf_path: str) -> List[Document]:
    """
    Load a single PDF file.

    Args:
        pdf_path: Path to the PDF file.

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

        # Enhance metadata
        for doc in documents:
            doc.metadata["source"] = "pdf"
            doc.metadata["type"] = _infer_pdf_type(path.name)

            faculty_name = _extract_faculty_name_from_filename(path.name)
            if faculty_name:
                doc.metadata["faculty_name"] = faculty_name

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

    # If no clear separator, return None
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

