"""PDF loader for faculty documents (CVs, papers, proposals)."""

from pathlib import Path
from typing import List, Optional, Dict
from llama_index.core import SimpleDirectoryReader, Document
from src.utils.config import Config
from src.models.llm import get_llm
from src.utils.faculty_id import load_faculty_id_mapping, map_name_to_faculty_id
from src.utils.logger import get_logger, debug, info, warning, error, verbose

logger = get_logger(__name__)


def _extract_metadata_with_llm(text: str, llm=None) -> Dict[str, Optional[str]]:
    """
    Extract summary, research interests, and faculty name from PDF text using LLM.

    Args:
        text: The PDF text content.
        llm: LLM instance. If None, uses default from config.

    Returns:
        Dict with keys: summary, research_interests, faculty_name
    """
    debug("Extracting metadata from PDF text using LLM...")
    verbose(f"Input text length: {len(text)} characters")

    if llm is None:
        debug("Initializing LLM for metadata extraction...")
        llm = get_llm()

    # Truncate text if too long (keep first portion which usually has key info)
    max_chars = 10000
    if len(text) > max_chars:
        text_sample = text[:max_chars] + "\n\n[... text truncated ...]"
        debug(f"Text truncated from {len(text)} to {max_chars} characters")
    else:
        text_sample = text
        debug(f"Text length within limit: {len(text)} characters")

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

    verbose(f"Prompt length: {len(prompt)} characters")
    debug("Querying LLM for metadata extraction...")

    try:
        response = llm.complete(prompt)
        response_text = str(response)
        debug(f"LLM response received: {len(response_text)} characters")
        verbose(f"LLM response preview: {response_text[:300]}...")

        # Parse response
        debug("Parsing LLM response...")
        result = {
            "faculty_name": None,
            "summary": None,
            "research_interests": None
        }

        lines = response_text.strip().split("\n")
        verbose(f"Response has {len(lines)} lines")
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

        debug("Metadata extraction complete")
        verbose(f"Extracted: faculty_name={bool(result['faculty_name'])}, summary={bool(result['summary'])}, research_interests={bool(result['research_interests'])}")
        if result['faculty_name']:
            verbose(f"Faculty name: {result['faculty_name']}")
        if result['summary']:
            verbose(f"Summary preview: {result['summary'][:100]}...")
        if result['research_interests']:
            verbose(f"Research interests: {result['research_interests']}")

        return result
    except Exception as e:
        error(f"Error extracting metadata with LLM: {e}")
        verbose(f"Metadata extraction exception: {type(e).__name__}: {str(e)}")
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
    info(f"Loading PDFs from directory: {dir_path}")

    if not dir_path.exists():
        warning(f"PDF directory not found: {dir_path}")
        info("Creating directory...")
        dir_path.mkdir(parents=True, exist_ok=True)
        debug(f"Created directory: {dir_path}")
        return []

    debug(f"PDF directory exists: {dir_path}")
    verbose(f"Directory contents: {list(dir_path.iterdir())[:5]}...")

    # Use SimpleDirectoryReader to load PDFs
    debug("Initializing SimpleDirectoryReader for PDFs...")
    reader = SimpleDirectoryReader(
        input_dir=str(dir_path),
        recursive=True,
        required_exts=[".pdf"],
    )

    try:
        debug("Loading PDF documents...")
        documents = reader.load_data()
        info(f"Loaded {len(documents)} documents from PDFs")
        verbose(f"Document sources: {[doc.metadata.get('file_name', 'unknown') for doc in documents[:5]]}...")

        # Use config default if extract_metadata not specified
        if extract_metadata is None:
            extract_metadata = Config.EXTRACT_PDF_METADATA_WITH_LLM
        debug(f"Extract metadata with LLM: {extract_metadata}")

        # Group documents by file (PDFs may be split into multiple pages)
        documents_by_file = {}
        for doc in documents:
            file_name = doc.metadata.get("file_name", "unknown")
            if file_name not in documents_by_file:
                documents_by_file[file_name] = []
            documents_by_file[file_name].append(doc)

        debug(f"Grouped documents into {len(documents_by_file)} PDF files")
        verbose(f"PDF files: {list(documents_by_file.keys())[:5]}...")

        # Load faculty_id mapping from CSV
        debug("Loading faculty_id mapping from CSV...")
        faculty_name_to_id = load_faculty_id_mapping()
        debug(f"Loaded {len(faculty_name_to_id)} faculty name mappings")
        verbose(f"Faculty mappings preview: {list(faculty_name_to_id.items())[:3]}...")

        # Enhance metadata for each document group
        llm = get_llm() if extract_metadata else None
        if extract_metadata:
            debug("LLM initialized for metadata extraction")
        enhanced_documents = []

        info(f"Processing {len(documents_by_file)} PDF files for metadata enhancement...")
        for i, (file_name, doc_group) in enumerate(documents_by_file.items(), 1):
            debug(f"Processing PDF {i}/{len(documents_by_file)}: {file_name} ({len(doc_group)} pages)")
            # Combine all pages for this PDF
            full_text = "\n\n".join([doc.text for doc in doc_group])
            verbose(f"Combined text length for {file_name}: {len(full_text)} characters")

            # Extract metadata from filename first
            doc_type = _infer_pdf_type(file_name)
            debug(f"Inferred PDF type: {doc_type}")
            faculty_name_from_filename = _extract_faculty_name_from_filename(file_name)
            if faculty_name_from_filename:
                debug(f"Extracted faculty name from filename: {faculty_name_from_filename}")
            else:
                debug(f"Could not extract faculty name from filename: {file_name}")

            # Extract metadata with LLM if enabled
            llm_metadata = {}
            if extract_metadata and full_text.strip():
                info(f"Extracting metadata from: {file_name}...")
                llm_metadata = _extract_metadata_with_llm(full_text, llm)
                debug(f"LLM metadata extraction complete for {file_name}")

            # Determine faculty_name for this PDF
            faculty_name = None
            if llm_metadata.get("faculty_name"):
                faculty_name = llm_metadata["faculty_name"]
                debug(f"Using faculty name from LLM: {faculty_name}")
            elif faculty_name_from_filename:
                faculty_name = faculty_name_from_filename
                debug(f"Using faculty name from filename: {faculty_name}")
            else:
                debug(f"No faculty name found for {file_name}")

            # Map faculty_name to faculty_id
            faculty_id = None
            if faculty_name:
                debug(f"Mapping faculty name '{faculty_name}' to faculty_id...")
                verbose(f"Available CSV names: {list(faculty_name_to_id.keys())[:5]}...")
                faculty_id = map_name_to_faculty_id(faculty_name, faculty_name_to_id)
                if faculty_id:
                    debug(f"Mapped '{faculty_name}' to faculty_id: {faculty_id}")
                else:
                    warning(f"Could not map faculty name '{faculty_name}' to faculty_id")
                    verbose(f"Tried matching against {len(faculty_name_to_id)} CSV names")
                    # Try to find close matches for debugging
                    from src.utils.name_matcher import normalize_faculty_name
                    norm_pdf_name = normalize_faculty_name(faculty_name)
                    close_matches = []
                    for csv_name in faculty_name_to_id.keys():
                        norm_csv_name = normalize_faculty_name(csv_name)
                        # Check if last names match
                        pdf_words = norm_pdf_name.split()
                        csv_words = norm_csv_name.split()
                        if pdf_words and csv_words and pdf_words[-1] == csv_words[-1]:
                            close_matches.append(csv_name)
                    if close_matches:
                        verbose(f"Close matches (same last name): {close_matches[:3]}")

            # Apply metadata to all documents in this group
            verbose(f"Applying metadata to {len(doc_group)} document pages...")
            for j, doc in enumerate(doc_group, 1):
                doc.metadata["source"] = "pdf"
                doc.metadata["type"] = "faculty_pdf"  # Changed to "faculty_pdf" for dual-store architecture

                # Faculty name
                if faculty_name:
                    doc.metadata["faculty_name"] = faculty_name
                    verbose(f"  Page {j}: Added faculty_name: {faculty_name}")

                # Faculty ID (for linking to CSV profiles)
                if faculty_id:
                    doc.metadata["faculty_id"] = faculty_id
                    verbose(f"  Page {j}: Added faculty_id: {faculty_id}")

                # Add LLM-extracted metadata
                if llm_metadata.get("summary"):
                    doc.metadata["summary"] = llm_metadata["summary"]
                    verbose(f"  Page {j}: Added summary ({len(llm_metadata['summary'])} chars)")
                if llm_metadata.get("research_interests"):
                    interests_str = ", ".join(llm_metadata["research_interests"])
                    doc.metadata["research_interests"] = interests_str
                    verbose(f"  Page {j}: Added research_interests: {interests_str[:50]}...")

                # Add document type (cv, paper, etc.)
                doc.metadata["pdf_type"] = doc_type
                verbose(f"  Page {j}: Added pdf_type: {doc_type}")

            enhanced_documents.extend(doc_group)
            debug(f"Enhanced {len(doc_group)} pages for {file_name}")

        info(f"PDF loading complete: {len(enhanced_documents)} documents processed")
        debug(f"Total enhanced documents: {len(enhanced_documents)}")
        faculty_ids_found = set(doc.metadata.get("faculty_id") for doc in enhanced_documents if doc.metadata.get("faculty_id"))
        debug(f"Faculty IDs found: {len(faculty_ids_found)} unique IDs")

        return enhanced_documents

    except Exception as e:
        error(f"Error loading PDFs: {e}")
        verbose(f"PDF loading exception: {type(e).__name__}: {str(e)}")
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
    info(f"Loading single PDF: {path}")

    if not path.exists():
        error(f"PDF file not found: {path}")
        raise FileNotFoundError(f"PDF file not found: {path}")

    debug(f"PDF file exists: {path}")
    verbose(f"File size: {path.stat().st_size} bytes")

    reader = SimpleDirectoryReader(input_files=[str(path)])
    debug("Initialized SimpleDirectoryReader for single PDF")

    try:
        debug("Loading PDF documents...")
        documents = reader.load_data()
        info(f"Loaded {len(documents)} documents from PDF")
        verbose(f"Documents metadata: {[doc.metadata for doc in documents[:2]]}...")

        # Combine all pages
        full_text = "\n\n".join([doc.text for doc in documents])
        debug(f"Combined text length: {len(full_text)} characters")

        # Extract metadata from filename
        doc_type = _infer_pdf_type(path.name)
        debug(f"Inferred PDF type: {doc_type}")
        faculty_name_from_filename = _extract_faculty_name_from_filename(path.name)
        if faculty_name_from_filename:
            debug(f"Extracted faculty name from filename: {faculty_name_from_filename}")

        # Use config default if extract_metadata not specified
        if extract_metadata is None:
            extract_metadata = Config.EXTRACT_PDF_METADATA_WITH_LLM
        debug(f"Extract metadata with LLM: {extract_metadata}")

        # Extract metadata with LLM if enabled
        llm_metadata = {}
        if extract_metadata and full_text.strip():
            info(f"Extracting metadata from: {path.name}...")
            llm = get_llm()
            llm_metadata = _extract_metadata_with_llm(full_text, llm)
            debug(f"LLM metadata extraction complete")

        # Determine faculty_name for this PDF
        faculty_name = None
        if llm_metadata.get("faculty_name"):
            faculty_name = llm_metadata["faculty_name"]
            debug(f"Using faculty name from LLM: {faculty_name}")
        elif faculty_name_from_filename:
            faculty_name = faculty_name_from_filename
            debug(f"Using faculty name from filename: {faculty_name}")

        # Load faculty_id mapping and map faculty_name to faculty_id
        debug("Loading faculty_id mapping from CSV...")
        faculty_name_to_id = load_faculty_id_mapping()
        faculty_id = None
        if faculty_name:
            debug(f"Mapping faculty name '{faculty_name}' to faculty_id...")
            faculty_id = map_name_to_faculty_id(faculty_name, faculty_name_to_id)
            if faculty_id:
                debug(f"Mapped to faculty_id: {faculty_id}")
            else:
                warning(f"Could not map faculty name '{faculty_name}' to faculty_id")

        # Enhance metadata
        debug(f"Enhancing metadata for {len(documents)} document pages...")
        for i, doc in enumerate(documents, 1):
            doc.metadata["source"] = "pdf"
            doc.metadata["type"] = "faculty_pdf"  # Changed to "faculty_pdf" for dual-store architecture

            # Faculty name
            if faculty_name:
                doc.metadata["faculty_name"] = faculty_name
                verbose(f"  Page {i}: Added faculty_name: {faculty_name}")

            # Faculty ID (for linking to CSV profiles)
            if faculty_id:
                doc.metadata["faculty_id"] = faculty_id
                verbose(f"  Page {i}: Added faculty_id: {faculty_id}")

            # Add LLM-extracted metadata
            if llm_metadata.get("summary"):
                doc.metadata["summary"] = llm_metadata["summary"]
                verbose(f"  Page {i}: Added summary ({len(llm_metadata['summary'])} chars)")
            if llm_metadata.get("research_interests"):
                interests_str = ", ".join(llm_metadata["research_interests"])
                doc.metadata["research_interests"] = interests_str
                verbose(f"  Page {i}: Added research_interests: {interests_str[:50]}...")

            # Add document type (cv, paper, etc.)
            doc.metadata["pdf_type"] = doc_type
            verbose(f"  Page {i}: Added pdf_type: {doc_type}")

        info(f"Single PDF loading complete: {len(documents)} documents processed")
        return documents

    except Exception as e:
        error(f"Error loading PDF {path}: {e}")
        verbose(f"Single PDF loading exception: {type(e).__name__}: {str(e)}")
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
    verbose(f"Inferring PDF type from filename: {filename}")

    if "cv" in filename_lower or "resume" in filename_lower:
        doc_type = "cv"
    elif "paper" in filename_lower or "publication" in filename_lower:
        doc_type = "paper"
    elif "proposal" in filename_lower:
        doc_type = "proposal"
    else:
        doc_type = "document"

    debug(f"Inferred type '{doc_type}' for {filename}")
    return doc_type


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
    verbose(f"Extracting faculty name from filename: {filename}")
    # Remove extension
    name_part = Path(filename).stem
    debug(f"Filename stem: {name_part}")

    # Look for common separators before type indicators
    for separator in ["_CV", "_cv", "_Paper", "_paper", "_Proposal", "_proposal"]:
        if separator in name_part:
            faculty_part = name_part.split(separator)[0]
            # Replace underscores/dots with spaces and title case
            faculty_name = faculty_part.replace("_", " ").replace(".", " ").title()
            debug(f"Extracted faculty name using separator '{separator}': {faculty_name}")
            return faculty_name

    # If no separator found, assume the entire filename is the faculty name
    if name_part:  # Make sure it's not empty
        faculty_name = name_part.replace("_", " ").replace(".", " ").title()
        debug(f"Using entire filename as faculty name: {faculty_name}")
        return faculty_name

    debug(f"Could not extract faculty name from: {filename}")
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
    debug(f"Getting PDF statistics for directory: {dir_path}")

    if not dir_path.exists():
        warning(f"PDF directory not found for stats: {dir_path}")
        return {
            "exists": False,
            "path": str(dir_path),
            "pdf_count": 0,
        }

    debug(f"Scanning for PDF files in: {dir_path}")
    pdf_files = list(dir_path.rglob("*.pdf"))
    info(f"Found {len(pdf_files)} PDF files")

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
    debug(f"PDF type distribution: {type_counts}")

    return stats

