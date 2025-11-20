"""Configuration management for the Agentic Research Matchmaker."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration class for all application settings."""

    # Base paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
    CSV_PATH = BASE_DIR / os.getenv("CSV_PATH", "data/DSAI-Faculties.csv")
    PDF_DIR = BASE_DIR / os.getenv("PDF_DIR", "data/pdfs")

    # HuggingFace API
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN environment variable is required. "
            "Please set it in your .env file or environment."
        )

    # ChromaDB Configuration
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./faculty_chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "faculty_all")

    # Separate collections for dual-store architecture
    FACULTY_PROFILES_COLLECTION = os.getenv("FACULTY_PROFILES_COLLECTION", "faculty_profiles")
    FACULTY_PDFS_COLLECTION = os.getenv("FACULTY_PDFS_COLLECTION", "faculty_pdfs")
    FACULTY_WEBSITES_COLLECTION = os.getenv("FACULTY_WEBSITES_COLLECTION", "faculty_websites")

    # Model Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

    # LLM Parameters
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
    USE_8BIT_QUANTIZATION = os.getenv("USE_8BIT_QUANTIZATION", "false").lower() == "true"
    LLM_DEVICE = os.getenv("LLM_DEVICE", "auto")

    # Retrieval Configuration
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "10"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "128"))

    # Ingestion Configuration
    # ChromaDB has a max batch size limit (typically ~5461), so we use a safe batch size
    INGESTION_BATCH_SIZE = int(os.getenv("INGESTION_BATCH_SIZE", "5000"))
    # Whether to use LLM to extract metadata (summary, research interests, faculty name) from PDFs
    EXTRACT_PDF_METADATA_WITH_LLM = os.getenv("EXTRACT_PDF_METADATA_WITH_LLM", "true").lower() == "true"

    # Debug Configuration
    # Debug levels: ERROR=0, WARNING=1, INFO=2, DEBUG=3, VERBOSE=4
    # Can be set as integer (0-4) or string ("ERROR", "WARNING", "INFO", "DEBUG", "VERBOSE")
    DEBUG_LEVEL = os.getenv("DEBUG_LEVEL", "DEBUG").upper()

    @classmethod
    def validate(cls):
        """Validate that required configuration is present."""
        errors = []

        if not cls.HF_TOKEN:
            errors.append("HF_TOKEN is required")

        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")

    @classmethod
    def ensure_directories(cls):
        """Ensure required directories exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.PDF_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_config_summary(cls):
        """Return a summary of current configuration (excluding sensitive data)."""
        return {
            "data_dir": str(cls.DATA_DIR),
            "csv_path": str(cls.CSV_PATH),
            "pdf_dir": str(cls.PDF_DIR),
            "chroma_path": cls.CHROMA_PATH,
            "collection_name": cls.COLLECTION_NAME,
            "embedding_model": cls.EMBEDDING_MODEL,
            "llm_model": cls.LLM_MODEL,
            "llm_temperature": cls.LLM_TEMPERATURE,
            "llm_max_tokens": cls.LLM_MAX_TOKENS,
            "top_k_results": cls.TOP_K_RESULTS,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "debug_level": cls.DEBUG_LEVEL,
        }


# Validate configuration on import
try:
    Config.validate()
except ValueError as e:
    print(f"Warning: {e}")
    print("Some features may not work without proper configuration.")

