"""System test script - verify all components are working."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from src.utils.config import Config
        from src.models.embeddings import get_embedding_model
        from src.models.llm import get_llm
        from src.indexing.vector_store import get_chroma_client
        from src.indexing.index_builder import IndexManager
        from src.ingestion.csv_loader import load_faculty_csv
        from src.ingestion.pdf_loader import load_pdfs_from_directory
        from src.ingestion.pipeline import preview_documents
        from src.agents.proposal_analyzer import ProposalAnalyzer
        from src.agents.faculty_retriever import FacultyRetriever
        from src.agents.recommender import RecommendationAgent
        from src.agents.orchestrator import ResearchMatchOrchestrator

        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")

    try:
        from src.utils.config import Config

        # Try to access config (may fail if HF_TOKEN not set)
        try:
            token_set = Config.HF_TOKEN is not None and len(Config.HF_TOKEN) > 0
            print(f"  HF_TOKEN set: {token_set}")
        except ValueError as e:
            print(f"  ⚠️  {e}")
            return False

        print(f"  Data directory: {Config.DATA_DIR}")
        print(f"  CSV path: {Config.CSV_PATH}")
        print(f"  PDF directory: {Config.PDF_DIR}")
        print("✓ Configuration OK")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def test_data_files():
    """Test that data files exist."""
    print("\nTesting data files...")

    from src.utils.config import Config

    csv_exists = Config.CSV_PATH.exists()
    pdf_dir_exists = Config.PDF_DIR.exists()

    print(f"  CSV file exists: {csv_exists}")
    print(f"  PDF directory exists: {pdf_dir_exists}")

    if csv_exists:
        from src.ingestion.csv_loader import validate_csv_format
        info = validate_csv_format()
        if info['valid']:
            print(f"  CSV rows: {info['rows']}")
            print(f"  CSV columns: {', '.join(info['columns'])}")
        else:
            print(f"  ⚠️  CSV validation error: {info.get('error')}")

    if pdf_dir_exists:
        from src.ingestion.pdf_loader import get_pdf_stats
        stats = get_pdf_stats()
        print(f"  PDF files: {stats['pdf_count']}")
        if stats['pdf_count'] > 0:
            print(f"  PDF types: {stats.get('type_counts', {})}")

    if csv_exists or pdf_dir_exists:
        print("✓ Data files found")
        return True
    else:
        print("⚠️  No data files found (but this is OK for initial setup)")
        return True


def test_chroma_connection():
    """Test ChromaDB connection."""
    print("\nTesting ChromaDB connection...")

    try:
        from src.indexing.vector_store import get_chroma_client, list_collections

        client = get_chroma_client()
        print("  ✓ ChromaDB client initialized")

        collections = list_collections()
        print(f"  Collections found: {len(collections)}")

        print("✓ ChromaDB connection OK")
        return True
    except Exception as e:
        print(f"✗ ChromaDB error: {e}")
        return False


def test_models():
    """Test model initialization (requires HF_TOKEN)."""
    print("\nTesting model initialization...")

    try:
        from src.models.embeddings import get_embedding_model
        from src.models.llm import get_llm

        print("  Initializing embedding model...")
        embed_model = get_embedding_model()
        print("  ✓ Embedding model ready")

        print("  Initializing LLM...")
        llm = get_llm()
        print("  ✓ LLM ready")

        print("✓ Models initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Model initialization error: {e}")
        print("  Note: This requires a valid HF_TOKEN and internet connection")
        return False


def test_agents():
    """Test agent initialization."""
    print("\nTesting agent initialization...")

    try:
        from src.agents.proposal_analyzer import ProposalAnalyzer
        from src.agents.faculty_retriever import FacultyRetriever
        from src.agents.recommender import RecommendationAgent
        from src.agents.orchestrator import ResearchMatchOrchestrator

        print("  Creating agents...")
        analyzer = ProposalAnalyzer()
        retriever = FacultyRetriever()
        recommender = RecommendationAgent()
        orchestrator = ResearchMatchOrchestrator()

        print("✓ All agents initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Agent initialization error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print(" SYSTEM TEST SUITE")
    print("=" * 80)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_configuration()))
    results.append(("Data Files", test_data_files()))
    results.append(("ChromaDB", test_chroma_connection()))

    # These tests require HF_TOKEN and may download models
    try:
        results.append(("Models", test_models()))
        results.append(("Agents", test_agents()))
    except Exception as e:
        print(f"\n⚠️  Advanced tests skipped: {e}")

    # Summary
    print("\n" + "=" * 80)
    print(" TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! System is ready.")
        print("\nNext steps:")
        print("  1. Ensure HF_TOKEN is set in your .env file")
        print("  2. Add data: CSV file and PDFs")
        print("  3. Run ingestion: python scripts/ingest.py")
        print("  4. Try the demo: python scripts/demo.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

