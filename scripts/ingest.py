"""Data ingestion script - main entry point for indexing data."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pipeline import run_ingestion_pipeline_sync, preview_documents
from src.utils.config import Config
from src.indexing.vector_store import get_collection_stats


def main():
    """Main ingestion script."""
    print("Faculty Research Matchmaker - Data Ingestion")
    print("=" * 60)

    # Ensure directories exist
    Config.ensure_directories()

    # Show configuration
    print("\nConfiguration:")
    config_summary = Config.get_config_summary()
    for key, value in config_summary.items():
        print(f"  {key}: {value}")
    print()

    # Preview documents
    print("\n[Preview] Checking what documents will be loaded...")
    preview = preview_documents()

    print(f"\nCSV Documents: {preview['csv']['count']}")
    if preview['csv'].get('error'):
        print(f"  Error: {preview['csv']['error']}")
    elif preview['csv']['count'] > 0:
        print(f"  Sample metadata: {preview['csv']['sample']['metadata']}")

    print(f"\nPDF Documents: {preview['pdf']['count']}")
    if preview['pdf'].get('error'):
        print(f"  Error: {preview['pdf']['error']}")
    elif preview['pdf']['count'] > 0:
        print(f"  Types: {preview['pdf']['types']}")

    total_docs = preview['csv']['count'] + preview['pdf']['count']

    if total_docs == 0:
        print("\n⚠️  No documents found to ingest!")
        print("\nPlease add:")
        print(f"  - Faculty CSV file: {Config.CSV_PATH}")
        print(f"  - PDF files in: {Config.PDF_DIR}")
        return

    # Ask for confirmation
    print(f"\n📊 Ready to ingest {total_docs} document(s)")
    response = input("Proceed with ingestion? [y/N]: ").strip().lower()

    if response != 'y':
        print("Ingestion cancelled.")
        return

    # Check if we should reset the collections
    try:
        profiles_stats = get_collection_stats(Config.FACULTY_PROFILES_COLLECTION)
        pdfs_stats = get_collection_stats(Config.FACULTY_PDFS_COLLECTION)

        total_items = profiles_stats['count'] + pdfs_stats['count']

        if total_items > 0:
            print(f"\n⚠️  Existing data found:")
            print(f"  - {Config.FACULTY_PROFILES_COLLECTION}: {profiles_stats['count']} items")
            print(f"  - {Config.FACULTY_PDFS_COLLECTION}: {pdfs_stats['count']} items")
            reset = input("Reset collections and start fresh? [y/N]: ").strip().lower()
            reset_collection = (reset == 'y')
        else:
            reset_collection = False
    except Exception:
        reset_collection = False

    # Run ingestion
    try:
        num_nodes = run_ingestion_pipeline_sync(reset_collection=reset_collection)

        print("\n✅ Ingestion completed successfully!")
        print(f"Created {num_nodes} nodes in the vector store")

        # Show final stats for both collections
        print("\nFinal collection stats:")
        try:
            profiles_stats = get_collection_stats(Config.FACULTY_PROFILES_COLLECTION)
            pdfs_stats = get_collection_stats(Config.FACULTY_PDFS_COLLECTION)

            print(f"  📋 {Config.FACULTY_PROFILES_COLLECTION}: {profiles_stats['count']} items")
            print(f"  📄 {Config.FACULTY_PDFS_COLLECTION}: {pdfs_stats['count']} items")
            print(f"  📊 Total: {profiles_stats['count'] + pdfs_stats['count']} items")
        except Exception as e:
            print(f"  ⚠️  Could not retrieve final stats: {e}")

    except Exception as e:
        print(f"\n❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

