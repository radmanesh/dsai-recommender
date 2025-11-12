"""Collection management utility - inspect and manage ChromaDB collections."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing.vector_store import (
    list_collections,
    get_collection_stats,
    delete_collection,
)


def show_collections():
    """Show all collections and their stats."""
    print("=" * 80)
    print(" ChromaDB Collections")
    print("=" * 80)

    collections = list_collections()

    if not collections:
        print("\nNo collections found.")
        return

    for col in collections:
        print(f"\n{col.name}:")
        stats = get_collection_stats(col.name)
        print(f"  Items: {stats['count']}")
        if stats['metadata']:
            print(f"  Metadata: {stats['metadata']}")


def inspect_collection(collection_name: str):
    """Inspect a specific collection."""
    print(f"=" * 80)
    print(f" Collection: {collection_name}")
    print("=" * 80)

    try:
        from src.indexing.vector_store import get_or_create_collection

        collection = get_or_create_collection(collection_name)

        # Get basic stats
        count = collection.count()
        print(f"\nTotal items: {count}")

        if count > 0:
            # Get a sample
            results = collection.peek(10)

            if results and 'metadatas' in results:
                print("\nSample metadata fields:")
                sample_meta = results['metadatas'][0] if results['metadatas'] else {}
                for key in sample_meta.keys():
                    print(f"  - {key}")

            # Count by source type
            print("\nBreakdown by source:")
            # Note: This is a simplified count, full stats would require querying all items
            print("  (Run full query for detailed breakdown)")

    except Exception as e:
        print(f"Error: {e}")


def delete_collection_interactive(collection_name: str):
    """Delete a collection with confirmation."""
    print(f"\n⚠️  WARNING: This will permanently delete collection '{collection_name}'")

    try:
        stats = get_collection_stats(collection_name)
        print(f"This collection contains {stats['count']} items.")
    except Exception as e:
        print(f"Collection may not exist: {e}")
        return

    response = input(f"\nType '{collection_name}' to confirm deletion: ").strip()

    if response == collection_name:
        delete_collection(collection_name)
        print(f"✓ Collection '{collection_name}' deleted")
    else:
        print("Deletion cancelled")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Manage ChromaDB collections"
    )
    parser.add_argument(
        "action",
        choices=["list", "inspect", "delete", "stats"],
        help="Action to perform"
    )
    parser.add_argument(
        "--collection",
        "-c",
        type=str,
        help="Collection name (required for inspect/delete)"
    )

    args = parser.parse_args()

    if args.action == "list":
        show_collections()

    elif args.action == "stats":
        if args.collection:
            inspect_collection(args.collection)
        else:
            show_collections()

    elif args.action == "inspect":
        if not args.collection:
            print("Error: --collection is required for inspect")
            return
        inspect_collection(args.collection)

    elif args.action == "delete":
        if not args.collection:
            print("Error: --collection is required for delete")
            return
        delete_collection_interactive(args.collection)


if __name__ == "__main__":
    main()

