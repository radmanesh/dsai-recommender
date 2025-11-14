"""Utility script to inspect nodes in the vector store."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import nest_asyncio
from src.indexing.index_builder import IndexManager
from src.indexing.vector_store import get_collection_stats

# Apply nest_asyncio for compatibility
nest_asyncio.apply()


def inspect_nodes(query: str = None, top_k: int = 10, show_full_text: bool = False):
    """
    Inspect nodes from the vector store.

    Args:
        query: Query string. If None, uses a broad query.
        top_k: Number of nodes to retrieve.
        show_full_text: If True, shows full text instead of preview.
    """
    # Check collection
    stats = get_collection_stats()
    print(f"Collection: {stats['name']} ({stats['count']} items)\n")

    if stats['count'] == 0:
        print("⚠️  Collection is empty!")
        print("Please run the ingestion script first:")
        print("  python scripts/ingest.py")
        return

    # Initialize manager
    print("Initializing index and retriever...")
    manager = IndexManager()
    print("✓ Ready\n")

    # Use broad query if none provided
    if query is None:
        query = "research"

    # Retrieve nodes
    print(f"Retrieving top {top_k} nodes for query: '{query}'\n")
    try:
        nodes = manager.retrieve(query)
    except Exception as e:
        print(f"❌ Error retrieving nodes: {e}")
        import traceback
        traceback.print_exc()
        return

    if not nodes:
        print("No nodes found for the query.")
        return

    # Display nodes
    num_to_show = min(top_k, len(nodes))
    print(f"Showing {num_to_show} of {len(nodes)} retrieved nodes:\n")

    for i, node in enumerate(nodes[:top_k], 1):
        print(f"{'='*80}")
        print(f"Node {i}/{num_to_show}")
        print(f"{'='*80}")
        print(f"ID: {node.node_id}")

        if hasattr(node, 'score'):
            print(f"Score: {node.score:.4f}")

        print(f"\nMetadata:")
        for key, value in node.metadata.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}...")
            elif isinstance(value, list):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")

        print(f"\nText:")
        if show_full_text:
            print(node.text)
        else:
            text_preview = node.text[:500] + "..." if len(node.text) > 500 else node.text
            print(text_preview)

        if hasattr(node, 'embedding') and node.embedding:
            print(f"\nEmbedding: {len(node.embedding)} dimensions")

        print()


def inspect_node_by_id(node_id: str):
    """
    Inspect a specific node by its ID.

    Args:
        node_id: The node ID to inspect.
    """
    # Check collection
    stats = get_collection_stats()
    print(f"Collection: {stats['name']} ({stats['count']} items)\n")

    if stats['count'] == 0:
        print("⚠️  Collection is empty!")
        return

    # Initialize manager
    print("Initializing index...")
    manager = IndexManager()

    # Try to retrieve the node
    # Note: This requires querying and filtering, or direct access to the vector store
    print(f"Searching for node ID: {node_id}\n")

    # Use a broad query and filter
    nodes = manager.retrieve("research")

    found_node = None
    for node in nodes:
        if node.node_id == node_id:
            found_node = node
            break

    if found_node:
        print(f"{'='*80}")
        print(f"Found Node: {node_id}")
        print(f"{'='*80}")
        print(f"ID: {found_node.node_id}")

        if hasattr(found_node, 'score'):
            print(f"Score: {found_node.score:.4f}")

        print(f"\nMetadata:")
        for key, value in found_node.metadata.items():
            print(f"  {key}: {value}")

        print(f"\nFull Text:")
        print(found_node.text)

        if hasattr(found_node, 'embedding') and found_node.embedding:
            print(f"\nEmbedding: {len(found_node.embedding)} dimensions")
    else:
        print(f"❌ Node with ID '{node_id}' not found in retrieved results.")
        print("Note: This searches through retrieved nodes. For direct access,")
        print("you may need to query the vector store directly.")


def list_node_summaries(query: str = None, top_k: int = 20):
    """
    List a summary of multiple nodes.

    Args:
        query: Query string. If None, uses a broad query.
        top_k: Number of nodes to retrieve.
    """
    # Check collection
    stats = get_collection_stats()
    print(f"Collection: {stats['name']} ({stats['count']} items)\n")

    if stats['count'] == 0:
        print("⚠️  Collection is empty!")
        return

    # Initialize manager
    manager = IndexManager()

    # Use broad query if none provided
    if query is None:
        query = "research"

    # Retrieve nodes
    print(f"Retrieving top {top_k} nodes for query: '{query}'\n")
    nodes = manager.retrieve(query)

    if not nodes:
        print("No nodes found.")
        return

    # Display summaries
    num_to_show = min(top_k, len(nodes))
    print(f"Node Summaries ({num_to_show} of {len(nodes)}):\n")

    for i, node in enumerate(nodes[:top_k], 1):
        print(f"{i}. [{node.node_id[:8]}...] ", end="")
        if hasattr(node, 'score'):
            print(f"Score: {node.score:.4f} | ", end="")
        print(f"Source: {node.metadata.get('source', 'N/A')} | ", end="")
        print(f"Type: {node.metadata.get('type', 'N/A')}")

        if 'faculty_name' in node.metadata:
            print(f"   Faculty: {node.metadata['faculty_name']}")
        elif 'name' in node.metadata:
            print(f"   Faculty: {node.metadata['name']}")

        text_preview = node.text[:100].replace('\n', ' ') + "..."
        print(f"   Text: {text_preview}")
        print()


def inspect_csv_only():
    """
    Inspect only CSV-derived entries in the collection.
    """
    from collections import defaultdict
    from src.indexing.vector_store import get_or_create_collection

    # Get collection
    collection = get_or_create_collection()

    # Check if empty
    count = collection.count()
    print(f"{'='*80}")
    print(f"CSV Data Inspection")
    print(f"{'='*80}")
    print(f"Total items in collection: {count}\n")

    if count == 0:
        print("⚠️  Collection is empty!")
        print("Please run the ingestion script first:")
        print("  python scripts/ingest.py")
        return

    # Get only CSV entries
    print("Fetching CSV entries...")
    results = collection.get(
        where={"type": "csv"},
        include=["documents", "metadatas"]
    )

    csv_count = len(results['ids'])
    print(f"Found {csv_count} CSV-derived entries\n")

    if csv_count == 0:
        print("⚠️  No CSV entries found in the collection.")
        print("Your data may have been ingested with a different type label.")
        return

    # Group by faculty
    faculty_data = defaultdict(list)

    for doc, meta in zip(results['documents'], results['metadatas']):
        faculty_name = meta.get('faculty_name', meta.get('name', 'Unknown'))
        faculty_data[faculty_name].append((doc, meta))

    # Display by faculty
    print(f"{'='*80}")
    print(f"CSV Data by Faculty ({len(faculty_data)} faculty members):")
    print(f"{'='*80}\n")

    for faculty_name in sorted(faculty_data.keys()):
        entries = faculty_data[faculty_name]
        print(f"\n{'▶'*40}")
        print(f"Faculty: {faculty_name}")
        print(f"CSV Entries: {len(entries)}")
        print(f"{'▶'*40}\n")

        for i, (doc, meta) in enumerate(entries, 1):
            print(f"  Entry {i}:")
            print(f"  ID: {results['ids'][results['documents'].index(doc)][:16]}...")
            print(f"  Source: {meta.get('source', 'N/A')}")

            # Show key metadata fields
            if 'department' in meta:
                print(f"  Department: {meta['department']}")
            if 'role' in meta:
                print(f"  Role: {meta['role']}")
            if 'research_areas' in meta:
                areas = meta['research_areas']
                if isinstance(areas, list):
                    print(f"  Research Areas: {', '.join(areas[:5])}")
                else:
                    print(f"  Research Areas: {areas}")
            if 'email' in meta:
                print(f"  Email: {meta['email']}")
            if 'website' in meta:
                print(f"  Website: {meta['website']}")

            # Show all other metadata
            other_fields = [k for k in meta.keys()
                          if k not in ['type', 'source', 'faculty_name', 'name',
                                     'department', 'role', 'research_areas', 'email', 'website']]
            if other_fields:
                print(f"  Other fields: {', '.join(other_fields)}")

            # Show content preview
            content_preview = doc[:200] + "..." if len(doc) > 200 else doc
            print(f"  Content: {content_preview}")
            print()

    # Summary
    print(f"{'='*80}")
    print(f"Summary:")
    print(f"{'='*80}")
    print(f"Total CSV entries: {csv_count}")
    print(f"Faculty members: {len(faculty_data)}")
    if faculty_data:
        avg_per_faculty = csv_count / len(faculty_data)
        print(f"Average entries per faculty: {avg_per_faculty:.1f}")

        # Show distribution
        max_entries = max(len(entries) for entries in faculty_data.values())
        min_entries = min(len(entries) for entries in faculty_data.values())
        print(f"Entry range: {min_entries}-{max_entries} per faculty")
    print()


def inspect_metadata_schema():
    """
    Inspect what metadata fields are stored in the collection
    and show statistics about data types.
    """
    from collections import Counter, defaultdict
    from src.indexing.vector_store import get_or_create_collection

    # Get collection
    collection = get_or_create_collection()

    # Check if empty
    count = collection.count()
    print(f"{'='*80}")
    print(f"Collection Metadata Schema Analysis")
    print(f"{'='*80}")
    print(f"Total items: {count}\n")

    if count == 0:
        print("⚠️  Collection is empty!")
        print("Please run the ingestion script first:")
        print("  python scripts/ingest.py")
        return

    # Get all items with metadata
    print("Fetching all items from collection...")
    results = collection.get(include=["metadatas", "documents"])

    # Analyze metadata keys
    all_keys = set()
    key_counts = Counter()
    field_values = defaultdict(set)
    type_counts = Counter()

    for meta in results['metadatas']:
        all_keys.update(meta.keys())
        for key in meta.keys():
            key_counts[key] += 1

            # Collect sample values (limit to avoid memory issues)
            value = meta[key]
            if isinstance(value, (str, int, float, bool)):
                if len(field_values[key]) < 10:  # Keep max 10 sample values
                    field_values[key].add(str(value))
            elif isinstance(value, list):
                if len(field_values[key]) < 10:
                    field_values[key].add(str(value[:3]))  # First 3 items

        # Count by type
        doc_type = meta.get('type', 'unknown')
        type_counts[doc_type] += 1

    # Display metadata fields
    print(f"\n{'='*80}")
    print(f"Metadata Fields Found ({len(all_keys)} unique fields):")
    print(f"{'='*80}")

    for key in sorted(all_keys):
        coverage = (key_counts[key] / count) * 100
        print(f"\n📌 {key}")
        print(f"   Present in: {key_counts[key]}/{count} items ({coverage:.1f}%)")

        if field_values[key]:
            sample_values = list(field_values[key])[:5]
            print(f"   Sample values:")
            for val in sample_values:
                # Truncate long values
                display_val = val[:60] + "..." if len(val) > 60 else val
                print(f"     - {display_val}")

    # Display type distribution
    print(f"\n{'='*80}")
    print(f"Data Types Distribution:")
    print(f"{'='*80}")

    for doc_type, type_count in type_counts.most_common():
        percentage = (type_count / count) * 100
        bar_length = int(percentage / 2)  # Scale to max 50 chars
        bar = "█" * bar_length
        print(f"{doc_type:15s} | {bar} {type_count:4d} ({percentage:5.1f}%)")

    # Check for CSV-specific entries
    csv_count = type_counts.get('csv', 0)
    pdf_count = type_counts.get('pdf', 0)

    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"{'='*80}")
    print(f"Total items:        {count}")
    print(f"CSV-derived items:  {csv_count}")
    print(f"PDF-derived items:  {pdf_count}")
    print(f"Other items:        {count - csv_count - pdf_count}")
    print(f"Unique fields:      {len(all_keys)}")

    # Show faculty names if available
    faculty_names = set()
    for meta in results['metadatas']:
        if 'faculty_name' in meta:
            faculty_names.add(meta['faculty_name'])
        elif 'name' in meta:
            faculty_names.add(meta['name'])

    if faculty_names:
        print(f"\n{'='*80}")
        print(f"Faculty Members in Collection ({len(faculty_names)}):")
        print(f"{'='*80}")
        for name in sorted(faculty_names):
            # Count items per faculty
            faculty_item_count = sum(
                1 for meta in results['metadatas']
                if meta.get('faculty_name') == name or meta.get('name') == name
            )
            print(f"  • {name}: {faculty_item_count} items")

    print()


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect nodes in the vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect top 10 nodes for a query
  python scripts/inspect_nodes.py -q "machine learning" -k 10

  # Show full text of nodes
  python scripts/inspect_nodes.py -q "deep learning" -f

  # List summaries of many nodes
  python scripts/inspect_nodes.py --list -k 50

  # Inspect a specific node by ID
  python scripts/inspect_nodes.py --id <node_id>

  # Show metadata schema and statistics
  python scripts/inspect_nodes.py --schema

  # Show only CSV-derived entries
  python scripts/inspect_nodes.py --csv
        """
    )

    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Query string (default: 'research')"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=10,
        help="Number of nodes to show (default: 10)"
    )
    parser.add_argument(
        "--full-text", "-f",
        action="store_true",
        help="Show full text instead of preview"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List summaries of multiple nodes"
    )
    parser.add_argument(
        "--id",
        type=str,
        help="Inspect a specific node by ID"
    )
    parser.add_argument(
        "--schema", "-s",
        action="store_true",
        help="Show metadata schema and collection statistics"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Show only CSV-derived entries"
    )

    args = parser.parse_args()

    if args.csv:
        inspect_csv_only()
    elif args.schema:
        inspect_metadata_schema()
    elif args.id:
        inspect_node_by_id(args.id)
    elif args.list:
        list_node_summaries(args.query, args.top_k)
    else:
        inspect_nodes(args.query, args.top_k, args.full_text)


if __name__ == "__main__":
    main()

