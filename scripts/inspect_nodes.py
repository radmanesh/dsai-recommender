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

    args = parser.parse_args()

    if args.id:
        inspect_node_by_id(args.id)
    elif args.list:
        list_node_summaries(args.query, args.top_k)
    else:
        inspect_nodes(args.query, args.top_k, args.full_text)


if __name__ == "__main__":
    main()

