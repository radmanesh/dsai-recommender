"""Query demonstration script - test the system with example queries."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import nest_asyncio
from src.indexing.index_builder import IndexManager
from src.indexing.vector_store import get_collection_stats
from src.utils.config import Config

# Apply nest_asyncio for compatibility
nest_asyncio.apply()


def print_separator():
    """Print a visual separator."""
    print("\n" + "=" * 80 + "\n")


def display_response(response):
    """Display query response in a formatted way."""
    print("\n📝 Response:")
    print("-" * 80)
    print(response)
    print("-" * 80)

    # Display source nodes if available
    if hasattr(response, 'source_nodes') and response.source_nodes:
        print(f"\n📚 Retrieved {len(response.source_nodes)} source(s):")
        for i, node in enumerate(response.source_nodes[:3], 1):  # Show top 3
            print(f"\n  Source {i}:")
            print(f"  Score: {node.score:.4f}" if hasattr(node, 'score') else "")
            print(f"  Type: {node.metadata.get('type', 'unknown')}")
            if 'name' in node.metadata:
                print(f"  Faculty: {node.metadata['name']}")
            print(f"  Text preview: {node.text[:150]}...")


def display_retrieved_nodes(nodes):
    """Display retrieved nodes in a formatted way."""
    print(f"\n📚 Retrieved {len(nodes)} node(s):")
    print("-" * 80)

    for i, node in enumerate(nodes, 1):
        print(f"\nNode {i}:")
        print(f"  Score: {node.score:.4f}" if hasattr(node, 'score') else "")
        print(f"  Type: {node.metadata.get('type', 'unknown')}")
        print(f"  Source: {node.metadata.get('source', 'unknown')}")

        if 'name' in node.metadata:
            print(f"  Faculty: {node.metadata['name']}")
        if 'faculty_name' in node.metadata:
            print(f"  Faculty: {node.metadata['faculty_name']}")

        print(f"  Text: {node.text[:200]}...")
        print()


def run_example_queries():
    """Run a set of example queries."""
    print("Faculty Research Matchmaker - Query Demo")
    print_separator()

    # Check collection stats
    print("Checking collection status...")
    try:
        stats = get_collection_stats()
        print(f"✓ Collection: {stats['name']}")
        print(f"✓ Items in collection: {stats['count']}")

        if stats['count'] == 0:
            print("\n⚠️  Collection is empty!")
            print("Please run the ingestion script first:")
            print("  python scripts/ingest.py")
            return

    except Exception as e:
        print(f"❌ Error accessing collection: {e}")
        return

    print_separator()

    # Initialize IndexManager
    print("Initializing index and query engine...")
    manager = IndexManager()
    print("✓ Ready to query")

    # Example queries
    example_queries = [
        {
            "title": "Multi-Agent AI Systems",
            "query": (
                "Which faculty members specialize in multi-agent systems, "
                "LLM-based coding assistants, or AI agent orchestration?"
            ),
        },
        {
            "title": "Reinforcement Learning",
            "query": (
                "Find professors with expertise in reinforcement learning, "
                "especially for LLM post-training and RLHF."
            ),
        },
        {
            "title": "Natural Language Processing",
            "query": (
                "Who works on natural language processing, semantic search, "
                "or information retrieval?"
            ),
        },
    ]

    for i, example in enumerate(example_queries, 1):
        print_separator()
        print(f"Example Query {i}: {example['title']}")
        print("-" * 80)
        print(f"Query: {example['query']}")

        try:
            # Query with LLM synthesis
            print("\n🔍 Querying with LLM synthesis...")
            response = manager.query(example['query'])
            display_response(response)

        except Exception as e:
            print(f"❌ Error during query: {e}")
            import traceback
            traceback.print_exc()

    print_separator()


def interactive_mode():
    """Run in interactive mode, allowing user to input queries."""
    print("Faculty Research Matchmaker - Interactive Query Mode")
    print_separator()

    # Check collection
    try:
        stats = get_collection_stats()
        print(f"Collection: {stats['name']} ({stats['count']} items)")

        if stats['count'] == 0:
            print("\n⚠️  Collection is empty!")
            print("Please run the ingestion script first.")
            return

    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Initialize manager
    print("\nInitializing query engine...")
    manager = IndexManager()
    print("✓ Ready")

    print("\nEnter your queries (type 'quit' or 'exit' to stop)")
    print("Type 'retrieval' to switch to retrieval-only mode")
    print("Type 'query' to switch back to query mode with LLM")
    print_separator()

    mode = "query"  # "query" or "retrieval"

    while True:
        try:
            user_input = input(f"\n[{mode}] > ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if user_input.lower() == 'retrieval':
                mode = "retrieval"
                print("Switched to retrieval-only mode")
                continue

            if user_input.lower() == 'query':
                mode = "query"
                print("Switched to query mode with LLM")
                continue

            # Process query
            if mode == "query":
                print("\n🔍 Querying...")
                response = manager.query(user_input)
                display_response(response)
            else:
                print("\n🔍 Retrieving...")
                nodes = manager.retrieve(user_input)
                display_retrieved_nodes(nodes)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Query the Faculty Research Matchmaker system"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Run a single query"
    )

    args = parser.parse_args()

    if args.query:
        # Single query mode
        print("Faculty Research Matchmaker - Single Query")
        print_separator()

        stats = get_collection_stats()
        if stats['count'] == 0:
            print("⚠️  Collection is empty! Run ingestion first.")
            return

        manager = IndexManager()
        print(f"Query: {args.query}\n")

        response = manager.query(args.query)
        display_response(response)

    elif args.interactive:
        # Interactive mode
        interactive_mode()

    else:
        # Default: run examples
        run_example_queries()


if __name__ == "__main__":
    main()

