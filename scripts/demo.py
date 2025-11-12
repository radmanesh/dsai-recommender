"""Complete system demonstration script."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import nest_asyncio
from src.agents.orchestrator import ResearchMatchOrchestrator
from src.indexing.vector_store import get_collection_stats
from src.utils.config import Config

nest_asyncio.apply()


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def demo_text_matching():
    """Demo: Match a text-based proposal."""
    print_section("DEMO 1: Text-Based Proposal Matching")

    # Sample proposal text
    sample_text = """
    Research Proposal: Multi-Agent Reinforcement Learning for Cooperative AI

    This PhD research focuses on developing novel algorithms for multi-agent
    reinforcement learning (MARL) in cooperative settings. The work will explore
    how multiple AI agents can learn to coordinate effectively through game-theoretic
    principles and emergent communication protocols.

    Key areas: Multi-agent systems, reinforcement learning, game theory, cooperative AI,
    emergent behaviors, RLHF for agent alignment.

    The research will contribute new coordination mechanisms and demonstrate applications
    in collaborative robotics and LLM-based agent teams.
    """

    # Initialize orchestrator
    orchestrator = ResearchMatchOrchestrator(top_n_recommendations=3)

    # Run matching
    result = orchestrator.match_proposal_text(sample_text, generate_report=True)

    # Show email draft for top recommendation
    if result.recommendations:
        print("\n" + "-" * 80)
        print("SAMPLE EMAIL DRAFT (for top recommendation)")
        print("-" * 80 + "\n")

        email = orchestrator.generate_email_for_recommendation(
            recommendation=result.recommendations[0],
            analysis=result.proposal_analysis,
            sender_name="Jane Smith",
        )
        print(email)


def demo_quick_search():
    """Demo: Quick search functionality."""
    print_section("DEMO 2: Quick Search")

    orchestrator = ResearchMatchOrchestrator(top_n_recommendations=5)

    queries = [
        "natural language processing and semantic search",
        "code generation with large language models",
        "explainable AI and fairness",
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 80)

        recommendations = orchestrator.quick_search(query, top_n=3)

        for rec in recommendations:
            print(f"\n{rec.rank}. {rec.faculty_name} (Score: {rec.score:.3f})")
            print(f"   {rec.explanation}")


def demo_pdf_matching():
    """Demo: Match a PDF proposal (if available)."""
    print_section("DEMO 3: PDF Proposal Matching")

    # Check if sample PDF exists
    pdf_path = Config.DATA_DIR / "sample_proposal.pdf"

    if not pdf_path.exists():
        print(f"⚠️  No PDF found at: {pdf_path}")
        print("This demo would analyze a PDF proposal and generate recommendations.")
        print("\nTo try this feature:")
        print("1. Place a PDF proposal in: data/sample_proposal.pdf")
        print("2. Run this demo again")
        return

    # Run PDF matching
    orchestrator = ResearchMatchOrchestrator(top_n_recommendations=3)
    result = orchestrator.match_proposal_pdf(pdf_path, generate_report=True)

    print(f"\n✓ Successfully matched PDF proposal with {len(result.recommendations)} recommendations")


def demo_system_status():
    """Show system status and configuration."""
    print_section("SYSTEM STATUS")

    # Configuration
    print("Configuration:")
    config = Config.get_config_summary()
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Collection stats
    print("\nCollection Status:")
    try:
        stats = get_collection_stats()
        print(f"  Collection: {stats['name']}")
        print(f"  Items: {stats['count']}")

        if stats['count'] == 0:
            print("\n  ⚠️  Collection is empty!")
            print("  Run 'python scripts/ingest.py' to index data")
        else:
            print(f"\n  ✓ System ready with {stats['count']} indexed items")

    except Exception as e:
        print(f"  ❌ Error: {e}")


def main():
    """Main demonstration."""
    print("\n" + "=" * 80)
    print(" AGENTIC RESEARCH MATCHMAKER - SYSTEM DEMONSTRATION")
    print("=" * 80)

    # Check system status first
    demo_system_status()

    try:
        stats = get_collection_stats()
        if stats['count'] == 0:
            print("\n⚠️  Cannot run demos - collection is empty")
            print("Please run 'python scripts/ingest.py' first")
            return
    except Exception as e:
        print(f"\n❌ Error accessing collection: {e}")
        return

    # Run demos
    try:
        demo_text_matching()
        demo_quick_search()
        demo_pdf_matching()

        print_section("DEMONSTRATION COMPLETE")
        print("✓ All demos executed successfully")
        print("\nNext steps:")
        print("  - Try interactive queries: python scripts/query_demo.py --interactive")
        print("  - Add your own PDFs to data/pdfs/")
        print("  - Update faculty data in data/DSAI-Faculties.csv")
        print("  - Re-run ingestion to update the index")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

