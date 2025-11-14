#!/usr/bin/env python
"""
Simple command-line interface for the Faculty Matchmaker.

Usage:
    python match.py "Your proposal text here..."
    python match.py --file proposal.pdf
    python match.py --query "machine learning"
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import nest_asyncio
from src.agents.orchestrator import ResearchMatchOrchestrator
from src.indexing.vector_store import get_collection_stats

nest_asyncio.apply()


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Match PhD proposals to faculty members",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Match text proposal:
    python match.py "Research on multi-agent AI systems..."

  Match PDF proposal:
    python match.py --file proposal.pdf

  Quick search:
    python match.py --query "machine learning and NLP"

  Generate email for top match:
    python match.py --file proposal.pdf --email "John Doe"
        """
    )

    parser.add_argument(
        "text",
        nargs="?",
        help="Proposal text (if not using --file or --query)"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Path to proposal PDF file"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Quick search query"
    )
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=5,
        help="Number of recommendations (default: 5)"
    )
    parser.add_argument(
        "--email", "-e",
        type=str,
        help="Generate email draft for top recommendation (provide sender name)"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating full report"
    )

    args = parser.parse_args()

    # Check collection
    try:
        stats = get_collection_stats()
        if stats['count'] == 0:
            print("❌ Error: Collection is empty!")
            print("Please run 'python scripts/ingest.py' first to index data.")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error accessing collection: {e}")
        print("Please run 'python scripts/ingest.py' first.")
        sys.exit(1)

    # Initialize orchestrator
    orchestrator = ResearchMatchOrchestrator(top_n_recommendations=args.top)

    # Determine mode and execute
    try:
        if args.file:
            # PDF mode
            pdf_path = Path(args.file)
            if not pdf_path.exists():
                print(f"❌ Error: File not found: {args.file}")
                sys.exit(1)

            result = orchestrator.match_proposal_pdf(
                pdf_path,
                generate_report=not args.no_report
            )

            # Generate email if requested
            if args.email and result.recommendations:
                print("\n" + "=" * 80)
                print(" EMAIL DRAFT")
                print("=" * 80 + "\n")
                email = orchestrator.generate_email_for_recommendation(
                    result.recommendations[0],
                    result.proposal_analysis,
                    sender_name=args.email
                )
                print(email)

        elif args.query:
            # Quick search mode
            print(f"\n🔍 Quick Search: {args.query}")
            print("=" * 80 + "\n")

            recommendations = orchestrator.quick_search(args.query, top_n=args.top)

            for rec in recommendations:
                print(f"{rec.rank}. {rec.faculty_name} (Score: {rec.score:.3f})")
                print(f"   {rec.explanation}")
                if rec.contact_info.get('email'):
                    print(f"   Email: {rec.contact_info['email']}")
                print()

        elif args.text:
            # Text mode
            result = orchestrator.match_proposal_text(
                args.text,
                generate_report=not args.no_report
            )

            # Generate email if requested
            if args.email and result.recommendations:
                print("\n" + "=" * 80)
                print(" EMAIL DRAFT")
                print("=" * 80 + "\n")
                email = orchestrator.generate_email_for_recommendation(
                    result.recommendations[0],
                    result.proposal_analysis,
                    sender_name=args.email
                )
                print(email)

        else:
            parser.print_help()
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

