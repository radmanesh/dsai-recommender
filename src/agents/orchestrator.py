"""Orchestrator - Coordinates the multi-agent workflow for faculty matching."""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from src.agents.proposal_analyzer import ProposalAnalyzer, ProposalAnalysis
from src.agents.faculty_retriever import FacultyRetriever, FacultyMatch
from src.agents.recommender import RecommendationAgent, FacultyRecommendation


@dataclass
class MatchingResult:
    """Complete result from the matching workflow."""

    proposal_analysis: ProposalAnalysis
    faculty_matches: List[FacultyMatch]
    recommendations: List[FacultyRecommendation]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "proposal_analysis": self.proposal_analysis.to_dict(),
            "faculty_matches": [m.to_dict() for m in self.faculty_matches],
            "recommendations": [r.to_dict() for r in self.recommendations],
        }


class ResearchMatchOrchestrator:
    """
    Main orchestrator for the faculty matching system.

    Coordinates the workflow: proposal analysis → faculty retrieval → recommendations.
    """

    def __init__(
        self,
        top_k_retrieval: int = None,
        top_n_recommendations: int = 5,
        collection_name: str = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            top_k_retrieval: Number of faculty to retrieve. If None, uses config default.
            top_n_recommendations: Number of final recommendations to generate.
            collection_name: ChromaDB collection name. If None, uses config default.
        """
        self.top_n_recommendations = top_n_recommendations

        # Initialize agents
        self.analyzer = ProposalAnalyzer()
        self.retriever = FacultyRetriever(
            top_k=top_k_retrieval,
            collection_name=collection_name
        )
        self.recommender = RecommendationAgent()

        print("✓ Research Match Orchestrator initialized")

    def match_proposal_pdf(
        self,
        pdf_path: Union[str, Path],
        generate_report: bool = True,
    ) -> MatchingResult:
        """
        Complete workflow: analyze PDF proposal and generate faculty recommendations.

        Args:
            pdf_path: Path to the proposal PDF.
            generate_report: Whether to print a summary report.

        Returns:
            MatchingResult: Complete matching results.
        """
        print("\n" + "=" * 80)
        print("FACULTY MATCHING WORKFLOW")
        print("=" * 80)

        # Step 1: Analyze proposal
        print("\n[Step 1/3] Analyzing proposal PDF...")
        analysis = self.analyzer.analyze_pdf(pdf_path)
        print(f"✓ Analysis complete")
        print(f"  Domain: {analysis.domain}")
        print(f"  Topics: {', '.join(analysis.topics[:3])}")
        print(f"  Methods: {', '.join(analysis.methods[:3])}")

        # Step 2: Retrieve faculty
        print(f"\n[Step 2/3] Retrieving matching faculty...")
        matches = self.retriever.retrieve_from_analysis(analysis)
        print(f"✓ Retrieved {len(matches)} faculty matches")

        # Step 3: Generate recommendations
        print(f"\n[Step 3/3] Generating recommendations...")
        recommendations = self.recommender.generate_recommendations(
            matches=matches,
            analysis=analysis,
            top_n=self.top_n_recommendations,
        )
        print(f"✓ Generated {len(recommendations)} recommendations")

        # Create result
        result = MatchingResult(
            proposal_analysis=analysis,
            faculty_matches=matches,
            recommendations=recommendations,
        )

        # Generate report if requested
        if generate_report:
            print("\n" + "=" * 80)
            report = self.recommender.generate_summary_report(recommendations, analysis)
            print(report)

        return result

    def match_proposal_text(
        self,
        text: str,
        generate_report: bool = True,
    ) -> MatchingResult:
        """
        Complete workflow: analyze proposal text and generate faculty recommendations.

        Args:
            text: The proposal text.
            generate_report: Whether to print a summary report.

        Returns:
            MatchingResult: Complete matching results.
        """
        print("\n" + "=" * 80)
        print("FACULTY MATCHING WORKFLOW")
        print("=" * 80)

        # Step 1: Analyze proposal
        print("\n[Step 1/3] Analyzing proposal text...")
        analysis = self.analyzer.analyze_text(text)
        print(f"✓ Analysis complete")
        print(f"  Domain: {analysis.domain}")
        print(f"  Topics: {', '.join(analysis.topics[:3])}")

        # Step 2: Retrieve faculty
        print(f"\n[Step 2/3] Retrieving matching faculty...")
        matches = self.retriever.retrieve_from_analysis(analysis)
        print(f"✓ Retrieved {len(matches)} faculty matches")

        # Step 3: Generate recommendations
        print(f"\n[Step 3/3] Generating recommendations...")
        recommendations = self.recommender.generate_recommendations(
            matches=matches,
            analysis=analysis,
            top_n=self.top_n_recommendations,
        )
        print(f"✓ Generated {len(recommendations)} recommendations")

        # Create result
        result = MatchingResult(
            proposal_analysis=analysis,
            faculty_matches=matches,
            recommendations=recommendations,
        )

        # Generate report if requested
        if generate_report:
            print("\n" + "=" * 80)
            report = self.recommender.generate_summary_report(recommendations, analysis)
            print(report)

        return result

    def quick_search(
        self,
        query: str,
        top_n: int = None,
    ) -> List[FacultyRecommendation]:
        """
        Quick search without full proposal analysis.

        Args:
            query: Search query string.
            top_n: Number of recommendations. If None, uses self.top_n_recommendations.

        Returns:
            List[FacultyRecommendation]: Recommendations.
        """
        n = top_n or self.top_n_recommendations

        print(f"\nQuick search: {query}")

        # Retrieve matches
        matches = self.retriever.retrieve_from_query(query, top_k=n * 2)

        # Create a simple analysis object for the recommender
        simple_analysis = ProposalAnalysis(
            topics=[],
            methods=[],
            domain="",
            application_areas=[],
            key_phrases=[],
            summary=query,
            full_text=query,
        )

        # Generate recommendations
        recommendations = self.recommender.generate_recommendations(
            matches=matches,
            analysis=simple_analysis,
            top_n=n,
        )

        return recommendations

    def generate_email_for_recommendation(
        self,
        recommendation: FacultyRecommendation,
        analysis: ProposalAnalysis,
        sender_name: str = "PhD Candidate",
    ) -> str:
        """
        Generate an email draft for contacting a recommended faculty member.

        Args:
            recommendation: The faculty recommendation.
            analysis: The proposal analysis.
            sender_name: Name of the sender.

        Returns:
            str: Email draft.
        """
        return self.recommender.generate_email_draft(
            recommendation=recommendation,
            analysis=analysis,
            sender_name=sender_name,
        )

    def compare_proposals(
        self,
        proposals: List[Union[str, Path]],
    ) -> Dict[str, MatchingResult]:
        """
        Compare multiple proposals and generate recommendations for each.

        Args:
            proposals: List of proposal file paths.

        Returns:
            Dict[str, MatchingResult]: Results keyed by proposal filename.
        """
        results = {}

        for i, proposal_path in enumerate(proposals, 1):
            print(f"\n{'=' * 80}")
            print(f"Processing proposal {i}/{len(proposals)}: {proposal_path}")
            print("=" * 80)

            result = self.match_proposal_pdf(
                pdf_path=proposal_path,
                generate_report=False,
            )

            key = Path(proposal_path).name
            results[key] = result

        # Print comparison summary
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)

        for proposal_name, result in results.items():
            print(f"\n{proposal_name}:")
            print(f"  Domain: {result.proposal_analysis.domain}")
            print(f"  Top recommendation: {result.recommendations[0].faculty_name}")

        return results

