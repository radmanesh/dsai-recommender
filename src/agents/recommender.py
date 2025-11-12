"""Recommendation Agent - Ranks faculty and generates explanations."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.agents.faculty_retriever import FacultyMatch
from src.agents.proposal_analyzer import ProposalAnalysis
from src.models.llm import get_llm, get_llm_with_params


@dataclass
class FacultyRecommendation:
    """A faculty recommendation with explanation."""

    rank: int
    faculty_name: str
    score: float
    explanation: str
    research_areas: List[str]
    contact_info: Dict[str, str]
    supporting_evidence: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "rank": self.rank,
            "faculty_name": self.faculty_name,
            "score": self.score,
            "explanation": self.explanation,
            "research_areas": self.research_areas,
            "contact_info": self.contact_info,
            "supporting_evidence": self.supporting_evidence,
        }


class RecommendationAgent:
    """Agent for ranking faculty and generating explanations."""

    def __init__(self, llm=None):
        """
        Initialize RecommendationAgent.

        Args:
            llm: LLM instance. If None, uses default from config.
        """
        self.llm = llm or get_llm_with_params(temperature=0.3, max_tokens=1024)

    def generate_recommendations(
        self,
        matches: List[FacultyMatch],
        analysis: ProposalAnalysis,
        top_n: int = 5,
    ) -> List[FacultyRecommendation]:
        """
        Generate ranked recommendations with explanations.

        Args:
            matches: List of faculty matches from retrieval.
            analysis: Original proposal analysis.
            top_n: Number of top recommendations to generate.

        Returns:
            List[FacultyRecommendation]: Ranked recommendations with explanations.
        """
        print(f"Generating top {top_n} recommendations...")

        # Take top matches
        top_matches = matches[:top_n]

        recommendations = []

        for i, match in enumerate(top_matches, 1):
            print(f"  Generating recommendation {i}/{len(top_matches)}...")

            # Generate explanation for this match
            explanation = self._generate_explanation(match, analysis)

            # Extract research areas
            research_areas = self._extract_research_areas(match)

            # Extract contact info
            contact_info = self._extract_contact_info(match)

            # Create supporting evidence summary
            supporting_evidence = self._create_supporting_evidence(match)

            recommendation = FacultyRecommendation(
                rank=i,
                faculty_name=match.faculty_name or "Unknown Faculty",
                score=match.score,
                explanation=explanation,
                research_areas=research_areas,
                contact_info=contact_info,
                supporting_evidence=supporting_evidence,
            )

            recommendations.append(recommendation)

        print(f"✓ Generated {len(recommendations)} recommendations")
        return recommendations

    def _generate_explanation(
        self,
        match: FacultyMatch,
        analysis: ProposalAnalysis
    ) -> str:
        """
        Generate a natural language explanation for why this faculty is recommended.
        Incorporates PDF support information if available.

        Args:
            match: The faculty match.
            analysis: The proposal analysis.

        Returns:
            str: Explanation text.
        """
        # Build faculty information string
        faculty_info = f"Name: {match.faculty_name or 'Faculty Member'}\n{match.text[:500]}"

        # Add PDF support information if available
        if match.pdf_support:
            faculty_info += "\n\nAdditional Information from Documents:"
            for pdf_info in match.pdf_support:
                if pdf_info.get('summary'):
                    summary = pdf_info['summary'][:200]
                    faculty_info += f"\n- {summary}"
                if pdf_info.get('research_interests'):
                    faculty_info += f"\n- Research: {pdf_info['research_interests']}"

        prompt = f"""You are an expert academic advisor. Explain why this faculty member is a good match for the given PhD proposal.

PROPOSAL SUMMARY:
Domain: {analysis.domain}
Topics: {', '.join(analysis.topics)}
Methods: {', '.join(analysis.methods)}
Application Areas: {', '.join(analysis.application_areas)}

FACULTY INFORMATION:
{faculty_info}

Provide a clear, concise explanation (2-3 sentences) of why this faculty member is recommended for supervising or collaborating on this proposal. Focus on specific overlaps in research interests, methods, or domains.

Explanation:"""

        response = self.llm.complete(prompt)
        explanation = str(response).strip()

        return explanation

    def _extract_research_areas(self, match: FacultyMatch) -> List[str]:
        """
        Extract research areas from faculty match metadata.

        Args:
            match: Faculty match object.

        Returns:
            List[str]: Research areas.
        """
        areas = []

        # Check metadata fields
        if 'areas' in match.metadata:
            areas_str = match.metadata['areas']
            areas = [a.strip() for a in areas_str.split(',')]

        if 'research_interests' in match.metadata and not areas:
            interests = match.metadata['research_interests']
            # Take first few phrases
            areas = [i.strip() for i in interests.split(',')][:3]

        return areas

    def _extract_contact_info(self, match: FacultyMatch) -> Dict[str, str]:
        """
        Extract contact information from faculty match metadata.

        Args:
            match: Faculty match object.

        Returns:
            Dict[str, str]: Contact information.
        """
        contact = {}

        if 'email' in match.metadata:
            contact['email'] = match.metadata['email']

        if 'website' in match.metadata:
            contact['website'] = match.metadata['website']

        if 'department' in match.metadata:
            contact['department'] = match.metadata['department']

        return contact

    def _create_supporting_evidence(self, match: FacultyMatch) -> str:
        """
        Create a summary of supporting evidence for the match.
        Includes PDF support information if available.

        Args:
            match: Faculty match object.

        Returns:
            str: Supporting evidence summary.
        """
        evidence_parts = []

        # Add CSV faculty profile text (truncated)
        csv_text = match.text[:300]
        if len(match.text) > 300:
            csv_text += "..."
        evidence_parts.append(f"Profile: {csv_text}")

        # Add PDF support information if available
        if match.pdf_support:
            evidence_parts.append(f"\nSupporting Documents ({len(match.pdf_support)} PDFs):")
            for i, pdf_info in enumerate(match.pdf_support, 1):
                if pdf_info.get('summary'):
                    summary = pdf_info['summary']
                    if len(summary) > 150:
                        summary = summary[:150] + "..."
                    evidence_parts.append(f"  PDF {i}: {summary}")
                if pdf_info.get('research_interests'):
                    evidence_parts.append(f"  Research: {pdf_info['research_interests']}")

        return "\n".join(evidence_parts)

    def generate_email_draft(
        self,
        recommendation: FacultyRecommendation,
        analysis: ProposalAnalysis,
        sender_name: str = "PhD Candidate"
    ) -> str:
        """
        Generate an email draft for contacting the recommended faculty.

        Args:
            recommendation: The faculty recommendation.
            analysis: The proposal analysis.
            sender_name: Name of the person sending the email.

        Returns:
            str: Email draft.
        """
        prompt = f"""Generate a professional email draft for a PhD candidate reaching out to a potential supervisor.

CANDIDATE: {sender_name}

FACULTY: {recommendation.faculty_name}

PROPOSAL SUMMARY:
{analysis.summary}

Research Domain: {analysis.domain}
Key Topics: {', '.join(analysis.topics[:3])}

WHY THIS FACULTY:
{recommendation.explanation}

Write a concise, professional email (150-200 words) that:
1. Introduces the candidate and their research interest
2. Explains why they're reaching out to this specific faculty member
3. Requests a meeting or discussion
4. Is respectful of the faculty member's time

Email Draft:"""

        response = self.llm.complete(prompt)
        email_draft = str(response).strip()

        return email_draft

    def generate_summary_report(
        self,
        recommendations: List[FacultyRecommendation],
        analysis: ProposalAnalysis
    ) -> str:
        """
        Generate a summary report of all recommendations.

        Args:
            recommendations: List of recommendations.
            analysis: The proposal analysis.

        Returns:
            str: Summary report.
        """
        report_lines = [
            "=" * 80,
            "FACULTY RECOMMENDATION REPORT",
            "=" * 80,
            "",
            "PROPOSAL SUMMARY:",
            f"Domain: {analysis.domain}",
            f"Topics: {', '.join(analysis.topics)}",
            f"Methods: {', '.join(analysis.methods)}",
            "",
            "=" * 80,
            "TOP RECOMMENDATIONS:",
            "=" * 80,
            "",
        ]

        for rec in recommendations:
            report_lines.extend([
                f"{rec.rank}. {rec.faculty_name} (Match Score: {rec.score:.3f})",
                f"   Research Areas: {', '.join(rec.research_areas) if rec.research_areas else 'N/A'}",
                f"   {rec.explanation}",
                "",
            ])

            if rec.contact_info:
                if 'email' in rec.contact_info:
                    report_lines.append(f"   Email: {rec.contact_info['email']}")
                if 'website' in rec.contact_info:
                    report_lines.append(f"   Website: {rec.contact_info['website']}")
                report_lines.append("")

        report_lines.extend([
            "=" * 80,
            "END OF REPORT",
            "=" * 80,
        ])

        return "\n".join(report_lines)

