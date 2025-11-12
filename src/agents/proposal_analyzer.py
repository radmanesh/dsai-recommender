"""Proposal Analysis Agent - Extracts key information from PhD proposals."""

from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from src.ingestion.pdf_loader import load_single_pdf
from src.models.llm import get_llm


@dataclass
class ProposalAnalysis:
    """Structured representation of a proposal analysis."""

    topics: List[str]
    methods: List[str]
    domain: str
    application_areas: List[str]
    key_phrases: List[str]
    summary: str
    full_text: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "topics": self.topics,
            "methods": self.methods,
            "domain": self.domain,
            "application_areas": self.application_areas,
            "key_phrases": self.key_phrases,
            "summary": self.summary,
        }

    def to_search_query(self) -> str:
        """
        Convert analysis to a search query string.

        Returns:
            str: Formatted search query.
        """
        query_parts = []

        if self.domain:
            query_parts.append(f"Domain: {self.domain}")

        if self.topics:
            query_parts.append(f"Topics: {', '.join(self.topics)}")

        if self.methods:
            query_parts.append(f"Methods: {', '.join(self.methods)}")

        if self.application_areas:
            query_parts.append(f"Applications: {', '.join(self.application_areas)}")

        return ". ".join(query_parts)


class ProposalAnalyzer:
    """Agent for analyzing PhD proposals and extracting key information."""

    def __init__(self, llm=None):
        """
        Initialize ProposalAnalyzer.

        Args:
            llm: LLM instance. If None, uses default from config.
        """
        self.llm = llm or get_llm()

    def analyze_pdf(self, pdf_path: Union[str, Path]) -> ProposalAnalysis:
        """
        Analyze a proposal PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ProposalAnalysis: Structured analysis of the proposal.
        """
        print(f"Analyzing proposal: {pdf_path}")

        # Load PDF
        documents = load_single_pdf(str(pdf_path))

        if not documents:
            raise ValueError(f"Could not load PDF: {pdf_path}")

        # Combine all pages
        full_text = "\n\n".join([doc.text for doc in documents])

        # Analyze with LLM
        analysis = self._analyze_text(full_text)
        analysis.full_text = full_text

        return analysis

    def analyze_text(self, text: str) -> ProposalAnalysis:
        """
        Analyze proposal text directly.

        Args:
            text: The proposal text.

        Returns:
            ProposalAnalysis: Structured analysis of the proposal.
        """
        print("Analyzing proposal text...")
        analysis = self._analyze_text(text)
        analysis.full_text = text
        return analysis

    def _analyze_text(self, text: str) -> ProposalAnalysis:
        """
        Internal method to analyze text with LLM.

        Args:
            text: The text to analyze.

        Returns:
            ProposalAnalysis: Structured analysis.
        """
        # Truncate text if too long (keep first portion which usually has the key info)
        max_chars = 8000
        if len(text) > max_chars:
            text_sample = text[:max_chars] + "\n\n[... text truncated ...]"
        else:
            text_sample = text

        # Create analysis prompt
        prompt = self._create_analysis_prompt(text_sample)

        # Query LLM
        response = self.llm.complete(prompt)
        response_text = str(response)

        # Parse response
        analysis = self._parse_llm_response(response_text)

        return analysis

    def _create_analysis_prompt(self, text: str) -> str:
        """
        Create the analysis prompt for the LLM.

        Args:
            text: Proposal text.

        Returns:
            str: Formatted prompt.
        """
        prompt = f"""You are an expert academic research analyzer. Analyze the following PhD proposal and extract key information.

PROPOSAL TEXT:
{text}

Please analyze this proposal and provide:

1. TOPICS: List 3-5 main research topics or keywords (comma-separated)
2. METHODS: List 2-4 key methods or techniques mentioned (comma-separated)
3. DOMAIN: Identify the primary research domain (single phrase)
4. APPLICATION_AREAS: List 2-3 potential application areas (comma-separated)
5. KEY_PHRASES: List 3-5 distinctive phrases or terms that characterize this work (comma-separated)
6. SUMMARY: Write a 2-3 sentence summary of the proposal

Format your response EXACTLY as follows:

TOPICS: [topic1, topic2, topic3, ...]
METHODS: [method1, method2, method3, ...]
DOMAIN: [domain]
APPLICATION_AREAS: [area1, area2, area3, ...]
KEY_PHRASES: [phrase1, phrase2, phrase3, ...]
SUMMARY: [your summary here]

Be specific and use technical terminology where appropriate."""

        return prompt

    def _parse_llm_response(self, response_text: str) -> ProposalAnalysis:
        """
        Parse the LLM response into a ProposalAnalysis object.

        Args:
            response_text: Raw LLM response.

        Returns:
            ProposalAnalysis: Parsed analysis.
        """
        # Initialize defaults
        topics = []
        methods = []
        domain = "General Research"
        application_areas = []
        key_phrases = []
        summary = ""

        # Parse line by line
        lines = response_text.strip().split("\n")
        current_field = None
        summary_lines = []

        for line in lines:
            line = line.strip()

            if line.startswith("TOPICS:"):
                topics = self._parse_list_field(line, "TOPICS:")
            elif line.startswith("METHODS:"):
                methods = self._parse_list_field(line, "METHODS:")
            elif line.startswith("DOMAIN:"):
                domain = line.replace("DOMAIN:", "").strip()
            elif line.startswith("APPLICATION_AREAS:"):
                application_areas = self._parse_list_field(line, "APPLICATION_AREAS:")
            elif line.startswith("KEY_PHRASES:"):
                key_phrases = self._parse_list_field(line, "KEY_PHRASES:")
            elif line.startswith("SUMMARY:"):
                current_field = "SUMMARY"
                summary_part = line.replace("SUMMARY:", "").strip()
                if summary_part:
                    summary_lines.append(summary_part)
            elif current_field == "SUMMARY" and line:
                summary_lines.append(line)

        summary = " ".join(summary_lines)

        return ProposalAnalysis(
            topics=topics,
            methods=methods,
            domain=domain,
            application_areas=application_areas,
            key_phrases=key_phrases,
            summary=summary,
            full_text="",  # Will be set by caller
        )

    def _parse_list_field(self, line: str, prefix: str) -> List[str]:
        """
        Parse a comma-separated list field.

        Args:
            line: The line to parse.
            prefix: The field prefix to remove.

        Returns:
            List[str]: Parsed list items.
        """
        content = line.replace(prefix, "").strip()

        # Remove brackets if present
        content = content.strip("[]")

        # Split by comma and clean
        items = [item.strip() for item in content.split(",")]
        items = [item for item in items if item]

        return items

