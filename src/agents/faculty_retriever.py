"""Faculty Retrieval Agent - Retrieves relevant faculty based on proposal analysis."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.indexing.index_builder import IndexManager
from src.agents.proposal_analyzer import ProposalAnalysis
from src.utils.config import Config


@dataclass
class FacultyMatch:
    """Represents a matched faculty member with supporting context."""

    score: float
    faculty_name: Optional[str]
    text: str
    metadata: Dict[str, Any]
    source_type: str  # 'csv', 'pdf', etc.

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "faculty_name": self.faculty_name,
            "text": self.text,
            "metadata": self.metadata,
            "source_type": self.source_type,
        }


class FacultyRetriever:
    """Agent for retrieving relevant faculty based on proposal analysis."""

    def __init__(self, top_k: int = None, collection_name: str = None):
        """
        Initialize FacultyRetriever.

        Args:
            top_k: Number of results to retrieve. Defaults to Config.TOP_K_RESULTS.
            collection_name: ChromaDB collection name. Defaults to Config.COLLECTION_NAME.
        """
        self.top_k = top_k or Config.TOP_K_RESULTS
        self.index_manager = IndexManager(collection_name=collection_name)

    def retrieve_from_analysis(self, analysis: ProposalAnalysis) -> List[FacultyMatch]:
        """
        Retrieve faculty based on proposal analysis.

        Args:
            analysis: ProposalAnalysis object.

        Returns:
            List[FacultyMatch]: List of matched faculty with scores.
        """
        print(f"Retrieving top {self.top_k} faculty matches...")

        # Create search query from analysis
        search_query = analysis.to_search_query()
        print(f"Search query: {search_query}")

        # Retrieve nodes
        nodes = self.index_manager.retriever.retrieve(search_query)

        # Convert nodes to FacultyMatch objects
        matches = []
        for node in nodes[:self.top_k]:
            match = self._node_to_faculty_match(node)
            matches.append(match)

        # Group by faculty name
        grouped_matches = self._group_by_faculty(matches)

        print(f"✓ Retrieved {len(grouped_matches)} unique faculty matches")
        return grouped_matches

    def retrieve_from_query(self, query: str, top_k: int = None) -> List[FacultyMatch]:
        """
        Retrieve faculty based on a text query.

        Args:
            query: Search query string.
            top_k: Number of results. If None, uses self.top_k.

        Returns:
            List[FacultyMatch]: List of matched faculty with scores.
        """
        k = top_k or self.top_k
        print(f"Retrieving top {k} faculty matches for query...")

        nodes = self.index_manager.retriever.retrieve(query)

        matches = []
        for node in nodes[:k]:
            match = self._node_to_faculty_match(node)
            matches.append(match)

        grouped_matches = self._group_by_faculty(matches)

        print(f"✓ Retrieved {len(grouped_matches)} unique faculty matches")
        return grouped_matches

    def _node_to_faculty_match(self, node) -> FacultyMatch:
        """
        Convert a retrieved node to a FacultyMatch.

        Args:
            node: Retrieved node from index.

        Returns:
            FacultyMatch: Converted match object.
        """
        # Extract faculty name from metadata
        faculty_name = None
        if 'name' in node.metadata:
            faculty_name = node.metadata['name']
        elif 'faculty_name' in node.metadata:
            faculty_name = node.metadata['faculty_name']

        # Get source type
        source_type = node.metadata.get('source', 'unknown')

        # Get score
        score = node.score if hasattr(node, 'score') else 0.0

        return FacultyMatch(
            score=score,
            faculty_name=faculty_name,
            text=node.text,
            metadata=node.metadata,
            source_type=source_type,
        )

    def _group_by_faculty(self, matches: List[FacultyMatch]) -> List[FacultyMatch]:
        """
        Group matches by faculty name, keeping the highest-scored match per faculty.

        Args:
            matches: List of faculty matches.

        Returns:
            List[FacultyMatch]: Grouped matches (one per faculty).
        """
        faculty_dict = {}

        for match in matches:
            name = match.faculty_name or "Unknown"

            if name not in faculty_dict:
                faculty_dict[name] = match
            else:
                # Keep the match with higher score
                if match.score > faculty_dict[name].score:
                    faculty_dict[name] = match

        # Sort by score descending
        grouped = sorted(faculty_dict.values(), key=lambda x: x.score, reverse=True)

        return grouped

    def get_faculty_context(
        self,
        faculty_name: str,
        top_k: int = 3
    ) -> List[FacultyMatch]:
        """
        Get all available context about a specific faculty member.

        Args:
            faculty_name: Name of the faculty member.
            top_k: Number of documents to retrieve.

        Returns:
            List[FacultyMatch]: Context documents about the faculty.
        """
        query = f"Faculty member: {faculty_name}. Research interests, publications, expertise."
        nodes = self.index_manager.retriever.retrieve(query)

        # Filter for this faculty member
        matches = []
        for node in nodes:
            if (node.metadata.get('name') == faculty_name or
                node.metadata.get('faculty_name') == faculty_name):
                match = self._node_to_faculty_match(node)
                matches.append(match)

            if len(matches) >= top_k:
                break

        return matches

