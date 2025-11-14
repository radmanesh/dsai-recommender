"""Faculty Retrieval Agent - Retrieves relevant faculty based on proposal analysis."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from src.indexing.index_builder import IndexManager
from src.agents.proposal_analyzer import ProposalAnalysis
from src.utils.config import Config
from src.utils.name_matcher import normalize_faculty_name, names_match


@dataclass
class FacultyMatch:
    """Represents a matched faculty member with supporting context."""

    score: float
    faculty_name: Optional[str]
    text: str
    metadata: Dict[str, Any]
    source_type: str  # 'csv', 'pdf', etc.
    pdf_support: Optional[List[Dict[str, Any]]] = None  # Supporting PDF information

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "faculty_name": self.faculty_name,
            "text": self.text,
            "metadata": self.metadata,
            "source_type": self.source_type,
            "pdf_support": self.pdf_support,
        }


class FacultyRetriever:
    """Agent for retrieving relevant faculty based on proposal analysis using dual-store architecture."""

    def __init__(
        self,
        top_k: int = None,
        profiles_collection: str = None,
        pdfs_collection: str = None
    ):
        """
        Initialize FacultyRetriever with dual-store support.

        Args:
            top_k: Number of results to retrieve. Defaults to Config.TOP_K_RESULTS.
            profiles_collection: Collection for faculty profiles. Defaults to Config.FACULTY_PROFILES_COLLECTION.
            pdfs_collection: Collection for PDF documents. Defaults to Config.FACULTY_PDFS_COLLECTION.
        """
        self.top_k = top_k or Config.TOP_K_RESULTS
        self.profiles_collection = profiles_collection or Config.FACULTY_PROFILES_COLLECTION
        self.pdfs_collection = pdfs_collection or Config.FACULTY_PDFS_COLLECTION

        # Create separate index managers for profiles and PDFs
        self.profiles_manager = IndexManager(collection_name=self.profiles_collection)
        self.pdfs_manager = IndexManager(collection_name=self.pdfs_collection)

    def retrieve_from_analysis(self, analysis: ProposalAnalysis) -> List[FacultyMatch]:
        """
        Retrieve faculty based on proposal analysis using dual-store architecture.
        Ranks using faculty_profiles collection, then enriches with PDF evidence.

        Args:
            analysis: ProposalAnalysis object.

        Returns:
            List[FacultyMatch]: List of matched faculty profiles with PDF support.
        """
        print(f"Retrieving top {self.top_k} faculty matches...")

        # Create search query from analysis
        search_query = analysis.to_search_query()
        print(f"Search query: {search_query}")

        # Retrieve from faculty_profiles collection (this is our main ranking)
        profile_nodes = self.profiles_manager.retriever.retrieve(search_query)
        print(f"  Found {len(profile_nodes)} profile nodes")

        # Convert profile nodes to FacultyMatch objects
        matches = []
        for node in profile_nodes[:self.top_k]:
            match = self._node_to_faculty_match(node)
            matches.append(match)

        # Now query PDF store for each faculty to get supporting evidence
        print(f"  Querying PDF store for supporting evidence...")
        for match in matches:
            faculty_id = match.metadata.get('faculty_id')
            if faculty_id:
                # Query PDFs collection with faculty_id filter
                try:
                    filters = MetadataFilters(
                        filters=[ExactMatchFilter(key="faculty_id", value=faculty_id)]
                    )
                    pdf_retriever = self.pdfs_manager.index.as_retriever(
                        similarity_top_k=5,
                        filters=filters
                    )
                    pdf_nodes = pdf_retriever.retrieve(search_query)

                    if pdf_nodes:
                        pdf_support_list = []
                        for pdf_node in pdf_nodes:
                            pdf_support_list.append(self._extract_pdf_support(pdf_node))
                        match.pdf_support = pdf_support_list
                except Exception as e:
                    print(f"    ⚠ Error retrieving PDFs for {faculty_id}: {e}")

        print(f"✓ Retrieved {len(matches)} faculty matches with PDF evidence")
        if any(m.pdf_support for m in matches):
            pdf_count = sum(1 for m in matches if m.pdf_support)
            print(f"  {pdf_count} faculty have supporting PDF documents")

        return matches

    def retrieve_from_query(self, query: str, top_k: int = None) -> List[FacultyMatch]:
        """
        Retrieve faculty based on a text query using dual-store architecture.
        Ranks using faculty_profiles collection, then enriches with PDF evidence.

        Args:
            query: Search query string.
            top_k: Number of results. If None, uses self.top_k.

        Returns:
            List[FacultyMatch]: List of matched faculty profiles with PDF support.
        """
        k = top_k or self.top_k
        print(f"Retrieving top {k} faculty matches for query...")

        # Retrieve from faculty_profiles collection (this is our main ranking)
        profile_nodes = self.profiles_manager.retriever.retrieve(query)
        print(f"  Found {len(profile_nodes)} profile nodes")

        # Convert profile nodes to FacultyMatch objects
        matches = []
        for node in profile_nodes[:k]:
            match = self._node_to_faculty_match(node)
            matches.append(match)

        # Now query PDF store for each faculty to get supporting evidence
        print(f"  Querying PDF store for supporting evidence...")
        for match in matches:
            faculty_id = match.metadata.get('faculty_id')
            if faculty_id:
                # Query PDFs collection with faculty_id filter
                try:
                    filters = MetadataFilters(
                        filters=[ExactMatchFilter(key="faculty_id", value=faculty_id)]
                    )
                    pdf_retriever = self.pdfs_manager.index.as_retriever(
                        similarity_top_k=5,
                        filters=filters
                    )
                    pdf_nodes = pdf_retriever.retrieve(query)

                    if pdf_nodes:
                        pdf_support_list = []
                        for pdf_node in pdf_nodes:
                            pdf_support_list.append(self._extract_pdf_support(pdf_node))
                        match.pdf_support = pdf_support_list
                except Exception as e:
                    print(f"    ⚠ Error retrieving PDFs for {faculty_id}: {e}")

        print(f"✓ Retrieved {len(matches)} faculty matches with PDF evidence")
        if any(m.pdf_support for m in matches):
            pdf_count = sum(1 for m in matches if m.pdf_support)
            print(f"  {pdf_count} faculty have supporting PDF documents")

        return matches

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

    def _normalize_faculty_name(self, name: Optional[str]) -> str:
        """
        Normalize a faculty name for matching.

        Args:
            name: The faculty name to normalize.

        Returns:
            str: Normalized name.
        """
        return normalize_faculty_name(name)

    def _separate_nodes_by_source(self, nodes: List) -> Tuple[List, List]:
        """
        Separate nodes into CSV nodes and PDF nodes.

        Args:
            nodes: List of all retrieved nodes.

        Returns:
            Tuple[List, List]: (csv_nodes, pdf_nodes)
        """
        csv_nodes = []
        pdf_nodes = []

        for node in nodes:
            source = node.metadata.get('source', '')
            if source == 'csv':
                csv_nodes.append(node)
            elif source == 'pdf':
                pdf_nodes.append(node)

        return csv_nodes, pdf_nodes

    def _map_pdfs_to_csv(self, csv_nodes: List, pdf_nodes: List) -> Dict[str, List]:
        """
        Map PDF nodes to CSV faculty names using normalized name matching.

        Args:
            csv_nodes: List of CSV nodes.
            pdf_nodes: List of PDF nodes.

        Returns:
            Dict[str, List]: Mapping of CSV faculty name -> list of matching PDF nodes
        """
        faculty_pdf_map = {}

        for csv_node in csv_nodes:
            csv_faculty_name = csv_node.metadata.get('name')
            if not csv_faculty_name:
                continue

            # Initialize list for this faculty
            if csv_faculty_name not in faculty_pdf_map:
                faculty_pdf_map[csv_faculty_name] = []

            # Find matching PDFs
            for pdf_node in pdf_nodes:
                pdf_faculty_name = pdf_node.metadata.get('faculty_name')
                if pdf_faculty_name and names_match(csv_faculty_name, pdf_faculty_name):
                    faculty_pdf_map[csv_faculty_name].append(pdf_node)

        return faculty_pdf_map

    def _extract_pdf_support(self, pdf_node) -> Dict[str, Any]:
        """
        Extract summary, research_interests, and faculty_name from PDF node metadata.

        Args:
            pdf_node: A PDF node.

        Returns:
            Dict: PDF support information with keys: summary, research_interests, faculty_name
        """
        return {
            'faculty_name': pdf_node.metadata.get('faculty_name'),
            'summary': pdf_node.metadata.get('summary'),
            'research_interests': pdf_node.metadata.get('research_interests'),
            'type': pdf_node.metadata.get('type'),
            'file_name': pdf_node.metadata.get('file_name'),
        }

