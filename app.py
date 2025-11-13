"""Streamlit web interface for Faculty Research Matchmaker."""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.agents.orchestrator import ResearchMatchOrchestrator
from src.indexing.vector_store import get_collection_stats
import tempfile
import nest_asyncio

nest_asyncio.apply()

# Page configuration
st.set_page_config(
    page_title="Faculty Research Matchmaker",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'top_n' not in st.session_state:
    st.session_state.top_n = 5


def init_orchestrator():
    """Initialize the orchestrator (cached)."""
    if st.session_state.orchestrator is None:
        with st.spinner("Loading models..."):
            st.session_state.orchestrator = ResearchMatchOrchestrator(
                top_n_recommendations=st.session_state.top_n
            )
    return st.session_state.orchestrator


def main():
    """Main application function."""
    # Header
    st.title("🎓 Faculty Research Matchmaker")
    st.markdown("Match PhD proposals to relevant OU faculty using AI")

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Settings")

        # System status
        st.subheader("System Status")
        try:
            stats = get_collection_stats()
            st.success(f"✓ Collection: {stats['count']} items indexed")
        except Exception as e:
            st.error(f"✗ Collection error: {e}")
            st.stop()

        # Settings
        st.subheader("Query Settings")
        st.session_state.top_n = st.slider(
            "Number of recommendations",
            min_value=1,
            max_value=10,
            value=5
        )

        # About
        with st.expander("ℹ️ About"):
            st.markdown("""
            This system uses:
            - Local Qwen 1.5B LLM
            - BGE embeddings
            - ChromaDB vector storage

            For best results, provide detailed
            research proposals.
            """)

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📄 Upload PDF", "✍️ Text Input", "🔍 Quick Search"])

    with tab1:
        st.header("Upload Proposal PDF")
        uploaded_file = st.file_uploader(
            "Drop your PhD proposal PDF here",
            type=['pdf'],
            help="Upload a PDF file containing the research proposal"
        )

        if uploaded_file is not None:
            st.info(f"📎 Uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")

            if st.button("🚀 Analyze Proposal", type="primary", key="analyze_pdf"):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # Process
                orchestrator = init_orchestrator()

                with st.spinner("Analyzing proposal and matching faculty..."):
                    result = orchestrator.match_proposal_pdf(
                        tmp_path,
                        generate_report=False
                    )
                    st.session_state.results = result

                # Clean up
                Path(tmp_path).unlink()

                st.success("✅ Analysis complete!")

    with tab2:
        st.header("Enter Proposal Text")
        proposal_text = st.text_area(
            "Paste your research proposal here",
            height=300,
            placeholder="Enter your PhD proposal text, research interests, or description..."
        )

        if st.button("🚀 Analyze Text", type="primary", key="analyze_text") and proposal_text:
            orchestrator = init_orchestrator()

            with st.spinner("Analyzing proposal and matching faculty..."):
                result = orchestrator.match_proposal_text(
                    proposal_text,
                    generate_report=False
                )
                st.session_state.results = result

            st.success("✅ Analysis complete!")

    with tab3:
        st.header("Quick Search")
        query = st.text_input(
            "Search keywords or research area",
            placeholder="e.g., machine learning, multi-agent systems"
        )

        if st.button("🔍 Search", type="primary", key="quick_search") and query:
            orchestrator = init_orchestrator()

            with st.spinner("Searching..."):
                recommendations = orchestrator.quick_search(query, top_n=st.session_state.top_n)
                # Create minimal result object
                from types import SimpleNamespace
                st.session_state.results = SimpleNamespace(
                    recommendations=recommendations,
                    proposal_analysis=None
                )

            st.success("✅ Search complete!")

    # Display results
    if st.session_state.results is not None:
        st.markdown("---")
        st.header("📊 Recommendations")

        results = st.session_state.results

        # Show proposal analysis if available
        if hasattr(results, 'proposal_analysis') and results.proposal_analysis:
            analysis = results.proposal_analysis
            with st.expander("📋 Proposal Analysis", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Domain:**")
                    st.info(analysis.domain)
                    st.markdown("**Topics:**")
                    for topic in analysis.topics:
                        st.markdown(f"- {topic}")
                with col2:
                    st.markdown("**Methods:**")
                    for method in analysis.methods:
                        st.markdown(f"- {method}")
                    st.markdown("**Applications:**")
                    for app in analysis.application_areas:
                        st.markdown(f"- {app}")

        # Display recommendations
        for rec in results.recommendations:
            with st.container():
                # Header with rank and name
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader(f"#{rec.rank} {rec.faculty_name}")
                with col2:
                    st.metric("Match Score", f"{rec.score:.3f}")

                # Explanation
                st.markdown(rec.explanation)

                # Details in expander
                with st.expander("🔍 View Details"):
                    if rec.research_areas:
                        st.markdown("**Research Areas:**")
                        for area in rec.research_areas:
                            st.markdown(f"- {area}")

                    if rec.contact_info:
                        st.markdown("**Contact Information:**")
                        if 'email' in rec.contact_info:
                            st.markdown(f"📧 Email: {rec.contact_info['email']}")
                        if 'website' in rec.contact_info:
                            st.markdown(f"🌐 Website: [{rec.contact_info['website']}]({rec.contact_info['website']})")
                        if 'department' in rec.contact_info:
                            st.markdown(f"🏛️ Department: {rec.contact_info['department']}")

                    # Generate email draft
                    if hasattr(results, 'proposal_analysis') and results.proposal_analysis:
                        if st.button(f"✉️ Generate Email Draft", key=f"email_{rec.rank}"):
                            with st.spinner("Generating email..."):
                                email = st.session_state.orchestrator.generate_email_for_recommendation(
                                    rec,
                                    results.proposal_analysis,
                                    sender_name="PhD Candidate"
                                )
                            st.text_area(
                                "Email Draft:",
                                value=email,
                                height=200,
                                key=f"email_text_{rec.rank}"
                            )

                st.markdown("---")


if __name__ == "__main__":
    main()

