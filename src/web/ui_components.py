"""Reusable UI components for Streamlit app."""

import streamlit as st


def display_faculty_card(recommendation, index):
    """
    Display a faculty recommendation as a card.

    Args:
        recommendation: FacultyRecommendation object
        index: Position in the list
    """
    with st.container():
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"### {index}. {recommendation.faculty_name}")
        with col2:
            st.metric("Score", f"{recommendation.score:.3f}")

        st.markdown(recommendation.explanation)

        if recommendation.research_areas:
            st.markdown("**Areas:** " + ", ".join(recommendation.research_areas))

        if recommendation.contact_info.get('email'):
            st.markdown(f"📧 {recommendation.contact_info['email']}")


def show_system_status():
    """
    Display system status in sidebar.

    Returns:
        bool: True if system is ready, False otherwise
    """
    from src.indexing.vector_store import get_collection_stats

    try:
        stats = get_collection_stats()
        st.success(f"✓ {stats['count']} faculty indexed")
        return True
    except Exception as e:
        st.error(f"✗ System error: {e}")
        return False


def show_loading_spinner(message="Processing..."):
    """
    Show a loading spinner with custom message.

    Args:
        message: Message to display during loading

    Returns:
        Spinner context manager
    """
    return st.spinner(message)


def display_proposal_analysis(analysis):
    """
    Display proposal analysis in a formatted way.

    Args:
        analysis: ProposalAnalysis object
    """
    with st.expander("📋 Proposal Analysis", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Domain:**")
            st.info(analysis.domain)

            if analysis.topics:
                st.markdown("**Topics:**")
                for topic in analysis.topics:
                    st.markdown(f"- {topic}")

        with col2:
            if analysis.methods:
                st.markdown("**Methods:**")
                for method in analysis.methods:
                    st.markdown(f"- {method}")

            if analysis.application_areas:
                st.markdown("**Applications:**")
                for app in analysis.application_areas:
                    st.markdown(f"- {app}")

        if analysis.summary:
            st.markdown("**Summary:**")
            st.write(analysis.summary)


def display_recommendation_card(rec, results, orchestrator):
    """
    Display a single recommendation card with expandable details.

    Args:
        rec: FacultyRecommendation object
        results: MatchingResult object
        orchestrator: ResearchMatchOrchestrator instance
    """
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
                        email = orchestrator.generate_email_for_recommendation(
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


def show_error_message(error, context=""):
    """
    Display a formatted error message.

    Args:
        error: Exception or error message
        context: Additional context about where the error occurred
    """
    st.error(f"❌ Error{' in ' + context if context else ''}")
    st.exception(error)


def show_success_message(message):
    """
    Display a success message.

    Args:
        message: Success message to display
    """
    st.success(f"✅ {message}")


def show_info_message(message):
    """
    Display an info message.

    Args:
        message: Info message to display
    """
    st.info(f"ℹ️ {message}")

