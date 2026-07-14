"""Step 1 — Set the scene: interview context, documents, and the status line."""

from typing import Tuple

import streamlit as st
from openai import OpenAI, OpenAIError

from core.config import ModelSettings
from core.resume import process_resume_file
from core.state import record_cost
from evaluation import summarize_resume
from prompts import DEFAULT_TARGET_ROLE, INTERVIEW_TYPES
from ui.errors import show_api_error


def _sync_resume_state(resume_file) -> None:
    """Keep resume-derived session state in sync with the uploader widget."""
    if resume_file is None:
        # File removed: drop all resume-derived state.
        st.session_state.resume_text = ""
        st.session_state.resume_summary = ""
        st.session_state.resume_fingerprint = ""
        st.session_state.resume_match = None
        st.session_state.resume_error = ""
        return

    fingerprint = f"{resume_file.name}:{resume_file.size}"
    if fingerprint == st.session_state.resume_fingerprint:
        return

    # Store the fingerprint even when the file is rejected, so a blocked
    # file is not re-extracted on every rerun.
    st.session_state.resume_fingerprint = fingerprint
    st.session_state.resume_match = None
    resume_text, resume_error = process_resume_file(resume_file)
    # On a blocked file process_resume_file returns empty text, so ALL resume
    # state (including any previously accepted resume) is dropped and stale
    # text never leaks into prompts or a false "Resume loaded" message.
    st.session_state.resume_text = resume_text
    st.session_state.resume_summary = ""  # rebuilt lazily via ensure_resume_summary()
    st.session_state.resume_error = resume_error


def render_setup_block() -> Tuple[str, str, str, str]:
    """Step 1: interview context. Collapses once the conversation starts."""
    with st.expander(
        ":blue[**Set the scene**] — who you are and what you're applying for",
        expanded=not st.session_state.messages,
    ):
        col_context, col_documents = st.columns(2, gap="large")

        with col_context:
            target_role = st.text_input(
                "Target role",
                value=DEFAULT_TARGET_ROLE,
                help="The position you are preparing to interview for.",
                placeholder="e.g., Python Developer, QA Automation, Product Manager",
            )
            company_name = st.text_input(
                "Company name (optional)",
                help="The company you are interviewing with — questions get tailored to it.",
                placeholder="e.g., Spotify, Deutsche Bahn, a local fintech startup",
            )
            interview_type = st.selectbox(
                "Interview type",
                options=list(INTERVIEW_TYPES.keys()),
                index=0,
                help="What kind of interview practice you want.",
            )

        with col_documents:
            job_description = st.text_area(
                "Job description (optional)",
                height=150,
                help="Paste the job posting to get questions tailored to it.",
                placeholder="Paste the job posting here...",
            )
            if job_description.strip():
                st.success(
                    f"Job description loaded ({len(job_description)} characters)."
                )

            resume_file = st.file_uploader(
                "Resume (optional)",
                type=["pdf", "txt", "md"],
                help="Upload your resume to match it against the job and personalize the interview.",
            )
            _sync_resume_state(resume_file)
            if resume_file is not None:
                if st.session_state.resume_error:
                    st.error(f"Resume was blocked: {st.session_state.resume_error}")
                elif st.session_state.resume_text:
                    st.success(
                        f"Resume loaded: {resume_file.name} "
                        f"({len(st.session_state.resume_text)} characters extracted)."
                    )
                    st.caption(
                        "Your resume is sent to the AI model when you use "
                        "matching or interview practice."
                    )

    return interview_type, target_role, company_name, job_description


def render_status_line(
    interview_type: str,
    effective_role: str,
    company_name: str,
    job_description: str,
    settings: ModelSettings,
) -> None:
    """One-line summary of the current context, always visible."""
    company_status = f" @ **{company_name.strip()}**" if company_name.strip() else ""
    jd_status = " · Job description" if job_description.strip() else ""
    resume_status = " · Resume" if st.session_state.resume_summary else ""
    st.caption(
        f"**{interview_type}** for **{effective_role}**{company_status} "
        f"· Technique: **{settings.prompt_technique}** · Model: `{settings.model}`"
        f"{jd_status}{resume_status}"
    )


def ensure_resume_summary(client: OpenAI, settings: ModelSettings) -> None:
    """Summarize the resume once per upload; the summary is reused in every
    chat system prompt instead of the full resume text."""
    if not st.session_state.resume_text or st.session_state.resume_summary:
        return
    with st.spinner("Reading your resume..."):
        try:
            summary, usage = summarize_resume(
                client=client,
                model=settings.model,
                resume_text=st.session_state.resume_text,
            )
            record_cost(settings.model, usage)
            st.session_state.resume_summary = (
                summary or st.session_state.resume_text[:1500]
            )
        except OpenAIError as exc:
            show_api_error(exc, "Resume summarization")
