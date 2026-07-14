"""Everything that renders an AI report: the step-2 insight cards (job
analysis, resume match) and the step-4 evaluation reports (feedback, rubric)."""

from typing import List, Optional

import streamlit as st
from openai import OpenAI, OpenAIError

from core.config import ModelSettings
from core.state import record_cost
from evaluation import (
    analyse_job_description,
    get_interview_feedback,
    judge_answer,
    match_resume,
)
from pricing import format_cost
from security import MAX_JD_LENGTH, validate_user_input
from ui.errors import show_api_error

INVALID_JSON_MESSAGE = (
    "The model did not return a valid report (malformed or unexpected JSON). "
    "Please try again."
)


def badges(items: List[str], color: str) -> str:
    """Render a list of short labels as colored markdown badges."""
    safe = [str(item).replace("[", "(").replace("]", ")") for item in items]
    return " ".join(f":{color}-badge[{label}]" for label in safe)


def report_footer(data: dict, cost: Optional[float]) -> None:
    """Shared footer for AI reports: raw JSON + what the call cost."""
    col_json, col_cost = st.columns([1, 2])
    with col_json:
        with st.popover("View raw JSON"):
            st.json(data)
    if cost is not None:
        col_cost.caption(f"Report cost: {format_cost(cost)}")


# ---------------------------------------------------------------------------
# Step 2 — AI insights (job analysis + resume match)
# ---------------------------------------------------------------------------


def render_analyse_card(
    client: OpenAI,
    settings: ModelSettings,
    effective_role: str,
    job_description: str,
) -> None:
    """Card: analyse the job description; the report lives inside this card."""
    with st.container(border=True):
        st.markdown("#### Analyse the job")
        st.caption(
            "Extracts **key skills**, **likely interview topics**, "
            "and a short **study plan** from the pasted job description."
        )
        analyse_clicked = st.button(
            "Analyse job description",
            disabled=not job_description.strip(),
            use_container_width=True,
        )
        if not job_description.strip():
            st.caption("_Paste a job description in step 1 to enable this._")

        if analyse_clicked:
            jd_safe, jd_error = validate_user_input(
                job_description, max_length=MAX_JD_LENGTH
            )
            if not jd_safe:
                st.error(f"Job description was blocked: {jd_error}")
            else:
                with st.spinner("Analysing the job description..."):
                    try:
                        analysis, usage = analyse_job_description(
                            client=client,
                            model=settings.model,
                            job_description=job_description,
                            target_role=effective_role,
                        )
                        cost = record_cost(settings.model, usage)
                        if analysis is None:
                            st.error(INVALID_JSON_MESSAGE)
                        else:
                            st.session_state.jd_analysis = analysis
                            st.session_state.jd_analysis_source = job_description
                            st.session_state.jd_analysis_cost = cost
                    except OpenAIError as exc:
                        show_api_error(exc, "Job description analysis")

        data = st.session_state.jd_analysis
        if not data:
            return

        if job_description.strip() != st.session_state.jd_analysis_source.strip():
            st.warning(
                "The job description has changed — click **Analyse job description** "
                "again to refresh this report."
            )

        skills = data.get("key_skills", {})
        hard_skills = skills.get("hard", [])
        soft_skills = skills.get("soft", [])
        st.markdown("**Key skills**")
        if hard_skills:
            st.markdown(badges(hard_skills, "blue"))
        if soft_skills:
            st.markdown(badges(soft_skills, "green"))

        topics = data.get("interview_topics", [])
        st.markdown("**Likely interview topics**")
        for topic in topics:
            st.markdown(f"- {topic}")

        st.markdown("**Study plan**")
        for index, step in enumerate(data.get("study_plan", []), start=1):
            st.markdown(f"{index}. {step}")

        if topics and st.button(
            "Practice these topics",
            help="Start a practice round on the topics above.",
            use_container_width=True,
        ):
            # Bridge into the chat: queue a message as if the user typed it.
            st.session_state.pending_prompt = (
                "Let's practice the likely interview topics from the job "
                f"description analysis: {', '.join(topics)}. "
                "Ask me questions on these topics, one at a time."
            )

        report_footer(data, st.session_state.jd_analysis_cost)


def render_match_card(
    client: OpenAI,
    settings: ModelSettings,
    effective_role: str,
    job_description: str,
) -> None:
    """Card: match the resume against the job; the report lives inside this card."""
    with st.container(border=True):
        st.markdown("#### Match my resume")
        st.caption(
            "Compares your resume against the posting: **match score**, "
            "**strengths**, **gaps**, and **what to improve**."
        )
        ready = bool(st.session_state.resume_text and job_description.strip())
        match_clicked = st.button(
            "Match resume to this job",
            disabled=not ready,
            use_container_width=True,
        )
        if not ready:
            st.caption("_Add both a job description and a resume in step 1 to enable this._")

        if match_clicked:
            with st.spinner("Matching your resume against the job..."):
                try:
                    match, usage = match_resume(
                        client=client,
                        model=settings.model,
                        resume_text=st.session_state.resume_text,
                        job_description=job_description,
                        target_role=effective_role,
                    )
                    cost = record_cost(settings.model, usage)
                    if match is None:
                        st.error(INVALID_JSON_MESSAGE)
                    else:
                        st.session_state.resume_match = match
                        st.session_state.resume_match_source = (
                            st.session_state.resume_fingerprint + job_description
                        )
                        st.session_state.resume_match_cost = cost
                except OpenAIError as exc:
                    show_api_error(exc, "Resume matching")

        data = st.session_state.resume_match
        if not data:
            return

        current_source = st.session_state.resume_fingerprint + job_description
        if st.session_state.resume_match_source != current_source:
            st.warning(
                "The resume or job description has changed — click "
                "**Match resume to this job** again to refresh this report."
            )

        score = data.get("match_score")
        if isinstance(score, (int, float)):
            score = min(max(int(score), 0), 100)
            score_color = "green" if score >= 70 else "orange" if score >= 40 else "red"
            st.markdown(f"**Match score:** :{score_color}[**{score}/100**]")
            st.progress(score / 100)

        col_strengths, col_gaps = st.columns(2)
        with col_strengths:
            st.markdown("**Your strengths for this role**")
            for item in data.get("matching_strengths", []):
                st.markdown(f"- {item}")
        with col_gaps:
            st.markdown("**Gaps**")
            for item in data.get("gaps", []):
                st.markdown(f"- {item}")

        st.markdown("**What to improve for this job**")
        for index, step in enumerate(data.get("improvement_plan", []), start=1):
            st.markdown(f"{index}. {step}")

        if data.get("verdict"):
            st.info(data["verdict"])

        report_footer(data, st.session_state.resume_match_cost)


def render_insights_block(
    client: OpenAI,
    settings: ModelSettings,
    effective_role: str,
    job_description: str,
) -> None:
    """Step 2: the two AI-insight cards side by side."""
    has_reports = bool(st.session_state.jd_analysis or st.session_state.resume_match)
    with st.expander(
        ":violet[**Get AI insights**] — analyse the job and match your resume _(optional)_",
        expanded=has_reports or not st.session_state.messages,
    ):
        col_analyse, col_match = st.columns(2, gap="medium")
        with col_analyse:
            render_analyse_card(client, settings, effective_role, job_description)
        with col_match:
            render_match_card(client, settings, effective_role, job_description)


# ---------------------------------------------------------------------------
# Step 4 — Evaluation (feedback + judge)
# ---------------------------------------------------------------------------


def render_feedback_report(data: dict) -> None:
    """Render the structured JSON feedback report (output format #1)."""
    st.metric("Overall score", f"{data.get('overall_score', '?')}/10")
    col_strengths, col_weaknesses = st.columns(2)
    with col_strengths:
        st.markdown("**Strengths**")
        for item in data.get("strengths", []):
            st.markdown(f"- {item}")
    with col_weaknesses:
        st.markdown("**Weaknesses**")
        for item in data.get("weaknesses", []):
            st.markdown(f"- {item}")
    st.markdown("**Suggestions**")
    for item in data.get("suggestions", []):
        st.markdown(f"- {item}")
    if data.get("summary"):
        st.info(data["summary"])
    report_footer(data, st.session_state.last_feedback_cost)


def render_judgment(data: dict) -> None:
    """Render the LLM-as-a-judge rubric scores (output format #2)."""
    rubric = data.get("rubric", {})
    columns = st.columns(len(rubric) or 1)
    for column, (criterion, details) in zip(columns, rubric.items()):
        column.metric(criterion.title(), f"{details.get('score', '?')}/5")
        column.caption(details.get("rationale", ""))
    if data.get("average_score") is not None:
        st.metric("Average", f"{data['average_score']:.1f}/5")
    if data.get("verdict"):
        st.info(data["verdict"])

    judge_model = st.session_state.get("judge_model_used", "unknown model")
    col_judge, col_cost = st.columns([1, 2])
    with col_judge:
        st.caption(f"**Judge model:** {judge_model}")
    with col_cost:
        if st.session_state.last_judgment_cost is not None:
            st.caption(f"**Report cost:** {format_cost(st.session_state.last_judgment_cost)}")
        else:
            st.caption("**Report cost:** calculating...")


def run_evaluation(
    client: OpenAI,
    model: str,
    target_role: str,
    job_description: str,
    kind: str,
) -> None:
    """Run one of the structured evaluations and store the result in session state."""
    evaluate = get_interview_feedback if kind == "feedback" else judge_answer
    label = "Analyzing the interview..." if kind == "feedback" else "Judging your answer..."

    try:
        with st.spinner(label):
            data, usage = evaluate(
                client=client,
                model=model,
                messages=st.session_state.messages,
                target_role=target_role,
                job_description=job_description,
            )
        cost = record_cost(model, usage)
        if data is None:
            st.error(INVALID_JSON_MESSAGE)
            return
        state_key = "last_feedback" if kind == "feedback" else "last_judgment"
        st.session_state[state_key] = data
        st.session_state[f"{state_key}_cost"] = cost
    except OpenAIError as exc:
        show_api_error(exc, f"Evaluation ({kind})")


def render_evaluation_reports() -> None:
    """Step 4: show evaluation reports in tabs — only once one exists."""
    if not (st.session_state.last_feedback or st.session_state.last_judgment):
        return

    st.markdown("### :orange[How am I doing?]")
    with st.container(border=True):
        tab_feedback, tab_rubric = st.tabs(["Feedback", "Rubric"])
        with tab_feedback:
            if st.session_state.last_feedback:
                render_feedback_report(st.session_state.last_feedback)
            else:
                st.caption("Run **Get full feedback report** (sidebar) to see this.")
        with tab_rubric:
            if st.session_state.last_judgment:
                render_judgment(st.session_state.last_judgment)
            else:
                st.caption("Run **Evaluate the interviewer** (sidebar) to see this.")
