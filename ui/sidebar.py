"""Sidebar: session controls, evaluation actions, cost, and developer settings."""

from typing import Tuple
from datetime import datetime

import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from openai import OpenAI

from core.chat import has_candidate_answer
from core.config import get_judge_model, MODEL_OPTIONS, ModelSettings
from core.state import reset_chat
from pricing import format_cost, model_supports_parameter
from prompts import PROMPT_TECHNIQUES
from ui.reports import run_evaluation


def render_sidebar() -> Tuple[ModelSettings, DeltaGenerator]:
    """
    Sidebar: session controls, evaluation actions, cost, and dev settings.

    Returns the settings and an empty container reserved for the evaluation
    buttons — it is filled later in main(), once the setup values are known.
    """
    st.sidebar.header("Session")

    if st.sidebar.button("Clear Chat History", use_container_width=True):
        reset_chat()
        st.rerun()

    st.sidebar.divider()

    # Evaluation actions live here; the container is filled from main() once
    # the setup values (role, job description) are known.
    st.sidebar.subheader("Evaluation")
    evaluation_container = st.sidebar.container()

    st.sidebar.divider()

    # Developer-only settings, kept out of the regular user's way.
    with st.sidebar.expander("Developer settings", expanded=False):
        # Cost transparency: estimated from live OpenRouter pricing + token usage.
        st.subheader("Estimated cost")
        st.markdown(
            f"Last request: **{format_cost(st.session_state.last_cost)}**  \n"
            f"Session total: **{format_cost(st.session_state.total_cost)}**"
        )
        st.caption(
            "What your requests to the AI model cost (OpenRouter API). "
            "Calculated from the tokens used and live per-token prices, converted to EUR. "
            "Resets when you clear the chat."
        )

        st.divider()

        # st.radio instead of st.selectbox: a selectbox always shows an
        # editable-looking filter field, radio is plain non-editable text.
        model = st.radio("Model", options=MODEL_OPTIONS, index=0)

        prompt_technique = st.radio(
            "Prompt technique",
            options=list(PROMPT_TECHNIQUES.keys()),
            index=0,
        )
        st.caption(PROMPT_TECHNIQUES[prompt_technique]["description"])

        st.divider()

        # GPT-5 reasoning models run at a fixed temperature; OpenRouter would
        # silently drop the parameter, so disable the slider instead of lying.
        temperature_supported = model_supports_parameter(model, "temperature")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            disabled=not temperature_supported,
            help="Lower values are more focused; higher values are more creative.",
        )
        if not temperature_supported:
            st.caption(
                "_Not supported by the selected model — reasoning models "
                "always run at their default temperature._"
            )

        st.divider()

        max_tokens = st.slider(
            "Max tokens",
            min_value=1024,
            max_value=16384,
            value=8192,
            step=1024,
            help=(
                "Upper limit on the length of each reply. GPT-5 models spend "
                "part of this budget on hidden reasoning, so keep it generous."
            ),
        )

    return (
        ModelSettings(
            model=model,
            prompt_technique=prompt_technique,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        evaluation_container,
    )


def _format_transcript(messages: list) -> str:
    """Format chat messages into a readable transcript for analysis.

    Format: Speaker: message content on separate lines.
    """
    if not messages:
        return ""

    lines = []
    for i, message in enumerate(messages, 1):
        role = "You" if message["role"] == "user" else "Coach"
        content = message["content"]
        lines.append(f"{role}: {content}")
        lines.append("")  # Blank line between messages

    return "\n".join(lines)


def _download_transcript(messages: list) -> str:
    """Generate transcript content with metadata and return as formatted text."""
    transcript = _format_transcript(messages)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header = f"""Interview Practice Transcript
Generated: {timestamp}
Total exchanges: {len([m for m in messages if m['role'] == 'user'])}

---

"""
    return header + transcript


def render_evaluation_controls(
    container: DeltaGenerator,
    client: OpenAI,
    settings: ModelSettings,
    effective_role: str,
    job_description: str,
) -> None:
    """Fill the reserved sidebar container with the two evaluation actions.

    The buttons stay disabled until a real dialog exists (the interviewer
    has replied at least once), so the page never nags prematurely.
    """
    has_answer = has_candidate_answer(st.session_state.messages)

    with container:
        if st.button(
            "Get full feedback report",
            use_container_width=True,
            disabled=not has_answer,
            help="Feedback on the whole conversation as structured JSON.",
        ):
            run_evaluation(
                client, settings.model, effective_role, job_description, "feedback"
            )
        judge_model = get_judge_model(settings.model)
        if st.button(
            "Evaluate the interviewer",
            use_container_width=True,
            disabled=not has_answer,
            help=(
                f"Independent AI judge ({judge_model}) evaluates the interview quality. "
                f"Assesses the interviewer's performance ({settings.model}), not your answers. "
                "Rates question quality, fairness, topic coverage, and feedback (1-5 scale each)."
            ),
        ):
            st.session_state.judge_model_used = judge_model
            run_evaluation(
                client, judge_model, effective_role, job_description, "judge"
            )

        st.download_button(
            "Download Transcript",
            data=_download_transcript(st.session_state.messages),
            file_name=f"interview_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
            disabled=not has_answer,
            help="Download the conversation as a text file for analysis.",
        )

        if not has_answer:
            st.caption("Available once you've answered an interview question.")
