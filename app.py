"""All-in-One AI Interview Preparation Assistant — Streamlit entry point.

The page is organized as a four-step "candidate journey":
1. Set the scene   — role, company, job description, resume     (ui/setup.py)
2. Get AI insights — job analysis and resume match cards        (ui/reports.py)
3. Practice        — the interview chat                         (ui/chat.py)
4. How am I doing? — structured feedback and LLM-as-a-judge     (ui/reports.py)

This module only wires the pieces together; browser-independent logic lives
in core/ and the rendering in ui/.
"""

import logging
import os

import streamlit as st
from dotenv import load_dotenv

from core.client import get_openrouter_client
from core.config import APP_TITLE
from core.state import init_session_state
from prompts import DEFAULT_TARGET_ROLE
from ui.chat import handle_chat_submission, inject_voice_transcript, render_practice_block
from ui.reports import render_evaluation_reports, render_insights_block
from ui.setup import ensure_resume_summary, render_setup_block, render_status_line
from ui.sidebar import render_evaluation_controls, render_sidebar

load_dotenv()

# API error details are logged here (stderr of the `streamlit run` terminal)
# while the user sees only a safe message — see ui/errors.py.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
    )

    init_session_state()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        st.error(
            "Missing OPENROUTER_API_KEY. Add it to your `.env` file and restart the app."
        )
        st.stop()

    client = get_openrouter_client(api_key)
    settings, evaluation_container = render_sidebar()

    st.title(APP_TITLE)
    st.caption("Practice for your next interview with an AI coach — in four simple steps.")

    # Step 1 — context.
    interview_type, target_role, company_name, job_description = render_setup_block()
    effective_role = (target_role or DEFAULT_TARGET_ROLE).strip()
    render_status_line(
        interview_type, effective_role, company_name, job_description, settings
    )
    ensure_resume_summary(client, settings)

    # Step 2 — AI insights (analysis + match), results inside their cards.
    render_insights_block(client, settings, effective_role, job_description)

    # Step 3 — practice chat (plus filling the chat box with a fresh
    # voice transcript, when there is one).
    render_practice_block()
    inject_voice_transcript()

    # Step 4 — evaluation: actions live in the sidebar, reports on the page.
    render_evaluation_controls(
        evaluation_container, client, settings, effective_role, job_description
    )
    render_evaluation_reports()

    handle_chat_submission(
        client, settings, interview_type, target_role, company_name, job_description
    )


if __name__ == "__main__":
    main()
