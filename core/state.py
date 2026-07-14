"""Session-state bookkeeping — the single place that declares every key.

This is the one core module that imports Streamlit: its whole job is the
`st.session_state` bus shared by the UI modules. All other core modules stay
Streamlit-free.
"""

import copy
from typing import Optional

import streamlit as st

from core.costs import usage_cost

SESSION_DEFAULTS = {
    "messages": [],
    "total_cost": 0.0,
    "last_cost": None,
    "last_feedback": None,
    "last_feedback_cost": None,
    "last_judgment": None,
    "last_judgment_cost": None,
    "judge_model_used": None,
    "jd_analysis": None,
    "jd_analysis_source": "",
    "jd_analysis_cost": None,
    "pending_prompt": None,
    "voice_inject": "",
    "resume_text": "",
    "resume_summary": "",
    "resume_fingerprint": "",
    "resume_error": "",
    "resume_match": None,
    "resume_match_source": "",
    "resume_match_cost": None,
}

# Keys reset by "Clear Chat History". Resume state survives on purpose: the
# uploaded file is still in the uploader widget, so its derived state must
# stay in sync with it.
CHAT_RESET_KEYS = (
    "messages",
    "total_cost",
    "last_cost",
    "last_feedback",
    "last_feedback_cost",
    "last_judgment",
    "last_judgment_cost",
    "judge_model_used",
    "jd_analysis",
    "jd_analysis_source",
    "jd_analysis_cost",
    "resume_match",
    "resume_match_source",
    "resume_match_cost",
)

# A new chat message makes previous evaluation reports stale.
REPORT_KEYS = (
    "last_feedback",
    "last_feedback_cost",
    "last_judgment",
    "last_judgment_cost",
    "judge_model_used",
)


def _default(key: str):
    """A FRESH copy of a default value.

    Never hand out SESSION_DEFAULTS values directly: mutable ones (the
    `messages` list) would be shared with session state, and in-place
    appends would silently corrupt the defaults — making every later
    reset a no-op.
    """
    return copy.deepcopy(SESSION_DEFAULTS[key])


def init_session_state() -> None:
    for key in SESSION_DEFAULTS:
        if key not in st.session_state:
            st.session_state[key] = _default(key)


def reset_chat() -> None:
    """Reset everything the "Clear Chat History" button owns, from one source of truth."""
    for key in CHAT_RESET_KEYS:
        st.session_state[key] = _default(key)


def clear_evaluation_reports() -> None:
    for key in REPORT_KEYS:
        st.session_state[key] = _default(key)


def record_cost(model: str, usage: Optional[object]) -> Optional[float]:
    """Accumulate the estimated USD cost of a request; returns that cost."""
    cost = usage_cost(model, usage)
    if cost is not None:
        st.session_state.last_cost = cost
        st.session_state.total_cost += cost
    return cost
