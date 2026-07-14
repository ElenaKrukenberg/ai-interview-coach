"""User-facing error reporting: safe message on screen, full details in the log.

The log goes to stderr of the terminal running `streamlit run app.py` (the
standard `logging` root handler configured in app.py).
"""

import logging

import streamlit as st

from core.chat import describe_api_error

logger = logging.getLogger("interview_prep")


def show_api_error(exc: Exception, context: str) -> None:
    """Log the full exception with traceback; show the user a safe message only."""
    logger.error("%s failed", context, exc_info=exc)
    st.error(describe_api_error(exc))
