"""Smoke test: the full page renders headlessly and core UI-state transitions work.

Uses Streamlit's AppTest, so the real app.py (and the whole ui/ + core/ wiring)
runs without a browser. All outbound HTTP (pricing catalogs, exchange rate) is
cut off, exercising the offline fallbacks.
"""

import pathlib
from types import SimpleNamespace

import httpx
import pytest
import requests
from openai import APIConnectionError
from streamlit.testing.v1 import AppTest

import core.client

APP_PATH = str(pathlib.Path(__file__).resolve().parent.parent / "app.py")


class OfflineOpenAIClient:
    """Every completion call fails like a lost connection — no sockets opened."""

    def __init__(self):
        def _refuse(**kwargs):
            raise APIConnectionError(
                request=httpx.Request("POST", "https://openrouter.ai/api/v1")
            )

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=_refuse)
        )


@pytest.fixture()
def app(monkeypatch):
    def _no_network(*args, **kwargs):
        raise requests.RequestException("network disabled in tests")

    monkeypatch.setattr(requests, "get", _no_network)
    monkeypatch.setattr(
        core.client, "get_openrouter_client", lambda api_key: OfflineOpenAIClient()
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    return AppTest.from_file(APP_PATH, default_timeout=20)


def test_page_renders_without_exceptions(app):
    at = app.run()
    assert not at.exception
    assert at.title[0].value == "AI Interview Preparation Assistant"
    # A fresh session starts with an empty conversation and zero cost.
    assert at.session_state["messages"] == []
    assert at.session_state["total_cost"] == 0.0


def test_clear_chat_history_resets_session(app):
    at = app.run()
    # Simulate an existing conversation with stale reports and costs.
    at.session_state["messages"] = [
        {"role": "user", "content": "ask me"},
        {"role": "assistant", "content": "Why this role?"},
    ]
    at.session_state["total_cost"] = 0.5
    at.session_state["last_feedback"] = {"overall_score": 5}
    at.run()

    clear_button = next(b for b in at.sidebar.button if "Clear" in b.label)
    clear_button.click().run()

    assert not at.exception
    assert at.session_state["messages"] == []
    assert at.session_state["total_cost"] == 0.0
    assert at.session_state["last_feedback"] is None


def test_quick_start_button_queues_prompt_for_next_run(app):
    at = app.run()
    quick_start = next(
        b for b in at.main.button if b.label == "Ask me the first question"
    )
    quick_start.click().run()
    # The queued prompt is consumed by handle_chat_submission on the SAME
    # run, which then tries to call the API and fails safely offline: the
    # user message must have been rolled back, never stored unanswered.
    assert not at.exception
    assert at.session_state["messages"] == []
    assert at.session_state["pending_prompt"] is None


def test_clear_chat_survives_in_place_mutation(app):
    """Regression: the app appends to st.session_state.messages IN PLACE.

    If session state ever aliases the module-level SESSION_DEFAULTS list,
    those appends corrupt the defaults and Clear Chat History silently
    stops working (assigning the polluted list back to itself).
    """
    at = app.run()
    # Mutate the live list exactly like handle_chat_submission does.
    at.session_state["messages"].append({"role": "user", "content": "ask me"})
    at.session_state["messages"].append({"role": "assistant", "content": "Why?"})
    at.run()

    clear_button = next(b for b in at.sidebar.button if "Clear" in b.label)
    clear_button.click().run()
    assert at.session_state["messages"] == []

    # And it must keep working on the second round, not just the first.
    at.session_state["messages"].append({"role": "user", "content": "again"})
    at.run()
    clear_button = next(b for b in at.sidebar.button if "Clear" in b.label)
    clear_button.click().run()
    assert not at.exception
    assert at.session_state["messages"] == []


def test_quick_actions_replace_quick_starts_mid_conversation(app):
    at = app.run()
    at.session_state["messages"] = [
        {"role": "user", "content": "ask me"},
        {"role": "assistant", "content": "Why this role?"},
    ]
    at.run()

    labels = [b.label for b in at.main.button]
    # The opener buttons give way to follow-up actions, but a quick row
    # is still there — buttons never just disappear on the user.
    assert "Ask me the first question" not in labels
    for expected in ("Next question", "Give me a hint", "Make it harder"):
        assert expected in labels

    hint = next(b for b in at.main.button if b.label == "Give me a hint")
    hint.click().run()
    # Same contract as quick starts: consumed the same run, and with the
    # offline client the failed exchange is rolled back untouched.
    assert not at.exception
    assert len(at.session_state["messages"]) == 2
    assert at.session_state["pending_prompt"] is None


def test_voice_transcript_injection_is_one_shot(app):
    """A pending transcript is injected into the chat box exactly once.

    The flag must be consumed by the run that renders the injection script,
    so a later rerun can never overwrite the user's in-place edits.
    """
    at = app.run()
    at.session_state["voice_inject"] = "I led a data migration project."
    at.run()

    assert not at.exception
    assert at.session_state["voice_inject"] == ""
    # Nothing was sent anywhere — the text only sits in the input box.
    assert at.session_state["pending_prompt"] is None
    assert at.session_state["messages"] == []
