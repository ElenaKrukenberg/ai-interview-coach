"""Step 3 — Practice: the chat history, quick starts, and message handling.

The submission flow is split into small single-purpose functions so no step
grows back into a god function: read input → validate → build prompt →
stream reply → store or roll back.
"""

import json
from typing import List, Optional

import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

from core.chat import build_api_messages, stream_assistant_response
from core.config import QUICK_ACTIONS, QUICK_STARTS, TRANSCRIPTION_MODEL, ModelSettings
from core.state import clear_evaluation_reports, record_cost
from core.voice import recording_is_silent, transcribe_audio
from prompts import build_system_prompt
from security import MAX_JD_LENGTH, validate_user_input
from ui.errors import show_api_error


def _render_quick_buttons(actions, key_prefix: str) -> None:
    """One row of buttons that each queue a canned message as if the user typed it."""
    columns = st.columns(len(actions))
    for column, (label, prompt_text) in zip(columns, actions):
        if column.button(label, key=f"{key_prefix}:{label}", use_container_width=True):
            st.session_state.pending_prompt = prompt_text


def render_practice_block() -> None:
    """Step 3: welcome card with quick starts, or the running conversation.

    Quick buttons stay available in both states — only their wording changes:
    conversation openers before the first message, follow-up actions after.
    """
    st.markdown("### :green[Practice]")

    if not st.session_state.messages:
        with st.container(border=True):
            st.caption(
                "Type a message — or tap the microphone to answer by voice — "
                "in the chat box at the bottom of the page, "
                "or jump in with one of these:"
            )
            _render_quick_buttons(QUICK_STARTS, "quick_start")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Rendered after the last message, so the row sits right above the
    # chat input at the bottom of the page.
    st.caption("Quick actions:")
    _render_quick_buttons(QUICK_ACTIONS, "quick_action")


def _transcribe_submission(client: OpenAI, audio_file) -> Optional[str]:
    """Turn a chat-input voice recording into text; None when nothing usable."""
    audio_bytes = audio_file.getvalue()
    if recording_is_silent(audio_bytes):
        st.warning(
            "The recording contains no sound — the microphone is recording "
            "pure silence. Check that your browser is allowed to use the "
            "microphone and that the right input device is selected "
            "(macOS: System Settings → Sound → Input)."
        )
        return None
    try:
        with st.spinner("Transcribing your answer..."):
            transcript, usage = transcribe_audio(client, audio_bytes)
        record_cost(TRANSCRIPTION_MODEL, usage)
        if not transcript:
            st.warning(
                "No speech was recognized. Try again a bit closer to the microphone."
            )
        return transcript
    except Exception as exc:
        show_api_error(exc, "Voice transcription")
        return None


def inject_voice_transcript() -> None:
    """Put the pending transcript into the chat input box for editing.

    Streamlit cannot pre-fill st.chat_input from Python, so this is the one
    place the app steps outside the framework: a zero-height component whose
    script writes the transcript into the chat box in the browser DOM (via
    the native value setter plus an `input` event, so React registers it)
    and focuses the box. The user edits the text in place and presses Enter —
    a perfectly normal chat submission from there on.

    Runs exactly once per transcript (the flag is cleared before rendering),
    so a later rerun never overwrites the user's in-place edits. If a future
    Streamlit release renames the DOM hooks, the script logs to the browser
    console and degrades to an empty input.
    """
    transcript = st.session_state.voice_inject
    if not transcript:
        return
    st.session_state.voice_inject = ""

    payload = json.dumps(transcript)
    components.html(
        f"""<script>
        const fill = (attempt) => {{
            const box = parent.document.querySelector(
                '[data-testid="stChatInput"] textarea');
            if (!box) {{
                if (attempt < 20) setTimeout(() => fill(attempt + 1), 100);
                else console.warn("voice input: chat input textarea not found");
                return;
            }}
            const setter = Object.getOwnPropertyDescriptor(
                parent.window.HTMLTextAreaElement.prototype, "value").set;
            setter.call(box, {payload});
            box.dispatchEvent(new Event("input", {{ bubbles: true }}));
            box.focus();
        }};
        fill(0);
        </script>""",
        height=0,
    )


def get_submitted_prompt(client: OpenAI) -> Optional[str]:
    """One submitted message: typed text, a transcribed voice recording, or
    a message queued by a quick button.

    With ``accept_audio`` the chat input returns a dict-like value exactly
    once, on the submit rerun — so a recording is transcribed (a paid call)
    exactly once, with no bookkeeping. The transcript is not returned
    directly: it is queued for injection into the chat input itself, where
    the user reviews and edits it before sending.
    """
    submission = st.chat_input(
        "Ask your interview coach anything — type, or record your answer...",
        accept_audio=True,
    )
    if submission is None:
        if st.session_state.pending_prompt:
            prompt = st.session_state.pending_prompt
            st.session_state.pending_prompt = None
            return prompt
        return None

    text = (submission.text or "").strip()
    transcript = (
        _transcribe_submission(client, submission.audio) if submission.audio else None
    )
    if transcript:
        # Typed context and a spoken answer in one submission both land in
        # the input box; st.rerun() makes the injection happen immediately.
        st.session_state.voice_inject = f"{text}\n\n{transcript}" if text else transcript
        st.rerun()
    return text or None


def validate_chat_input(prompt: str, job_description: str, company_name: str) -> bool:
    """Security guards: the message and all optional context fields are scanned."""
    is_safe, error_message = validate_user_input(prompt)
    if not is_safe:
        st.error(error_message)
        return False

    jd_safe, jd_error = validate_user_input(
        job_description, allow_empty=True, max_length=MAX_JD_LENGTH
    )
    if not jd_safe:
        st.error(f"Job description was blocked: {jd_error}")
        return False

    company_safe, company_error = validate_user_input(company_name, allow_empty=True)
    if not company_safe:
        st.error(f"Company name was blocked: {company_error}")
        return False

    return True


def _stream_reply(
    client: OpenAI,
    api_messages: List[dict],
    settings: ModelSettings,
) -> Optional[str]:
    """Render the streaming reply; returns the full text, or None on API error."""
    try:
        stream = stream_assistant_response(client, api_messages, settings)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            for chunk in stream:
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response or "_No response generated._")
        record_cost(settings.model, stream.usage)
        return full_response
    except Exception as exc:
        show_api_error(exc, "Chat completion")
        return None


def handle_chat_submission(
    client: OpenAI,
    settings: ModelSettings,
    interview_type: str,
    target_role: str,
    company_name: str,
    job_description: str,
) -> None:
    """Read, validate, send, and store one chat exchange (if any was submitted)."""
    prompt = get_submitted_prompt(client)
    if not prompt:
        return

    if not validate_chat_input(prompt, job_description, company_name):
        return

    # A new message makes previous evaluation reports stale — clear them.
    clear_evaluation_reports()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    system_prompt = build_system_prompt(
        settings.prompt_technique,
        interview_type,
        target_role,
        company_name,
        job_description,
        st.session_state.resume_summary,
    )
    api_messages = build_api_messages(system_prompt, st.session_state.messages)

    assistant_reply = _stream_reply(client, api_messages, settings)

    if assistant_reply:
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_reply}
        )
        st.rerun()
    elif assistant_reply is not None:
        # Model produced no visible text — its hidden reasoning likely
        # consumed the whole token budget. Keep the user's message and
        # warn instead of storing an invisible empty reply.
        st.warning(
            "The model returned an empty reply. Try increasing "
            "**Max tokens** in Developer settings and send your message again."
        )
    else:
        # The request failed (the error is already shown above). Remove the
        # unanswered user message so the history never ends up with two
        # consecutive user turns on the next send.
        st.session_state.messages.pop()
        st.info("Your message was not sent. Copy it below and try again:")
        st.code(prompt, language=None)
