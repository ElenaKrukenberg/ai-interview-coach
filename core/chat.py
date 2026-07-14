"""Chat plumbing: streaming API calls, message assembly, error mapping.

No Streamlit here — functions take plain data and return plain data, so the
UI layer decides how chunks are rendered and errors are shown, and everything
in this module is testable without a browser.
"""

from typing import Dict, Iterator, List, Optional

from openai import (
    APIConnectionError,
    APIStatusError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import ChatCompletionMessageParam

from core.config import ModelSettings


class AssistantStream:
    """Iterable of assistant text chunks.

    Wraps a raw OpenAI streaming response; `usage` is populated once the
    stream has been fully consumed (the final chunk carries token usage
    instead of content).
    """

    def __init__(self, raw_stream) -> None:
        self._raw_stream = raw_stream
        self.usage: Optional[object] = None

    def __iter__(self) -> Iterator[str]:
        for chunk in self._raw_stream:
            if getattr(chunk, "usage", None):
                self.usage = chunk.usage
            if not chunk.choices:
                continue
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                yield content


def stream_assistant_response(
    client: OpenAI,
    messages: List[ChatCompletionMessageParam],
    settings: ModelSettings,
) -> AssistantStream:
    """Start a streaming completion; API errors propagate to the caller."""
    raw_stream = client.chat.completions.create(
        model=settings.model,
        messages=messages,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        stream=True,
        stream_options={"include_usage": True},
    )
    return AssistantStream(raw_stream)


def build_api_messages(
    system_prompt: str,
    history: List[Dict[str, str]],
) -> List[ChatCompletionMessageParam]:
    """Prepend the system prompt to the visible chat history."""
    api_messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt}
    ]
    api_messages.extend(history)
    return api_messages


def has_candidate_answer(messages: List[Dict[str, str]]) -> bool:
    """Whether the transcript contains something evaluable.

    There is something to evaluate only after the candidate has actually
    answered — i.e. sent a message AFTER the interviewer's reply. The very
    first user message is a request ("ask me a question"), not an answer.
    """
    seen_assistant = False
    for message in messages:
        if message["role"] == "assistant":
            seen_assistant = True
        elif message["role"] == "user" and seen_assistant:
            return True
    return False


def describe_api_error(exc: Exception) -> str:
    """Map an API exception to a safe, user-facing message.

    Deliberately excludes provider error bodies — they can contain internal
    details and are logged instead (see ui.errors.show_api_error).
    """
    if isinstance(exc, RateLimitError):
        return (
            "Rate limit exceeded. Please wait a moment and try again, "
            "or switch to a different model."
        )
    if isinstance(exc, APIConnectionError):
        return "Unable to connect to OpenRouter. Check your internet connection and try again."
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", "unknown")
        return (
            f"The model request failed (status {status}). "
            "Please try again or select another model."
        )
    if isinstance(exc, OpenAIError):
        return "The model request failed. Please try again or select another model."
    return "An unexpected error occurred. Please try again."
