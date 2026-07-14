"""Tests for core.chat: streaming, message assembly, evaluability, error mapping."""

from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, OpenAIError, RateLimitError

from core.chat import (
    AssistantStream,
    build_api_messages,
    describe_api_error,
    has_candidate_answer,
    stream_assistant_response,
)
from core.config import ModelSettings

SETTINGS = ModelSettings(
    model="openai/gpt-5-mini",
    prompt_technique="Zero-Shot",
    temperature=0.7,
    max_tokens=8192,
)


def _chunk(content=None, usage=None):
    choices = []
    if content is not None:
        choices = [SimpleNamespace(delta=SimpleNamespace(content=content))]
    return SimpleNamespace(choices=choices, usage=usage)


class TestAssistantStream:
    def test_yields_content_and_captures_usage(self):
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=3)
        stream = AssistantStream(
            [
                _chunk("Hel"),
                _chunk(None),  # keep-alive chunk without choices
                _chunk("lo"),
                _chunk(None, usage=usage),  # final chunk: usage, no content
            ]
        )
        assert stream.usage is None  # not consumed yet
        assert "".join(stream) == "Hello"
        assert stream.usage is usage

    def test_empty_stream(self):
        stream = AssistantStream([])
        assert "".join(stream) == ""
        assert stream.usage is None


class TestStreamAssistantResponse:
    def test_passes_settings_and_requests_usage(self):
        captured = {}

        class _Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return [_chunk("Hi")]

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=_Completions())
        )
        messages = [{"role": "system", "content": "sys"}]
        stream = stream_assistant_response(client, messages, SETTINGS)

        assert "".join(stream) == "Hi"
        assert captured["model"] == SETTINGS.model
        assert captured["temperature"] == SETTINGS.temperature
        assert captured["max_tokens"] == SETTINGS.max_tokens
        assert captured["stream"] is True
        assert captured["stream_options"] == {"include_usage": True}


class TestBuildApiMessages:
    def test_system_prompt_comes_first(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        messages = build_api_messages("SYS", history)
        assert messages[0] == {"role": "system", "content": "SYS"}
        assert messages[1:] == history

    def test_history_is_not_mutated(self):
        history = [{"role": "user", "content": "hi"}]
        build_api_messages("SYS", history)
        assert history == [{"role": "user", "content": "hi"}]


class TestHasCandidateAnswer:
    def test_empty_history(self):
        assert not has_candidate_answer([])

    def test_first_user_message_is_a_request_not_an_answer(self):
        assert not has_candidate_answer([{"role": "user", "content": "ask me"}])

    def test_assistant_reply_alone_is_not_evaluable(self):
        assert not has_candidate_answer(
            [
                {"role": "user", "content": "ask me"},
                {"role": "assistant", "content": "Why this role?"},
            ]
        )

    def test_user_message_after_assistant_is_an_answer(self):
        assert has_candidate_answer(
            [
                {"role": "user", "content": "ask me"},
                {"role": "assistant", "content": "Why this role?"},
                {"role": "user", "content": "Because..."},
            ]
        )


class TestDescribeApiError:
    @staticmethod
    def _status_error(cls, status_code):
        request = httpx.Request("POST", "https://openrouter.ai/api/v1")
        response = httpx.Response(status_code, request=request)
        return cls(
            "secret internal details", response=response, body={"leak": "nope"}
        )

    def test_rate_limit(self):
        exc = self._status_error(RateLimitError, 429)
        assert "Rate limit" in describe_api_error(exc)

    def test_connection_error(self):
        exc = APIConnectionError(request=httpx.Request("POST", "https://x"))
        assert "Unable to connect" in describe_api_error(exc)

    def test_status_error_exposes_status_but_not_details(self):
        exc = self._status_error(APIStatusError, 502)
        message = describe_api_error(exc)
        assert "502" in message
        assert "secret internal details" not in message
        assert "leak" not in message

    def test_generic_openai_error_is_masked(self):
        message = describe_api_error(OpenAIError("internal stack details"))
        assert "internal stack details" not in message
        assert "try again" in message.lower()

    def test_unexpected_error_is_masked(self):
        message = describe_api_error(ValueError("boom"))
        assert "boom" not in message
        assert "unexpected" in message.lower()
