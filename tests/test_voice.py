"""Tests for core.voice: silence detection and the speech-to-text call."""

import base64
import io
import math
import struct
import wave
from types import SimpleNamespace

from core.config import TRANSCRIPTION_MODEL
from core.voice import TRANSCRIPTION_PROMPT, recording_is_silent, transcribe_audio


def _wav(samples, framerate=16000) -> bytes:
    """Build an in-memory 16-bit mono WAV from integer samples."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(framerate)
        recording.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buffer.getvalue()


class TestRecordingIsSilent:
    def test_all_zero_recording_is_silent(self):
        # The macOS no-mic-permission case: the browser records pure zeros.
        assert recording_is_silent(_wav([0] * 16000)) is True

    def test_speech_level_signal_is_not_silent(self):
        tone = [int(10000 * math.sin(i / 10)) for i in range(16000)]
        assert recording_is_silent(_wav(tone)) is False

    def test_quiet_noise_floor_is_still_silent(self):
        # Sub-audible dither well below any real microphone's noise floor.
        assert recording_is_silent(_wav([16, -16] * 8000)) is True

    def test_garbage_bytes_fail_open(self):
        # Unparseable input must not block transcription.
        assert recording_is_silent(b"not a wav file at all") is False

    def test_empty_recording_is_silent(self):
        assert recording_is_silent(_wav([])) is True

USAGE = SimpleNamespace(prompt_tokens=120, completion_tokens=15)


def _client(content, captured=None):
    """A fake OpenAI client returning a fixed completion text."""

    def create(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        message = SimpleNamespace(content=content)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)], usage=USAGE
        )

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


class TestTranscribeAudio:
    def test_returns_stripped_transcript_and_usage(self):
        transcript, usage = transcribe_audio(
            _client("  I led a migration project. \n"), b"RIFFdata"
        )
        assert transcript == "I led a migration project."
        assert usage is USAGE

    def test_request_shape(self):
        captured = {}
        audio = b"\x00\x01binary-audio\xff"
        transcribe_audio(_client("ok", captured), audio, audio_format="wav")

        assert captured["model"] == TRANSCRIPTION_MODEL
        assert captured["temperature"] == 0.0

        (message,) = captured["messages"]
        assert message["role"] == "user"
        text_part, audio_part = message["content"]
        assert text_part == {"type": "text", "text": TRANSCRIPTION_PROMPT}
        assert audio_part["type"] == "input_audio"
        assert audio_part["input_audio"]["format"] == "wav"
        # The recording travels base64-encoded, decodable back to the bytes.
        assert base64.b64decode(audio_part["input_audio"]["data"]) == audio

    def test_no_speech_returns_none(self):
        # The prompt asks for empty output on silence; None tells the caller
        # "nothing was said" as opposed to an empty message.
        transcript, usage = transcribe_audio(_client("   \n"), b"silence")
        assert transcript is None
        assert usage is USAGE

    def test_null_content_returns_none(self):
        transcript, _ = transcribe_audio(_client(None), b"audio")
        assert transcript is None
