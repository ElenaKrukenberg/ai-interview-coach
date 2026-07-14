"""Voice input: speech-to-text through the OpenRouter chat API. No Streamlit.

OpenRouter offers no dedicated transcription endpoint (Whisper is not in its
catalog), so the recording is sent as an `input_audio` content part to an
audio-capable chat model instead. That keeps the whole app on its single
OpenRouter key, client, and cost accounting.
"""

import base64
import io
import struct
import wave
from typing import Optional, Tuple

from openai import OpenAI

from core.config import TRANSCRIPTION_MODEL

# Peak amplitude below ~0.1% of full scale (≈ -60 dBFS) means no signal:
# even quiet speech into a working microphone peaks orders of magnitude
# higher, and a live mic's noise floor alone usually exceeds this.
SILENCE_PEAK = 32

# A chat model asked to transcribe tends to chat: add preambles, translate,
# or invent text for silence. The prompt pins it to verbatim-or-nothing.
TRANSCRIPTION_PROMPT = (
    "Transcribe the spoken answer in this audio recording verbatim. "
    "Output ONLY the transcribed words: no commentary, no quotes, no "
    "speaker labels, and no translation — keep the original language. "
    "If the recording contains no intelligible speech, output nothing at all."
)


def recording_is_silent(audio_bytes: bytes) -> bool:
    """True when a WAV recording contains no audible signal at all.

    The important real-world case: on macOS a browser without the OS-level
    microphone permission "records" pure zeros — no error anywhere, and a
    transcription model then hallucinates plausible speech out of silence.
    Catching it locally turns that into a clear hint instead of a paid,
    garbage API call.

    Anything unparseable fails open (False): when in doubt, let the model try.
    """
    try:
        with wave.open(io.BytesIO(audio_bytes)) as recording:
            if recording.getsampwidth() != 2:  # only 16-bit PCM is analyzed
                return False
            frames = recording.readframes(recording.getnframes())
        samples = struct.unpack(f"<{len(frames) // 2}h", frames)
        peak = max((abs(sample) for sample in samples), default=0)
        return peak <= SILENCE_PEAK
    except Exception:
        return False


def transcribe_audio(
    client: OpenAI,
    audio_bytes: bytes,
    audio_format: str = "wav",
) -> Tuple[Optional[str], Optional[object]]:
    """Transcribe one recording; returns (transcript, usage).

    The transcript is None when the model heard no usable speech, so callers
    can distinguish "nothing said" from an empty message.
    """
    encoded = base64.b64encode(audio_bytes).decode("ascii")
    response = client.chat.completions.create(
        model=TRANSCRIPTION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TRANSCRIPTION_PROMPT},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": encoded, "format": audio_format},
                    },
                ],
            }
        ],
        temperature=0.0,  # transcription must not be creative
    )
    transcript = (response.choices[0].message.content or "").strip()
    return (transcript or None), response.usage
