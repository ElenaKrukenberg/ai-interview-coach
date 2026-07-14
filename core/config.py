"""Application-wide constants and settings types. No Streamlit imports."""

from dataclasses import dataclass

APP_TITLE = "AI Interview Preparation Assistant"

MODEL_OPTIONS = [
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "google/gemini-2.5-flash",
]

def get_judge_model(interviewer_model: str) -> str:
    """Select judge model from a different family than the interviewer.

    If interviewer is OpenAI → judge is Google.
    If interviewer is Google → judge is OpenAI.
    """
    if interviewer_model.startswith("google/"):
        return "openai/gpt-5-mini"  # OpenAI judge for Google interviewer
    return "google/gemini-2.5-flash"  # Google judge for OpenAI interviewer

# Fixed speech-to-text model for voice input. OpenRouter has no Whisper-style
# transcription endpoint, so voice notes go to an audio-capable chat model —
# the pragmatic choice that keeps the app on one provider and one API key.
TRANSCRIPTION_MODEL = "google/gemini-2.5-flash"

# Quick buttons: (label, the message sent as if the user typed it).
# QUICK_STARTS opens an empty chat; QUICK_ACTIONS keeps a running one moving.
QUICK_STARTS = [
    ("Ask me the first question", "Ask me the first interview question."),
    (
        "Warm-up: about me",
        "Ask me to introduce myself, then give feedback on my self-introduction.",
    ),
    (
        "Quiz me on key skills",
        "Quiz me on the key skills for this role, one question at a time.",
    ),
]

QUICK_ACTIONS = [
    ("Next question", "Ask me the next interview question."),
    (
        "Give me a hint",
        "Give me a hint for the current question, but don't reveal the full answer.",
    ),
    (
        "Make it harder",
        "Ask a harder follow-up question on the same topic.",
    ),
]


@dataclass
class ModelSettings:
    """Developer-tunable model configuration collected from the sidebar."""

    model: str
    prompt_technique: str
    temperature: float
    max_tokens: int
