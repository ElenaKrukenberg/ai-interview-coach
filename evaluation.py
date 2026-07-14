"""Structured JSON outputs: feedback, LLM-as-a-judge rubric, and JD analysis.

Implements the project's structured-output formats:
1. Interview feedback report (strengths / weaknesses / suggestions).
2. Per-answer rubric scoring via LLM-as-a-judge (a second, independent
   LLM call with a dedicated judge prompt and low temperature).
3. Job description analysis (key skills, likely topics, study plan).
"""

import json
from typing import Callable, Dict, List, Optional, Tuple

from openai import OpenAI

from prompts import UNTRUSTED_DATA_NOTICE, wrap_untrusted

# JSON output format #1: overall interview feedback.
FEEDBACK_SYSTEM_PROMPT = """You are an expert interview coach analyzing a mock-interview \
transcript for a candidate applying to the position of {target_role}.

Respond ONLY with valid JSON matching exactly this schema:
{{
  "overall_score": <integer 1-10>,
  "strengths": ["<short bullet>", ...],
  "weaknesses": ["<short bullet>", ...],
  "suggestions": ["<short actionable bullet>", ...],
  "summary": "<2-3 sentence overall assessment>"
}}

Base your analysis only on what the candidate actually said in the transcript. \
If the transcript is too short to judge, say so in the summary and score conservatively."""

# JSON output format #2: evaluate the quality of the interviewer's performance.
JUDGE_SYSTEM_PROMPT = """You are an expert evaluating the quality of a mock INTERVIEW SESSION \
for the position of {target_role}. Assess the INTERVIEWER's performance, not the candidate's answers. \
Judge whether the interviewer asked good questions, provided fair feedback, and conducted a \
professional interview.

Respond ONLY with valid JSON matching exactly this schema:
{{
  "rubric": {{
    "question_quality":   {{"score": <integer 1-5>, "rationale": "<one sentence>"}},
    "fairness":           {{"score": <integer 1-5>, "rationale": "<one sentence>"}},
    "topic_coverage":     {{"score": <integer 1-5>, "rationale": "<one sentence>"}},
    "feedback_quality":   {{"score": <integer 1-5>, "rationale": "<one sentence>"}}
  }},
  "average_score": <float>,
  "verdict": "<one-sentence overall assessment of the interview quality>"
}}"""


# One-time resume compression: the summary (not the full resume) is injected
# into every chat system prompt, keeping per-message cost and noise low.
RESUME_SUMMARY_PROMPT = """You are a resume parser. Compress the resume below into a compact \
candidate profile of at most 12 short lines: key skills, years and areas of experience, \
notable achievements with metrics, education if relevant. Plain text, no commentary, \
no markdown headers. Preserve concrete technology names and numbers."""

# JSON output format #4: resume-to-job match report.
MATCH_SYSTEM_PROMPT = """You are a career coach assessing how well a candidate's resume matches \
a job posting for the position of {target_role}.

Respond ONLY with valid JSON matching exactly this schema:
{{
  "match_score": <integer 0-100>,
  "matching_strengths": ["<skill or experience from the resume that fits the posting>", ...],
  "gaps": ["<requirement from the posting that the resume does not demonstrate>", ...],
  "improvement_plan": ["<short prioritized step to close a gap>", ...],
  "verdict": "<one-sentence honest overall assessment>"
}}

Guidelines:
- Be honest: base the score on real overlap, not politeness.
- List concrete strengths and gaps, quoting skills by name.
- Keep the improvement plan short (3-5 steps), ordered by impact."""

# JSON output format #3: job description analysis.
ANALYSIS_SYSTEM_PROMPT = """You are a career coach analyzing a job posting for a candidate \
preparing to interview for the position of {target_role}.

Respond ONLY with valid JSON matching exactly this schema:
{{
  "key_skills": {{
    "hard": ["<technical skill>", ...],
    "soft": ["<soft skill>", ...]
  }},
  "interview_topics": ["<topic the interview is likely to cover>", ...],
  "study_plan": ["<short prioritized step, e.g. 'Day 1-2: revise SQL window functions'>", ...]
}}

Guidelines:
- Extract only skills actually mentioned or clearly implied by the posting.
- List 4-8 likely interview topics, most probable first.
- Keep the study plan short (3-5 steps) and actionable, ordered by priority."""


def _job_description_block(job_description: str) -> str:
    """Optional system-prompt addition grounding the evaluation in the job posting."""
    job_description = (job_description or "").strip()
    if not job_description:
        return ""
    return (
        f"\n\n{UNTRUSTED_DATA_NOTICE}\n\n"
        "The candidate is applying to the following job posting. "
        "Check their answers against its requirements and call out any "
        "required skills they failed to demonstrate:\n"
        + wrap_untrusted("JOB_DESCRIPTION", job_description)
    )


_SPEAKER_LABELS = {"user": "Candidate", "assistant": "Interviewer"}


def format_transcript(messages: List[Dict[str, str]]) -> str:
    """Render the chat history as a plain-text transcript for evaluation."""
    lines = []
    for message in messages:
        role = message.get("role") or "unknown"
        speaker = _SPEAKER_LABELS.get(role, role.capitalize())
        lines.append(f"{speaker}: {message['content']}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Structural validation of model JSON
#
# `response_format={"type": "json_object"}` only guarantees syntactically
# valid JSON; nothing stops the model from returning the wrong shape. Each
# report format gets a validator, so callers receive either a dict matching
# the schema they prompted for, or None — never a surprise shape that blows
# up in the UI.
# ---------------------------------------------------------------------------


def _is_str_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_feedback(data: dict) -> bool:
    """Schema check for the interview feedback report (format #1)."""
    return (
        _is_number(data.get("overall_score"))
        and 0 <= data["overall_score"] <= 10
        and _is_str_list(data.get("strengths"))
        and _is_str_list(data.get("weaknesses"))
        and _is_str_list(data.get("suggestions"))
        and isinstance(data.get("summary"), str)
    )


def validate_judgment(data: dict) -> bool:
    """Schema check for the LLM-as-a-judge rubric (format #2)."""
    rubric = data.get("rubric")
    if not isinstance(rubric, dict) or not rubric:
        return False
    for details in rubric.values():
        if not isinstance(details, dict):
            return False
        if not (_is_number(details.get("score")) and 0 <= details["score"] <= 5):
            return False
        if not isinstance(details.get("rationale", ""), str):
            return False
    # average_score and verdict are rendered when present, so their types
    # must be right when they exist; their absence is tolerated.
    if "average_score" in data and not (
        data["average_score"] is None or _is_number(data["average_score"])
    ):
        return False
    if "verdict" in data and not isinstance(data["verdict"], str):
        return False
    return True


def validate_analysis(data: dict) -> bool:
    """Schema check for the job description analysis (format #3)."""
    skills = data.get("key_skills")
    return (
        isinstance(skills, dict)
        and _is_str_list(skills.get("hard"))
        and _is_str_list(skills.get("soft"))
        and _is_str_list(data.get("interview_topics"))
        and _is_str_list(data.get("study_plan"))
    )


def validate_match(data: dict) -> bool:
    """Schema check for the resume-to-job match report (format #4)."""
    return (
        _is_number(data.get("match_score"))
        and 0 <= data["match_score"] <= 100
        and _is_str_list(data.get("matching_strengths"))
        and _is_str_list(data.get("gaps"))
        and _is_str_list(data.get("improvement_plan"))
        and isinstance(data.get("verdict"), str)
    )


def _call_json_completion(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_content: str,
    validator: Optional[Callable[[dict], bool]] = None,
) -> Tuple[Optional[dict], Optional[object]]:
    """
    Make a non-streaming completion that must return JSON.

    Returns (parsed_json, usage). parsed_json is None if the model's output
    could not be parsed as a JSON object, or if it fails the structural
    `validator` for the expected report schema.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,  # low temperature for consistent, deterministic scoring
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or ""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None, response.usage
    if not isinstance(data, dict):
        return None, response.usage
    if validator is not None and not validator(data):
        return None, response.usage
    return data, response.usage


def get_interview_feedback(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    target_role: str,
    job_description: str = "",
) -> Tuple[Optional[dict], Optional[object]]:
    """Generate a structured JSON feedback report for the whole conversation."""
    system_prompt = FEEDBACK_SYSTEM_PROMPT.format(target_role=target_role)
    system_prompt += _job_description_block(job_description)
    transcript = wrap_untrusted("TRANSCRIPT", format_transcript(messages))
    return _call_json_completion(
        client, model, system_prompt, transcript, validator=validate_feedback
    )


def judge_answer(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    target_role: str,
    job_description: str = "",
) -> Tuple[Optional[dict], Optional[object]]:
    """Score the candidate's most recent answer with an LLM-as-a-judge rubric."""
    system_prompt = JUDGE_SYSTEM_PROMPT.format(target_role=target_role)
    system_prompt += _job_description_block(job_description)
    transcript = wrap_untrusted("TRANSCRIPT", format_transcript(messages))
    return _call_json_completion(
        client, model, system_prompt, transcript, validator=validate_judgment
    )


def analyse_job_description(
    client: OpenAI,
    model: str,
    job_description: str,
    target_role: str,
) -> Tuple[Optional[dict], Optional[object]]:
    """Extract key skills, likely interview topics, and a study plan from a posting."""
    system_prompt = ANALYSIS_SYSTEM_PROMPT.format(target_role=target_role)
    system_prompt += "\n\n" + UNTRUSTED_DATA_NOTICE
    user_content = wrap_untrusted("JOB_DESCRIPTION", job_description)
    return _call_json_completion(
        client, model, system_prompt, user_content, validator=validate_analysis
    )


def summarize_resume(
    client: OpenAI,
    model: str,
    resume_text: str,
) -> Tuple[Optional[str], Optional[object]]:
    """Compress the full resume into a short profile for reuse in chat prompts."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": RESUME_SUMMARY_PROMPT + "\n\n" + UNTRUSTED_DATA_NOTICE,
            },
            {"role": "user", "content": wrap_untrusted("RESUME", resume_text)},
        ],
        temperature=0.2,
    )
    summary = (response.choices[0].message.content or "").strip()
    return (summary or None), response.usage


def match_resume(
    client: OpenAI,
    model: str,
    resume_text: str,
    job_description: str,
    target_role: str,
) -> Tuple[Optional[dict], Optional[object]]:
    """Score how well the full resume matches the job posting."""
    system_prompt = MATCH_SYSTEM_PROMPT.format(target_role=target_role)
    system_prompt += "\n\n" + UNTRUSTED_DATA_NOTICE
    user_content = (
        wrap_untrusted("RESUME", resume_text)
        + "\n\n"
        + wrap_untrusted("JOB_POSTING", job_description)
    )
    return _call_json_completion(
        client, model, system_prompt, user_content, validator=validate_match
    )
