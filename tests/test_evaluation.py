"""Tests for evaluation.py: transcript formatting, JSON parsing, and the
structural validators guarding every report schema."""

import json
from types import SimpleNamespace

import pytest

from evaluation import (
    _call_json_completion,
    format_transcript,
    get_interview_feedback,
    judge_answer,
    validate_analysis,
    validate_feedback,
    validate_judgment,
    validate_match,
)


# ---------------------------------------------------------------------------
# Fake OpenAI client: returns a canned completion and records the request.
# ---------------------------------------------------------------------------


class FakeClient:
    def __init__(self, content: str):
        self.last_request = None
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.last_request = kwargs
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content=content))
                    ],
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                )

        self.chat = SimpleNamespace(completions=_Completions())


# ---------------------------------------------------------------------------
# format_transcript
# ---------------------------------------------------------------------------


class TestFormatTranscript:
    def test_known_roles_are_labeled(self):
        transcript = format_transcript(
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Tell me about yourself."},
            ]
        )
        assert transcript == "Candidate: Hi\n\nInterviewer: Tell me about yourself."

    def test_unknown_role_falls_back_to_capitalized_role(self):
        transcript = format_transcript([{"role": "system", "content": "x"}])
        assert transcript == "System: x"

    def test_missing_role_does_not_crash(self):
        transcript = format_transcript([{"role": "", "content": "x"}])
        assert transcript == "Unknown: x"

    def test_empty_history(self):
        assert format_transcript([]) == ""


# ---------------------------------------------------------------------------
# _call_json_completion: parsing + validation outcomes
# ---------------------------------------------------------------------------


class TestJsonCompletion:
    def test_invalid_json_returns_none_but_keeps_usage(self):
        client = FakeClient("this is not json {")
        data, usage = _call_json_completion(client, "m", "sys", "user")
        assert data is None
        assert usage.prompt_tokens == 10

    def test_valid_json_non_object_returns_none(self):
        client = FakeClient(json.dumps(["a", "list"]))
        data, _ = _call_json_completion(client, "m", "sys", "user")
        assert data is None

    def test_valid_json_failing_validator_returns_none(self):
        client = FakeClient(json.dumps({"unexpected": "shape"}))
        data, _ = _call_json_completion(
            client, "m", "sys", "user", validator=validate_feedback
        )
        assert data is None

    def test_valid_json_passing_validator_is_returned(self):
        payload = {
            "overall_score": 7,
            "strengths": ["clear"],
            "weaknesses": ["short"],
            "suggestions": ["use STAR"],
            "summary": "Decent.",
        }
        client = FakeClient(json.dumps(payload))
        data, _ = _call_json_completion(
            client, "m", "sys", "user", validator=validate_feedback
        )
        assert data == payload

    def test_no_validator_only_checks_object_shape(self):
        client = FakeClient(json.dumps({"anything": 1}))
        data, _ = _call_json_completion(client, "m", "sys", "user")
        assert data == {"anything": 1}


# ---------------------------------------------------------------------------
# Validators, schema by schema
# ---------------------------------------------------------------------------

VALID_FEEDBACK = {
    "overall_score": 7,
    "strengths": ["a"],
    "weaknesses": ["b"],
    "suggestions": ["c"],
    "summary": "ok",
}

VALID_JUDGMENT = {
    "rubric": {
        "question_quality": {"score": 4, "rationale": "good questions"},
        "fairness": {"score": 3, "rationale": "somewhat fair"},
        "topic_coverage": {"score": 4, "rationale": "covered key topics"},
        "feedback_quality": {"score": 3, "rationale": "adequate feedback"},
    },
    "average_score": 3.5,
    "verdict": "decent interview",
}

VALID_ANALYSIS = {
    "key_skills": {"hard": ["python"], "soft": ["communication"]},
    "interview_topics": ["testing"],
    "study_plan": ["revise SQL"],
}

VALID_MATCH = {
    "match_score": 72,
    "matching_strengths": ["python"],
    "gaps": ["kubernetes"],
    "improvement_plan": ["learn k8s"],
    "verdict": "good fit",
}


class TestValidators:
    def test_valid_payloads_pass(self):
        assert validate_feedback(VALID_FEEDBACK)
        assert validate_judgment(VALID_JUDGMENT)
        assert validate_analysis(VALID_ANALYSIS)
        assert validate_match(VALID_MATCH)

    @pytest.mark.parametrize(
        "mutation",
        [
            {"overall_score": "seven"},
            {"overall_score": 15},
            {"overall_score": True},
            {"strengths": "not a list"},
            {"strengths": [1, 2]},
            {"summary": None},
        ],
    )
    def test_feedback_rejects_wrong_shapes(self, mutation):
        assert not validate_feedback({**VALID_FEEDBACK, **mutation})

    @pytest.mark.parametrize(
        "mutation",
        [
            {"rubric": {}},
            {"rubric": "not a dict"},
            {"rubric": {"clarity": "not a dict"}},
            {"rubric": {"clarity": {"score": "four", "rationale": "x"}}},
            {"rubric": {"clarity": {"score": 9, "rationale": "x"}}},
            {"average_score": "3.5"},
            {"verdict": 42},
        ],
    )
    def test_judgment_rejects_wrong_shapes(self, mutation):
        assert not validate_judgment({**VALID_JUDGMENT, **mutation})

    def test_judgment_tolerates_missing_optional_fields(self):
        data = {"rubric": VALID_JUDGMENT["rubric"]}
        assert validate_judgment(data)

    @pytest.mark.parametrize(
        "mutation",
        [
            {"key_skills": None},
            {"key_skills": {"hard": "python", "soft": []}},
            {"interview_topics": None},
            {"study_plan": [{"step": 1}]},
        ],
    )
    def test_analysis_rejects_wrong_shapes(self, mutation):
        assert not validate_analysis({**VALID_ANALYSIS, **mutation})

    @pytest.mark.parametrize(
        "mutation",
        [
            {"match_score": "72"},
            {"match_score": 150},
            {"gaps": None},
            {"verdict": None},
        ],
    )
    def test_match_rejects_wrong_shapes(self, mutation):
        assert not validate_match({**VALID_MATCH, **mutation})


# ---------------------------------------------------------------------------
# End-to-end through the public functions: prompt contents and judge input
# ---------------------------------------------------------------------------


class TestEvaluationCalls:
    MESSAGES = [
        {"role": "user", "content": "Ask me a question."},
        {"role": "assistant", "content": "Why this role?"},
        {"role": "user", "content": "Because I love data."},
    ]

    def test_judge_evaluates_interviewer_from_full_transcript(self):
        client = FakeClient(json.dumps(VALID_JUDGMENT))
        data, _ = judge_answer(client, "m", self.MESSAGES, target_role="DS")
        assert data == VALID_JUDGMENT

        sent = client.last_request["messages"]
        user_content = sent[1]["content"]
        # The judge evaluates the interview quality from the full transcript.
        assert "Because I love data." in user_content
        assert user_content.strip().endswith("</TRANSCRIPT>")
        assert user_content.index("Why this role?") < user_content.index(
            "Because I love data."
        )
        # The judge prompt itself targets interviewer evaluation, not candidate scoring.
        assert "INTERVIEWER" in sent[0]["content"]

    def test_feedback_grounds_in_job_description_as_untrusted(self):
        client = FakeClient(json.dumps(VALID_FEEDBACK))
        get_interview_feedback(
            client, "m", self.MESSAGES, target_role="DS", job_description="Needs SQL."
        )
        system_content = client.last_request["messages"][0]["content"]
        assert "<JOB_DESCRIPTION>\nNeeds SQL.\n</JOB_DESCRIPTION>" in system_content
        assert "untrusted" in system_content

    def test_feedback_rejects_wrong_schema_from_model(self):
        client = FakeClient(json.dumps({"score": "great"}))
        data, usage = get_interview_feedback(
            client, "m", self.MESSAGES, target_role="DS"
        )
        assert data is None
        assert usage is not None
