"""Tests for prompts.build_system_prompt — the single prompt source of truth."""

import pytest

from prompts import (
    DEFAULT_TARGET_ROLE,
    INTERVIEW_TYPES,
    PROMPT_TECHNIQUES,
    UNTRUSTED_DATA_NOTICE,
    build_system_prompt,
    wrap_untrusted,
)

ANY_TECHNIQUE = next(iter(PROMPT_TECHNIQUES))
ANY_TYPE = next(iter(INTERVIEW_TYPES))


class TestUnknownSelections:
    def test_unknown_technique_raises(self):
        with pytest.raises(ValueError, match="Unknown prompt technique"):
            build_system_prompt("Nonexistent Technique", ANY_TYPE)

    def test_unknown_interview_type_raises(self):
        with pytest.raises(ValueError, match="Unknown interview type"):
            build_system_prompt(ANY_TECHNIQUE, "Nonexistent Type")

    @pytest.mark.parametrize("technique", list(PROMPT_TECHNIQUES))
    @pytest.mark.parametrize("interview_type", list(INTERVIEW_TYPES))
    def test_every_known_combination_builds(self, technique, interview_type):
        prompt = build_system_prompt(technique, interview_type)
        assert prompt.strip()


class TestRoleHandling:
    def test_empty_role_falls_back_to_default(self):
        prompt = build_system_prompt(ANY_TECHNIQUE, ANY_TYPE, target_role="")
        assert DEFAULT_TARGET_ROLE in prompt

    def test_whitespace_role_falls_back_to_default(self):
        prompt = build_system_prompt(ANY_TECHNIQUE, ANY_TYPE, target_role="   ")
        assert DEFAULT_TARGET_ROLE in prompt

    def test_explicit_role_is_used(self):
        prompt = build_system_prompt(ANY_TECHNIQUE, ANY_TYPE, target_role="Data Engineer")
        assert "Data Engineer" in prompt
        assert DEFAULT_TARGET_ROLE not in prompt


class TestPromptComposition:
    def test_company_block_included_when_provided(self):
        prompt = build_system_prompt(ANY_TECHNIQUE, ANY_TYPE, company_name="Spotify")
        assert "interviewing at Spotify" in prompt

    def test_interview_type_block_is_appended(self):
        prompt = build_system_prompt(ANY_TECHNIQUE, ANY_TYPE)
        assert INTERVIEW_TYPES[ANY_TYPE] in prompt

    def test_job_description_goes_into_tagged_block(self):
        jd = "We need Kubernetes and Terraform experience."
        prompt = build_system_prompt(ANY_TECHNIQUE, ANY_TYPE, job_description=jd)
        assert f"<JOB_DESCRIPTION>\n{jd}\n</JOB_DESCRIPTION>" in prompt
        assert UNTRUSTED_DATA_NOTICE in prompt

    def test_resume_summary_goes_into_tagged_block(self):
        summary = "5 years of Python, led a team of 3."
        prompt = build_system_prompt(ANY_TECHNIQUE, ANY_TYPE, resume_summary=summary)
        assert f"<RESUME_SUMMARY>\n{summary}\n</RESUME_SUMMARY>" in prompt
        assert UNTRUSTED_DATA_NOTICE in prompt

    def test_untrusted_notice_appears_once_for_both_documents(self):
        prompt = build_system_prompt(
            ANY_TECHNIQUE,
            ANY_TYPE,
            job_description="A job posting.",
            resume_summary="A resume summary.",
        )
        assert prompt.count(UNTRUSTED_DATA_NOTICE) == 1

    def test_no_notice_without_documents(self):
        prompt = build_system_prompt(ANY_TECHNIQUE, ANY_TYPE)
        assert UNTRUSTED_DATA_NOTICE not in prompt


def test_wrap_untrusted():
    assert wrap_untrusted("DOC", "text") == "<DOC>\ntext\n</DOC>"
