"""Unit tests for the explain-score chain.

Strategy
--------
* No real LLM calls — the model is replaced with a mock through ``invoke_with_tracing``.
* Validates that ``build_chain_inputs()`` populates every prompt variable.
* Validates that the chain parses LLM JSON output into ``ATSExplanation``.
* Validates that ``run_explain_score`` works end-to-end with the mock.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.ai.chains.explain_score_chain import build_chain_inputs, run_explain_score
from app.schemas.ai import ATSExplanation, SectionExplanation, RecommendationSchema
from app.schemas.ats import (
    ATSAnalyzeResponse,
    EducationSummarySchema,
    ExperienceSummarySchema,
    MatchedKeywordSchema,
)

# ---------------------------------------------------------------------------
# Fixture — minimal but valid ATSAnalyzeResponse
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_response() -> ATSAnalyzeResponse:
    """A valid ATSAnalyzeResponse used across tests."""
    return ATSAnalyzeResponse(
        overall_score=62.5,
        keyword_score=55.0,
        experience_score=70.0,
        skills_score=60.0,
        education_score=80.0,
        summary_score=40.0,
        formatting_score=75.0,
        matched_keywords=[
            MatchedKeywordSchema(keyword="python", matchType="EXACT"),
            MatchedKeywordSchema(keyword="docker", matchType="SYNONYM"),
        ],
        missing_keywords=["kubernetes", "terraform"],
        matched_skills=["Python", "Docker"],
        missing_skills=["Kubernetes"],
        experience_summary=ExperienceSummarySchema(
            total_entries=2,
            total_years=3.5,
            has_metrics=True,
        ),
        education_summary=EducationSummarySchema(
            highest_degree="Bachelor of Science",
            certifications=["AWS Certified"],
        ),
        processing_time_ms=12.34,
    )


_SAMPLE_JD = (
    "Looking for a Python engineer with Docker and Kubernetes skills. "
    "AWS experience required."
)

# ---------------------------------------------------------------------------
# build_chain_inputs tests
# ---------------------------------------------------------------------------


class TestBuildChainInputs:
    def test_returns_dict(self, sample_response: ATSAnalyzeResponse) -> None:
        result = build_chain_inputs(sample_response, _SAMPLE_JD, "Sample Resume Text")
        assert isinstance(result, dict)

    def test_all_score_fields_present(self, sample_response: ATSAnalyzeResponse) -> None:
        result = build_chain_inputs(sample_response, _SAMPLE_JD, "Sample Resume Text")
        for field in (
            "overall_score",
            "keyword_score",
            "experience_score",
            "skills_score",
            "education_score",
            "summary_score",
            "formatting_score",
        ):
            assert field in result, f"Missing field: {field}"

    def test_scores_are_rounded(self, sample_response: ATSAnalyzeResponse) -> None:
        result = build_chain_inputs(sample_response, _SAMPLE_JD, "Sample Resume Text")
        assert result["overall_score"] == 62.5

    def test_matched_keywords_joined(self, sample_response: ATSAnalyzeResponse) -> None:
        result = build_chain_inputs(sample_response, _SAMPLE_JD, "Sample Resume Text")
        assert "python" in str(result["matched_keywords"])
        assert "docker" in str(result["matched_keywords"])

    def test_missing_keywords_joined(self, sample_response: ATSAnalyzeResponse) -> None:
        result = build_chain_inputs(sample_response, _SAMPLE_JD, "Sample Resume Text")
        assert "kubernetes" in str(result["missing_keywords"])

    def test_jd_context_preprocessed(self, sample_response: ATSAnalyzeResponse) -> None:
        long_jd = "### Requirements:\nPython development\nKubernetes orchestration"
        result = build_chain_inputs(sample_response, long_jd, "Sample Resume Text")
        assert "Required Skills & Experience" in str(result["jd_context"])
        assert "Python development" in str(result["jd_context"])
        assert "Kubernetes orchestration" in str(result["jd_context"])

    def test_empty_missing_keywords_shows_none_string(
        self, sample_response: ATSAnalyzeResponse
    ) -> None:
        sample_response.missing_keywords = []
        result = build_chain_inputs(sample_response, _SAMPLE_JD, "Sample Resume Text")
        assert result["missing_keywords"] == "None"

    def test_certifications_joined(self, sample_response: ATSAnalyzeResponse) -> None:
        result = build_chain_inputs(sample_response, _SAMPLE_JD, "Sample Resume Text")
        assert "AWS Certified" in str(result["edu_certifications"])

    def test_experience_fields_present(self, sample_response: ATSAnalyzeResponse) -> None:
        result = build_chain_inputs(sample_response, _SAMPLE_JD, "Sample Resume Text")
        assert result["exp_total_entries"] == 2
        assert result["exp_total_years"] == 3.5
        assert result["exp_has_metrics"] is True


# ---------------------------------------------------------------------------
# run_explain_score — mock the LLM
# ---------------------------------------------------------------------------

_FAKE_EXPLANATION_DICT = {
    "strengths": ["Strong Python skills matched the JD."],
    "weaknesses": ["Missing Kubernetes experience."],
    "section_explanations": [
        {
            "section": "Keywords",
            "score": 55.0,
            "explanation": "Resume matched 2 of 5 required keywords.",
        }
    ],
    "suggestions": ["Add Kubernetes to your skills section."],
    "summary": "The resume scored 62.5/100 overall against the job description.",
    "recommendations": [
        {
            "priority": "High",
            "issue": "Missing professional summary.",
            "why": "A summary hooks the recruiter.",
            "copy_paste_content": "Motivated engineer.",
            "placement": "Top of resume.",
            "ats_impact": "+10 points"
        }
    ],
}


class TestRunExplainScore:
    @patch("app.ai.chains.explain_score_chain.invoke_with_tracing")
    def test_returns_ats_explanation_when_chain_succeeds(
        self, mock_invoke, sample_response: ATSAnalyzeResponse
    ) -> None:
        """When the chain returns ATSExplanation, run_explain_score returns it."""
        mock_msg = MagicMock()
        mock_msg.content = json.dumps(_FAKE_EXPLANATION_DICT)
        mock_invoke.return_value = mock_msg
        mock_llm = MagicMock()

        with patch("app.ai.chains.explain_score_chain.get_chat_model", return_value=mock_llm):
            result = run_explain_score(sample_response, _SAMPLE_JD)

        assert isinstance(result, ATSExplanation)
        assert result.summary == _FAKE_EXPLANATION_DICT["summary"]
        mock_invoke.assert_called_once()

    @patch("app.ai.chains.explain_score_chain.invoke_with_tracing")
    def test_chain_invoked_with_correct_input_keys(
        self, mock_invoke, sample_response: ATSAnalyzeResponse
    ) -> None:
        """The chain's invoke receives all required prompt template variables."""
        mock_msg = MagicMock()
        mock_msg.content = json.dumps(_FAKE_EXPLANATION_DICT)
        mock_invoke.return_value = mock_msg
        mock_llm = MagicMock()

        with patch("app.ai.chains.explain_score_chain.get_chat_model", return_value=mock_llm):
            run_explain_score(sample_response, _SAMPLE_JD)

        call_args = mock_invoke.call_args[0][1]
        assert "overall_score" in call_args
        assert "missing_keywords" in call_args
        assert "jd_context" in call_args

    @patch("app.ai.chains.explain_score_chain.invoke_with_tracing")
    def test_raises_on_llm_error(self, mock_invoke, sample_response: ATSAnalyzeResponse) -> None:
        """If the chain raises, run_explain_score propagates the exception."""
        mock_invoke.side_effect = RuntimeError("LLM unavailable")
        mock_llm = MagicMock()

        with patch("app.ai.chains.explain_score_chain.get_chat_model", return_value=mock_llm):
            with pytest.raises(RuntimeError, match="LLM unavailable"):
                run_explain_score(sample_response, _SAMPLE_JD)
