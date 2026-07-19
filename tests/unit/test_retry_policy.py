"""Unit tests for Phase 10.3 — Retry policy with JSON repair and local fallback.

Tests verify:
- Happy path: Attempt 1 is valid JSON, returns immediately with no retry.
- Retry success: Attempt 1 is invalid/incomplete, Attempt 2 succeeds.
- JSON Truncation Repair: Truncated JSON in recommendations array is repaired and salvaged.
- Local Recommendation Repair: Missing fields (like placement/ats_impact) are repaired locally.
- Propagate provider errors: Network/timeout errors propagate immediately.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.prompts import ChatPromptTemplate

from app.ai.guardrails.retry_policy import run_with_retry
from app.core.errors import AIGenerationError
from app.schemas.ai import ATSExplanation


class TestRetryPolicy:
    @pytest.fixture()
    def valid_json_dict(self) -> dict:
        return {
            "strengths": ["Strong programming skills in Python."],
            "weaknesses": ["Missing Kubernetes expertise."],
            "section_explanations": [
                {
                    "section": "Keywords",
                    "score": 75.0,
                    "explanation": "Matched 7 out of 10 keywords.",
                }
            ],
            "suggestions": ["Add Kubernetes to skills."],
            "summary": "Overall good fit.",
            "recommendations": [
                {
                    "priority": "High",
                    "issue": "Missing Kubernetes.",
                    "why": "Crucial for this role.",
                    "copy_paste_content": "Kubernetes core concepts.",
                    "placement": "Skills section.",
                    "ats_impact": "+10 points",
                }
            ],
        }

    @pytest.fixture()
    def mock_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template("dummy")

    @pytest.fixture()
    def mock_llm(self) -> MagicMock:
        return MagicMock()

    @patch("app.ai.guardrails.retry_policy.invoke_with_tracing")
    def test_happy_path_no_retry(
        self,
        mock_invoke,
        valid_json_dict: dict,
        mock_prompt: ChatPromptTemplate,
        mock_llm: MagicMock,
    ) -> None:
        """If attempt 1 succeeds, no retry should be performed and result returned."""
        mock_msg = MagicMock()
        mock_msg.content = json.dumps(valid_json_dict)
        mock_msg.response_metadata = {"finish_reason": "stop"}
        mock_invoke.return_value = mock_msg

        # Call under test
        result = run_with_retry(
            mock_prompt,
            mock_llm,
            {"jd_context": "original"},
            "resume text",
            "jd text",
        )

        assert isinstance(result, ATSExplanation)
        assert result.summary == "Overall good fit."
        assert len(result.recommendations) == 1
        assert mock_invoke.call_count == 1

    @patch("app.ai.guardrails.retry_policy.invoke_with_tracing")
    def test_retry_success_on_first_failure(
        self,
        mock_invoke,
        valid_json_dict: dict,
        mock_prompt: ChatPromptTemplate,
        mock_llm: MagicMock,
    ) -> None:
        """If attempt 1 fails JSON decoding but attempt 2 succeeds, return valid response."""
        mock_msg_fail = MagicMock()
        mock_msg_fail.content = "{ invalid json..."
        mock_msg_fail.response_metadata = {"finish_reason": "length"}

        mock_msg_success = MagicMock()
        mock_msg_success.content = json.dumps(valid_json_dict)
        mock_msg_success.response_metadata = {"finish_reason": "stop"}

        mock_invoke.side_effect = [mock_msg_fail, mock_msg_success]

        result = run_with_retry(
            mock_prompt,
            mock_llm,
            {"jd_context": "original"},
            "resume text",
            "jd text",
        )

        assert isinstance(result, ATSExplanation)
        assert result.summary == "Overall good fit."
        assert mock_invoke.call_count == 2

    @patch("app.ai.guardrails.retry_policy.invoke_with_tracing")
    def test_all_attempts_fail_raises_error(
        self,
        mock_invoke,
        mock_prompt: ChatPromptTemplate,
        mock_llm: MagicMock,
    ) -> None:
        """If all attempts (2) fail, raise AIGenerationError."""
        mock_msg = MagicMock()
        mock_msg.content = "{ invalid json..."
        mock_msg.response_metadata = {"finish_reason": "length"}
        mock_invoke.return_value = mock_msg

        with pytest.raises(AIGenerationError, match="AI explanation could not be generated"):
            run_with_retry(
                mock_prompt,
                mock_llm,
                {"jd_context": "original"},
                "resume text",
                "jd text",
            )

        assert mock_invoke.call_count == 2

    @patch("app.ai.guardrails.retry_policy.invoke_with_tracing")
    def test_truncated_json_repair_salvages_complete_items(
        self,
        mock_invoke,
        valid_json_dict: dict,
        mock_prompt: ChatPromptTemplate,
        mock_llm: MagicMock,
    ) -> None:
        """Verify that truncated JSON gets repaired and parsed successfully."""
        # Create a truncated JSON string ending in the middle of recommendations
        valid_json_dict["recommendations"] = [
            {
                "priority": "High",
                "issue": "Missing Kubernetes.",
                "why": "Crucial for this role.",
                "copy_paste_content": "Kubernetes core concepts.",
                "placement": "Skills section.",
                "ats_impact": "+10 points",
            }
        ]
        raw_json = json.dumps(valid_json_dict)
        # Cut it off in the middle of a second recommendation
        truncated_json = (
            raw_json[:-2]
            + ', {"priority": "Medium", "issue": "Missing Docker", "why": "Need containers'
        )

        mock_msg = MagicMock()
        mock_msg.content = truncated_json
        mock_msg.response_metadata = {"finish_reason": "length"}
        mock_invoke.return_value = mock_msg

        result = run_with_retry(
            mock_prompt,
            mock_llm,
            {"jd_context": "original"},
            "resume text",
            "jd text",
        )

        assert isinstance(result, ATSExplanation)
        # Should have salvaged the first recommendation and ignored the incomplete second one
        assert len(result.recommendations) == 1
        assert result.recommendations[0].issue == "Missing Kubernetes."

    @patch("app.ai.guardrails.retry_policy.invoke_with_tracing")
    def test_incomplete_recommendation_local_repair(
        self,
        mock_invoke,
        valid_json_dict: dict,
        mock_prompt: ChatPromptTemplate,
        mock_llm: MagicMock,
    ) -> None:
        """If a recommendation is missing required fields, it is repaired locally (no LLM call)."""
        # First recommendation is missing placement and ats_impact
        valid_json_dict["recommendations"] = [
            {
                "priority": "High",
                "issue": "Missing Kubernetes.",
                "why": "Crucial for this role.",
                "copy_paste_content": "Kubernetes core.",
                "placement": "",
                "ats_impact": "",
            }
        ]

        mock_msg_main = MagicMock()
        mock_msg_main.content = json.dumps(valid_json_dict)
        mock_msg_main.response_metadata = {"finish_reason": "stop"}
        mock_invoke.return_value = mock_msg_main

        result = run_with_retry(
            mock_prompt,
            mock_llm,
            {"jd_context": "original"},
            "resume text",
            "jd text",
        )

        assert isinstance(result, ATSExplanation)
        assert len(result.recommendations) == 1
        # Should be repaired locally to fallback values
        assert result.recommendations[0].placement == "In the relevant section."
        assert result.recommendations[0].ats_impact == "+5 points"
        # Only one LLM call should be made (no regeneration LLM calls!)
        assert mock_invoke.call_count == 1

    @patch("app.ai.guardrails.retry_policy.invoke_with_tracing")
    def test_propagate_provider_errors_immediately(
        self,
        mock_invoke,
        mock_prompt: ChatPromptTemplate,
        mock_llm: MagicMock,
    ) -> None:
        """Verify that provider timeouts, rate limits, or network failures propagate immediately."""
        mock_invoke.side_effect = RuntimeError("LiteLLM Rate Limit / Timeout")

        with pytest.raises(RuntimeError, match="LiteLLM Rate Limit / Timeout"):
            run_with_retry(
                mock_prompt,
                mock_llm,
                {"jd_context": "original"},
                "resume text",
                "jd text",
            )

        # Should fail fast and exit after exactly 1 call (no retries for provider exceptions)
        assert mock_invoke.call_count == 1
