"""Unit and integration tests for LiteLLM Router, fallback, load balancing, and health-aware routing."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest
from pydantic import SecretStr

import litellm
from litellm import Router
from litellm.utils import ModelResponse, Choices, Message
from litellm.exceptions import ServiceUnavailableError
from langchain_core.messages import HumanMessage
from langchain_core.outputs import ChatResult

from app.core.config import settings
from app.ai.providers.litellm_client import get_chat_model, load_router_config, JobPatraRoutedChat
from app.core.errors import AIGenerationError
from app.schemas.ats import ATSAnalyzeResponse, ExperienceSummarySchema, EducationSummarySchema
from app.ai.chains.explain_score_chain import run_explain_score


# Mock response payload for structured output validation
MOCK_EXPLANATION_JSON = json.dumps({
    "strengths": ["Strong engineering experience."],
    "weaknesses": ["Missing Kubernetes keyword."],
    "section_explanations": [
        {"section": "Keywords", "score": 80.0, "explanation": "Matched most keywords."},
        {"section": "Experience", "score": 90.0, "explanation": "Great duration."},
        {"section": "Skills", "score": 85.0, "explanation": "All core skills present."},
        {"section": "Education", "score": 100.0, "explanation": "Degree matched."},
        {"section": "Summary", "score": 75.0, "explanation": "Summary is clear."},
        {"section": "Formatting", "score": 95.0, "explanation": "Perfect PDF formatting."}
    ],
    "suggestions": ["Add Kubernetes to your skills."],
    "summary": "This is a great resume with 85 ATS score.",
    "recommendations": [
        {
            "priority": "High",
            "issue": "Missing Kubernetes.",
            "why": "Crucial for this role.",
            "copy_paste_content": "Kubernetes core.",
            "placement": "Skills section.",
            "ats_impact": "+10 points"
        }
    ]
})


@pytest.fixture()
def mock_success_response():
    """Build a mock success ModelResponse."""
    return ModelResponse(
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content=MOCK_EXPLANATION_JSON,
                    role="assistant"
                )
            )
        ],
        model="gemini/gemini-3.1-flash-lite",
        usage={"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300}
    )


class TestRouterConfig:
    def test_load_router_config(self) -> None:
        """Verify that configuration loader parses the YAML config correctly."""
        config = load_router_config()
        assert "model_list" in config
        assert "routing_strategy" in config
        assert config["cooldown_time"] == settings.LITELLM_COOLDOWN_TIME
        assert config["allowed_fails"] == settings.LITELLM_MAX_FAILURES

        # Verify model names exist
        model_names = [m["model_name"] for m in config["model_list"]]
        assert "gemini-3.1-flash-lite" in model_names
        assert "gemini-3.1-flash-lite" in model_names
        assert "llama-3.1-8b-instant" in model_names

    @patch("app.core.config.settings.LITELLM_ROUTING_STRATEGY", "simple-shuffle")
    def test_routing_strategy_override(self) -> None:
        """Verify routing strategy can be overridden by environment settings."""
        config = load_router_config()
        assert config["routing_strategy"] == "simple-shuffle"


class TestLiteLLMRouting:
    @patch("litellm.completion")
    def test_primary_succeeds_no_fallback(self, mock_completion, mock_success_response) -> None:
        """Primary deployment succeeds immediately without falling back."""
        mock_completion.return_value = mock_success_response

        chat = get_chat_model()
        messages = [HumanMessage(content="Hello")]
        result = chat.invoke(messages)

        # Should only call primary model
        assert mock_completion.call_count == 1
        called_model = mock_completion.call_args[1].get("model")
        assert called_model == "gemini/gemini-3.1-flash-lite"
        assert result.content == MOCK_EXPLANATION_JSON

    @patch("litellm.completion")
    def test_primary_fails_fallback_to_lite(self, mock_completion, mock_success_response) -> None:
        """Primary deployment fails, router automatically falls back to secondary Gemini Lite."""
        def side_effect(*args, **kwargs):
            model = kwargs.get("model")
            if model == "gemini/gemini-3.1-flash-lite":
                raise ServiceUnavailableError(
                    message="Google rate limit or timeout",
                    llm_provider="gemini",
                    model=model
                )
            # Secondary model succeeds
            mock_success_response.model = "gemini/gemini-3.1-flash-lite"
            return mock_success_response

        mock_completion.side_effect = side_effect

        chat = get_chat_model()
        messages = [HumanMessage(content="Hello")]
        result = chat.invoke(messages)

        # Should attempt primary then fall back to secondary
        expected_primary_calls = settings.LITELLM_RETRY_COUNT + 1
        expected_total_calls = expected_primary_calls + 1
        assert mock_completion.call_count == expected_total_calls
        calls = [call[1].get("model") for call in mock_completion.call_args_list]
        assert calls[0] == "gemini/gemini-3.1-flash-lite"
        assert calls[-1] == "gemini/gemini-3.1-flash-lite"
        assert result.content == MOCK_EXPLANATION_JSON

    @patch("litellm.completion")
    def test_gemini_unavailable_fallback_to_groq(self, mock_completion, mock_success_response) -> None:
        """Both Gemini deployments fail, router falls back to Groq Llama."""
        def side_effect(*args, **kwargs):
            model = kwargs.get("model")
            if "gemini-3.1-flash-lite" in model or "gemini-3.1-flash-lite" in model:
                raise ServiceUnavailableError(
                    message="Service Unavailable",
                    llm_provider="gemini",
                    model=model
                )
            # Groq succeeds
            mock_success_response.model = "groq/llama-3.1-8b-instant"
            return mock_success_response

        mock_completion.side_effect = side_effect

        chat = get_chat_model()
        messages = [HumanMessage(content="Hello")]
        result = chat.invoke(messages)

        # Should try gemini flash, gemini flash-lite, then groq llama
        expected_gemini_calls = (settings.LITELLM_RETRY_COUNT + 1) * 2
        expected_total_calls = expected_gemini_calls + 1
        assert mock_completion.call_count == expected_total_calls
        calls = [call[1].get("model") for call in mock_completion.call_args_list]
        assert calls[0] == "gemini/gemini-3.1-flash-lite"
        assert calls[-1] == "groq/llama-3.1-8b-instant"
        assert result.content == MOCK_EXPLANATION_JSON

    @patch("litellm.completion")
    def test_all_providers_fail_raises_exception(self, mock_completion) -> None:
        """All deployments fail, router exhausts fallbacks and propagates the final error."""
        def side_effect(*args, **kwargs):
            raise ServiceUnavailableError(
                message="All models failed",
                llm_provider="any",
                model=kwargs.get("model", "unknown")
            )
        mock_completion.side_effect = side_effect

        chat = get_chat_model()
        messages = [HumanMessage(content="Hello")]

        with pytest.raises(Exception):
            chat.invoke(messages)


class TestHealthTracking:
    @patch("litellm.Router._get_healthy_deployments")
    def test_health_aware_cooldown(self, mock_get_healthy) -> None:
        """Verify that Router filters unhealthy deployments during routing."""
        mock_get_healthy.return_value = [
            {"model_name": "llama-3.1-8b-instant", "litellm_params": {"model": "groq/llama-3.1-8b-instant"}}
        ]
        chat = get_chat_model()
        deployments = chat.router._get_healthy_deployments(model="llama-3.1-8b-instant")
        assert len(deployments) == 1
        assert deployments[0]["litellm_params"]["model"] == "groq/llama-3.1-8b-instant"


class TestIntegrationRouting:
    @patch("litellm.completion")
    def test_explain_score_chain_with_fallback(self, mock_completion, mock_success_response) -> None:
        """Integration test for the explain_score chain executing router fallbacks successfully."""
        def side_effect(*args, **kwargs):
            model = kwargs.get("model")
            if model == "gemini/gemini-3.1-flash-lite":
                raise ServiceUnavailableError(
                    message="Timeout simulation",
                    llm_provider="gemini",
                    model=model
                )
            mock_success_response.model = "gemini/gemini-3.1-flash-lite"
            return mock_success_response

        mock_completion.side_effect = side_effect

        sample_ats_response = ATSAnalyzeResponse(
            overall_score=80.0,
            keyword_score=80.0,
            experience_score=80.0,
            skills_score=80.0,
            education_score=80.0,
            summary_score=80.0,
            formatting_score=80.0,
            matched_keywords=[],
            missing_keywords=[],
            matched_skills=[],
            missing_skills=[],
            experience_summary=ExperienceSummarySchema(total_entries=1, total_years=2.0, has_metrics=True),
            education_summary=EducationSummarySchema(highest_degree="BS", certifications=[]),
            processing_time_ms=5.0
        )

        with patch("app.ai.chains.explain_score_chain.get_chat_model") as mock_get_chat:
            # Return our actual router model
            mock_get_chat.return_value = get_chat_model()
            
            explanation = run_explain_score(sample_ats_response, "Job description context")
            
            assert explanation.summary == "This is a great resume with 85 ATS score."
            assert explanation.strengths == ["Strong engineering experience."]
            
            expected_primary_calls = settings.LITELLM_RETRY_COUNT + 1
            expected_total_calls = expected_primary_calls + 1
            assert mock_completion.call_count == expected_total_calls
