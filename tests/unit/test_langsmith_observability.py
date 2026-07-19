"""Unit tests for Phase 10.2 — LangSmith observability.

Test strategy
-------------
* No real LangSmith API calls are made (``langsmith.Client`` is mocked).
* Tests verify:
    1. ``get_langsmith_callback()`` returns ``None`` when tracing is disabled.
    2. ``get_langsmith_callback()`` returns ``None`` when the API key is missing.
    3. ``get_langsmith_callback()`` returns a ``LangChainTracer`` when properly configured.
    4. ``invoke_with_tracing`` calls the chain with an empty callback list when
       tracing is off — behavior is identical to a plain ``chain.invoke``.
    5. ``invoke_with_tracing`` calls the chain with the tracer in the callback
       list when tracing is on.
    6. A failure inside ``get_langsmith_callback`` (e.g. network error during
       client construction) is swallowed — ``None`` is returned, not raised.
    7. The end-to-end ``run_explain_score`` result is unchanged regardless of
       whether tracing is enabled.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.schemas.ai import ATSExplanation, SectionExplanation, RecommendationSchema
from app.schemas.ats import (
    ATSAnalyzeResponse,
    EducationSummarySchema,
    ExperienceSummarySchema,
    MatchedKeywordSchema,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_response() -> ATSAnalyzeResponse:
    """Minimal valid ATSAnalyzeResponse for tracing tests."""
    return ATSAnalyzeResponse(
        overall_score=62.5,
        keyword_score=55.0,
        experience_score=70.0,
        skills_score=60.0,
        education_score=80.0,
        summary_score=40.0,
        formatting_score=75.0,
        matched_keywords=[MatchedKeywordSchema(keyword="python", matchType="EXACT")],
        missing_keywords=["kubernetes"],
        matched_skills=["Python"],
        missing_skills=["Kubernetes"],
        experience_summary=ExperienceSummarySchema(
            total_entries=1, total_years=2.0, has_metrics=False
        ),
        education_summary=EducationSummarySchema(
            highest_degree="B.Sc", certifications=[]
        ),
        processing_time_ms=10.0,
    )


_FAKE_EXPLANATION = ATSExplanation(
    strengths=["Python matched."],
    weaknesses=["Missing Kubernetes."],
    section_explanations=[
        SectionExplanation(section="Keywords", score=55.0, explanation="Partial match.")
    ],
    suggestions=["Add Kubernetes to skills."],
    summary="Score: 62.5/100.",
    recommendations=[
        RecommendationSchema(
            priority="High",
            issue="Missing professional summary.",
            why="A summary hooks the recruiter.",
            copy_paste_content="Motivated engineer.",
            placement="Top of resume.",
            ats_impact="+10 points"
        )
    ],
)

_SAMPLE_JD = "Python engineer needed. Kubernetes required."


# ---------------------------------------------------------------------------
# Tests for observability/langsmith_config.py
# ---------------------------------------------------------------------------


class TestGetLangsmithCallback:
    def test_returns_none_when_tracing_disabled(self) -> None:
        """When LANGSMITH_TRACING_ENABLED=False, callback must be None."""
        with patch("observability.langsmith_config.settings") as mock_settings:
            mock_settings.LANGSMITH_TRACING_ENABLED = False
            mock_settings.LANGSMITH_API_KEY = None

            from observability.langsmith_config import get_langsmith_callback

            result = get_langsmith_callback()

        assert result is None

    def test_returns_none_when_api_key_missing(self) -> None:
        """When tracing is enabled but API key is absent, callback must be None."""
        with patch("observability.langsmith_config.settings") as mock_settings:
            mock_settings.LANGSMITH_TRACING_ENABLED = True
            mock_settings.LANGSMITH_API_KEY = None
            mock_settings.LANGSMITH_PROJECT = "test-project"
            mock_settings.LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"

            from observability.langsmith_config import get_langsmith_callback

            result = get_langsmith_callback()

        assert result is None

    def test_returns_tracer_when_enabled_and_key_present(self) -> None:
        """Returns a LangChainTracer when properly configured."""
        from pydantic import SecretStr

        mock_secret = MagicMock(spec=SecretStr)
        mock_secret.get_secret_value.return_value = "ls-test-key"

        with patch("observability.langsmith_config.settings") as mock_settings:
            mock_settings.LANGSMITH_TRACING_ENABLED = True
            mock_settings.LANGSMITH_API_KEY = mock_secret
            mock_settings.LANGSMITH_PROJECT = "test-project"
            mock_settings.LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"

            with patch("langsmith.Client") as mock_client_cls:
                with patch(
                    "langchain_core.tracers.langchain.LangChainTracer"
                ) as mock_tracer_cls:
                    mock_tracer = MagicMock()
                    mock_tracer_cls.return_value = mock_tracer

                    from observability.langsmith_config import get_langsmith_callback

                    result = get_langsmith_callback(
                        tags=["test_tag"], metadata={"key": "val"}
                    )

        assert result is mock_tracer

    def test_returns_none_when_client_raises(self) -> None:
        """If LangSmith client construction raises, None is returned (no propagation)."""
        from pydantic import SecretStr

        mock_secret = MagicMock(spec=SecretStr)
        mock_secret.get_secret_value.return_value = "ls-bad-key"

        with patch("observability.langsmith_config.settings") as mock_settings:
            mock_settings.LANGSMITH_TRACING_ENABLED = True
            mock_settings.LANGSMITH_API_KEY = mock_secret
            mock_settings.LANGSMITH_PROJECT = "test-project"
            mock_settings.LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"

            with patch("langsmith.Client", side_effect=Exception("Network error")):
                from observability.langsmith_config import get_langsmith_callback

                result = get_langsmith_callback()

        assert result is None

    def test_tracing_disabled_sets_env_var_false(self) -> None:
        """When tracing is off, LANGCHAIN_TRACING_V2 env var is set to 'false'."""
        import os

        with patch("observability.langsmith_config.settings") as mock_settings:
            mock_settings.LANGSMITH_TRACING_ENABLED = False
            mock_settings.LANGSMITH_API_KEY = None

            from observability.langsmith_config import get_langsmith_callback

            get_langsmith_callback()

        assert os.environ.get("LANGCHAIN_TRACING_V2") == "false"


# ---------------------------------------------------------------------------
# Tests for app/ai/chains/base_chain.py
# ---------------------------------------------------------------------------


class TestInvokeWithTracing:
    def test_calls_chain_invoke_with_empty_callbacks_when_tracing_off(self) -> None:
        """When get_langsmith_callback returns None, chain.invoke gets empty callbacks."""
        from app.ai.chains.base_chain import invoke_with_tracing

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "result"

        with patch(
            "app.ai.chains.base_chain.get_langsmith_callback", return_value=None
        ):
            result = invoke_with_tracing(mock_chain, {"key": "val"})

        mock_chain.invoke.assert_called_once_with(
            {"key": "val"}, config={"callbacks": []}
        )
        assert result == "result"

    def test_calls_chain_invoke_with_tracer_when_tracing_on(self) -> None:
        """When get_langsmith_callback returns a tracer, it appears in callbacks."""
        from app.ai.chains.base_chain import invoke_with_tracing

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "traced-result"
        mock_tracer = MagicMock()

        with patch(
            "app.ai.chains.base_chain.get_langsmith_callback",
            return_value=mock_tracer,
        ):
            result = invoke_with_tracing(
                mock_chain,
                {"k": "v"},
                tags=["tag1"],
                metadata={"rid": "abc"},
            )

        mock_chain.invoke.assert_called_once_with(
            {"k": "v"}, config={"callbacks": [mock_tracer]}
        )
        assert result == "traced-result"

    def test_passes_tags_and_metadata_to_callback_factory(self) -> None:
        """Tags and metadata are forwarded to get_langsmith_callback."""
        from app.ai.chains.base_chain import invoke_with_tracing

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = None

        with patch(
            "app.ai.chains.base_chain.get_langsmith_callback", return_value=None
        ) as mock_get_cb:
            invoke_with_tracing(
                mock_chain,
                {},
                tags=["my_tag"],
                metadata={"env": "test"},
            )

        mock_get_cb.assert_called_once_with(tags=["my_tag"], metadata={"env": "test"})

    def test_returns_same_value_with_and_without_tracing(self) -> None:
        """invoke_with_tracing must not alter the chain's return value."""
        from app.ai.chains.base_chain import invoke_with_tracing

        expected = {"score": 72.5}
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = expected

        with patch(
            "app.ai.chains.base_chain.get_langsmith_callback", return_value=None
        ):
            result_no_tracing = invoke_with_tracing(mock_chain, {})

        mock_tracer = MagicMock()
        with patch(
            "app.ai.chains.base_chain.get_langsmith_callback",
            return_value=mock_tracer,
        ):
            result_with_tracing = invoke_with_tracing(mock_chain, {})

        assert result_no_tracing is expected
        assert result_with_tracing is expected


# ---------------------------------------------------------------------------
# End-to-end: run_explain_score returns same output regardless of tracing
# ---------------------------------------------------------------------------


class TestRunExplainScoreWithTracing:
    def test_result_unchanged_when_tracing_disabled(
        self, sample_response: ATSAnalyzeResponse
    ) -> None:
        """run_explain_score output is identical when tracing is off."""
        from app.ai.chains.explain_score_chain import run_explain_score

        mock_msg = MagicMock()
        mock_msg.content = _FAKE_EXPLANATION.model_dump_json()
        mock_msg.response_metadata = {"finish_reason": "stop"}

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_msg

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MagicMock()

        with patch("app.ai.chains.explain_score_chain.get_chat_model", return_value=mock_llm):
            with patch(
                "app.ai.chains.explain_score_chain.EXPLAIN_SCORE_PROMPT_V2"
            ) as mock_prompt:
                mock_prompt.__or__ = MagicMock(return_value=mock_chain)

                with patch(
                    "app.ai.chains.base_chain.get_langsmith_callback", return_value=None
                ):
                    result = run_explain_score(sample_response, _SAMPLE_JD)

        assert result == _FAKE_EXPLANATION

    def test_result_unchanged_when_tracing_enabled(
        self, sample_response: ATSAnalyzeResponse
    ) -> None:
        """run_explain_score output is identical when tracing is active."""
        from app.ai.chains.explain_score_chain import run_explain_score

        mock_msg = MagicMock()
        mock_msg.content = _FAKE_EXPLANATION.model_dump_json()
        mock_msg.response_metadata = {"finish_reason": "stop"}

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_msg

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MagicMock()

        mock_tracer = MagicMock()

        with patch("app.ai.chains.explain_score_chain.get_chat_model", return_value=mock_llm):
            with patch(
                "app.ai.chains.explain_score_chain.EXPLAIN_SCORE_PROMPT_V2"
            ) as mock_prompt:
                mock_prompt.__or__ = MagicMock(return_value=mock_chain)

                with patch(
                    "app.ai.chains.base_chain.get_langsmith_callback",
                    return_value=mock_tracer,
                ):
                    result = run_explain_score(sample_response, _SAMPLE_JD)

        assert result == _FAKE_EXPLANATION

    def test_tracer_passed_to_chain_config_when_enabled(
        self, sample_response: ATSAnalyzeResponse
    ) -> None:
        """When tracing is on, chain.invoke is called with the tracer in callbacks."""
        from app.ai.chains.explain_score_chain import run_explain_score

        mock_msg = MagicMock()
        mock_msg.content = _FAKE_EXPLANATION.model_dump_json()
        mock_msg.response_metadata = {"finish_reason": "stop"}

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_msg

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MagicMock()

        mock_tracer = MagicMock()

        with patch("app.ai.chains.explain_score_chain.get_chat_model", return_value=mock_llm):
            with patch(
                "app.ai.chains.explain_score_chain.EXPLAIN_SCORE_PROMPT_V2"
            ) as mock_prompt:
                mock_prompt.__or__ = MagicMock(return_value=mock_chain)

                with patch(
                    "app.ai.chains.base_chain.get_langsmith_callback",
                    return_value=mock_tracer,
                ):
                    run_explain_score(sample_response, _SAMPLE_JD)

        _, call_kwargs = mock_chain.invoke.call_args
        config = call_kwargs.get("config", {})
        assert mock_tracer in config.get("callbacks", [])
