"""Integration test for Phase 10.3 — Output guardrail failure & retry policy.

Tests verify:
- When the LLM provider returns a malformed response (fails output validation),
  the retry policy is triggered exactly once.
- If both attempts fail output validation, ``AIGenerationError`` is raised
  internally, caught, and logged.
- The HTTP response remains 200 (graceful degradation).
- The final response has ``ai_status="unavailable"`` and ``ai_explanation=None``.
- Deterministic ATS scores are still successfully returned.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx2 as httpx
import pytest
import pytest_asyncio

from app.schemas.ai import ATSExplanation
from main import app

_ENDPOINT = "/v1/ats/analyze"
_API_KEY = "test_api_key_123"
_AUTH_HEADERS = {"X-Internal-API-Key": _API_KEY}

_MINIMAL_RESUME = (
    "John Smith\n"
    "Software Engineer\n\n"
    "Experience\n"
    "Python Developer at Acme Corp 2020-2023\n\n"
    "Skills\n"
    "Python Docker\n\n"
    "Education\n"
    "B.Sc Computer Science 2019\n"
)

_SAMPLE_JD = "We are looking for a Python developer with Docker skills."


@pytest_asyncio.fixture()
async def client() -> httpx.AsyncClient:
    """Async ASGI test client for the FastAPI app."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


@pytest.fixture()
def text_payload() -> dict:
    return {
        "resume": {"text": _MINIMAL_RESUME},
        "job_description": {"text": _SAMPLE_JD},
    }


@pytest.mark.asyncio
class TestATSAnalyzeGuardrailFailure:
    async def test_retry_on_malformed_output_then_graceful_degradation(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        """Verify output validation failure triggers a retry, raises AIGenerationError, and degrades gracefully."""
        # Mock the LLM chain invoke to return a malformed JSON response (missing required fields)
        mock_message = MagicMock()
        mock_message.content = '{"strengths": []}'
        mock_message.response_metadata = {"finish_reason": "stop"}

        # Mock the LLM chain invoke to return the malformed response twice
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_message

        # We patch get_chat_model and EXPLAIN_SCORE_PROMPT_V2 in explain_score_chain
        # to construct our mock_chain.
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MagicMock()

        with patch("app.ai.chains.explain_score_chain.get_chat_model", return_value=mock_llm):
            with patch("app.ai.chains.explain_score_chain.EXPLAIN_SCORE_PROMPT_V2") as mock_prompt:
                mock_prompt.__or__ = MagicMock(return_value=mock_chain)

                response = await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)

        # ── Assertions ───────────────────────────────────────────────────────
        assert response.status_code == 200, response.text
        body = response.json()

        # Deterministic scores must still be present
        assert "overall_score" in body
        assert body["overall_score"] > 0

        # AI Status must be "unavailable" and explanation must be null
        assert body.get("ai_status") == "unavailable"
        assert body.get("ai_explanation") is None

        # Verify retry policy was executed exactly once (total 2 invocations: attempt 1 + retry)
        assert mock_chain.invoke.call_count == 2

        # Verify the second call received the correction instructions in jd_context
        first_call_args = mock_chain.invoke.call_args_list[0][0][0]
        second_call_args = mock_chain.invoke.call_args_list[1][0][0]

        assert "CORRECTION REQUIRED" not in first_call_args["jd_context"]
        assert "CORRECTION REQUIRED" in second_call_args["jd_context"]
