"""Integration tests for Phase 10.1 — AI explanation field.

Strategy
--------
* Mock ``run_explain_score`` at the service level so no real LLM is called.
* Run the real FastAPI app via httpx2 ASGITransport.
* Verify that:
    - ``ai_explanation`` is present in the response when AI succeeds.
    - ``ai_explanation`` is null when AI chain raises.
    - HTTP status is 200 in both cases (graceful degradation).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx2 as httpx
import pytest
import pytest_asyncio

from app.schemas.ai import ATSExplanation, SectionExplanation, RecommendationSchema
from main import app

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENDPOINT = "/v1/ats/analyze"
_API_KEY = "test_api_key_123"
_AUTH_HEADERS = {"X-Internal-API-Key": _API_KEY}

_SAMPLE_JD = (
    "We are looking for a Python engineer with Docker and Kubernetes skills."
)

_MINIMAL_RESUME = (
    "John Smith\n"
    "Software Engineer\n\n"
    "Experience\n"
    "Python Developer at Acme Corp 2020-2023\n"
    "Built REST APIs with FastAPI and Docker.\n\n"
    "Skills\n"
    "Python Docker AWS\n\n"
    "Education\n"
    "B.Sc Computer Science State University 2019\n"
)

_FAKE_AI_EXPLANATION = ATSExplanation(
    strengths=["Python skills matched the JD.", "Docker experience is relevant."],
    weaknesses=["Kubernetes is missing.", "Summary section could be stronger."],
    section_explanations=[
        SectionExplanation(
            section="Keywords",
            score=60.0,
            explanation="Matched 3 of 5 required keywords.",
        ),
    ],
    suggestions=["Add Kubernetes to your resume.", "Add metrics to the experience section."],
    summary="The resume scored well overall. Kubernetes is the primary gap.",
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

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tests — ai_explanation present (AI chain succeeds)
# ---------------------------------------------------------------------------


class TestAIExplanationPresent:
    async def test_response_is_200_with_ai_explanation(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        """Endpoint returns 200 and ai_explanation is populated when AI succeeds."""
        with patch(
            "app.services.ats_service.run_explain_score",
            return_value=_FAKE_AI_EXPLANATION,
        ):
            response = await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)

        assert response.status_code == 200, response.text
        body = response.json()
        assert "ai_explanation" in body
        assert body["ai_explanation"] is not None

    async def test_ai_explanation_has_strengths(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        with patch(
            "app.services.ats_service.run_explain_score",
            return_value=_FAKE_AI_EXPLANATION,
        ):
            body = (await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)).json()

        assert isinstance(body["ai_explanation"]["strengths"], list)
        assert len(body["ai_explanation"]["strengths"]) > 0

    async def test_ai_explanation_has_weaknesses(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        with patch(
            "app.services.ats_service.run_explain_score",
            return_value=_FAKE_AI_EXPLANATION,
        ):
            body = (await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)).json()

        assert isinstance(body["ai_explanation"]["weaknesses"], list)
        assert len(body["ai_explanation"]["weaknesses"]) > 0

    async def test_ai_explanation_has_section_explanations(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        with patch(
            "app.services.ats_service.run_explain_score",
            return_value=_FAKE_AI_EXPLANATION,
        ):
            body = (await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)).json()

        assert isinstance(body["ai_explanation"]["section_explanations"], list)

    async def test_ai_explanation_has_summary(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        with patch(
            "app.services.ats_service.run_explain_score",
            return_value=_FAKE_AI_EXPLANATION,
        ):
            body = (await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)).json()

        assert isinstance(body["ai_explanation"]["summary"], str)
        assert len(body["ai_explanation"]["summary"]) > 0

    async def test_deterministic_scores_still_present_with_ai(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        """AI explanation must NOT affect the deterministic scores."""
        with patch(
            "app.services.ats_service.run_explain_score",
            return_value=_FAKE_AI_EXPLANATION,
        ):
            body = (await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)).json()

        for field in ("overall_score", "keyword_score", "experience_score", "skills_score"):
            assert field in body, f"Missing score field: {field}"
            assert 0.0 <= body[field] <= 100.0


# ---------------------------------------------------------------------------
# Tests — graceful degradation (AI chain fails)
# ---------------------------------------------------------------------------


class TestAIExplanationGracefulDegradation:
    async def test_endpoint_returns_200_when_ai_fails(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        """HTTP 200 must be returned even when the AI chain raises an exception."""
        with patch(
            "app.services.ats_service.run_explain_score",
            side_effect=RuntimeError("LLM unavailable"),
        ):
            response = await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)

        assert response.status_code == 200, response.text

    async def test_ai_explanation_is_null_when_ai_fails(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        """ai_explanation must be null (not absent) when the AI chain fails."""
        with patch(
            "app.services.ats_service.run_explain_score",
            side_effect=RuntimeError("LLM unavailable"),
        ):
            body = (await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)).json()

        assert "ai_explanation" in body
        assert body["ai_explanation"] is None

    async def test_deterministic_scores_present_when_ai_fails(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        """All deterministic score fields must be present even when AI fails."""
        with patch(
            "app.services.ats_service.run_explain_score",
            side_effect=Exception("network timeout"),
        ):
            body = (await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)).json()

        for field in (
            "overall_score", "keyword_score", "experience_score",
            "skills_score", "education_score", "summary_score", "formatting_score",
        ):
            assert field in body, f"Score field missing when AI failed: {field}"
            assert isinstance(body[field], float)

    async def test_version_is_1_2(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        """Response schema version must now be 1.2."""
        with patch(
            "app.services.ats_service.run_explain_score",
            side_effect=RuntimeError("LLM unavailable"),
        ):
            body = (await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)).json()

        assert body.get("version") == "1.2"
        assert body.get("ai_status") == "unavailable"

