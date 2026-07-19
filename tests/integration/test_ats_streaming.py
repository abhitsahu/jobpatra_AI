"""Integration tests for Phase 10.6 — Streaming AI Explanation.

Strategy
--------
* Mock ``run_explain_score`` at the service level so no real LLM is called.
* Test that standard POST with ``stream=True`` yields all expected SSE events.
* Test that input guardrails reject invalid inputs with HTTP 422 immediately.
* Test that AI generation errors degrade gracefully into ``ai_unavailable`` event.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx2 as httpx
import pytest
import pytest_asyncio

from app.core.errors import InvalidInputError, AIGenerationError
from app.schemas.ai import ATSExplanation, SectionExplanation, RecommendationSchema
from main import app

_ENDPOINT = "/v1/ats/analyze"
_API_KEY = "test_api_key_123"
_AUTH_HEADERS = {"X-Internal-API-Key": _API_KEY}

_SAMPLE_JD = "We are looking for a Python engineer with Docker and Kubernetes skills."
_MINIMAL_RESUME = (
    "John Smith\nSoftware Engineer\n\nExperience\nPython Developer at Acme Corp\n"
)

_FAKE_AI_EXPLANATION = ATSExplanation(
    strengths=["Python matched."],
    weaknesses=["Kubernetes missing."],
    section_explanations=[
        SectionExplanation(section="Keywords", score=60.0, explanation="Good.")
    ],
    suggestions=["Add Kubernetes."],
    summary="Good resume.",
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


@pytest_asyncio.fixture()
async def client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


@pytest.fixture()
def valid_payload() -> dict:
    return {
        "resume": {"text": _MINIMAL_RESUME},
        "job_description": {"text": _SAMPLE_JD},
        "stream": True,
    }


def parse_sse_events(text: str) -> list[dict[str, str]]:
    """Parse raw SSE body into a list of {event: ..., data: ...} dicts."""
    events = []
    current_event = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            if current_event:
                events.append(current_event)
                current_event = {}
            continue
        if line.startswith("event:"):
            current_event["event"] = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current_event["data"] = json.loads(line[len("data:"):].strip())
    if current_event:
        events.append(current_event)
    return events


@pytest.mark.asyncio
class TestATSStreaming:
    async def test_streaming_happy_path(
        self, client: httpx.AsyncClient, valid_payload: dict
    ) -> None:
        """Verify successful stream yields all expected events in correct sequence."""
        with patch(
            "app.services.ats_service.run_explain_score",
            return_value=_FAKE_AI_EXPLANATION,
        ):
            response = await client.post(_ENDPOINT, json=valid_payload, headers=_AUTH_HEADERS)

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        events = parse_sse_events(response.text)
        assert len(events) >= 9

        # Verify correct sequence of event names
        event_names = [e["event"] for e in events]
        assert "pipeline_started" in event_names
        assert "ats_running" in event_names
        assert "ats_complete" in event_names
        assert "ai_started" in event_names
        assert "ai_analyzing_strengths" in event_names
        assert "ai_analyzing_weaknesses" in event_names
        assert "ai_generating_suggestions" in event_names
        assert "ai_complete" in event_names
        assert "complete" in event_names

        # Verify the complete payload is returned inside 'complete' event
        complete_event = [e for e in events if e["event"] == "complete"][0]
        payload = complete_event["data"]
        assert payload["overall_score"] > 0
        assert payload["ai_status"] == "ok"
        assert payload["ai_explanation"]["suggestions"] == ["Add Kubernetes."]

    async def test_streaming_fail_fast_input_guardrail(
        self, client: httpx.AsyncClient
    ) -> None:
        """Verify that invalid input fails fast in the route handler returning 422."""
        invalid_payload = {
            "resume": {"text": "  "},  # empty
            "job_description": {"text": _SAMPLE_JD},
            "stream": True,
        }

        response = await client.post(_ENDPOINT, json=invalid_payload, headers=_AUTH_HEADERS)
        assert response.status_code == 422
        # Verify it did not return an SSE event stream
        assert "text/event-stream" not in response.headers.get("content-type", "")

    async def test_streaming_graceful_ai_failure(
        self, client: httpx.AsyncClient, valid_payload: dict
    ) -> None:
        """Verify that AI failure yields ai_unavailable event but still returns deterministic results."""
        with patch(
            "app.services.ats_service.run_explain_score",
            side_effect=AIGenerationError("Output guardrail failed after retry."),
        ):
            response = await client.post(_ENDPOINT, json=valid_payload, headers=_AUTH_HEADERS)

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        events = parse_sse_events(response.text)
        event_names = [e["event"] for e in events]

        assert "ai_unavailable" in event_names
        assert "complete" in event_names

        complete_event = [e for e in events if e["event"] == "complete"][0]
        payload = complete_event["data"]
        assert payload["overall_score"] > 0
        assert payload["ai_status"] == "unavailable"
        assert payload["ai_explanation"] is None
