"""Integration tests for POST /v1/ats/analyze.

Uses httpx2's AsyncClient with ASGITransport to test the full FastAPI
application — no network socket, no mocking of internal modules.

The complete pipeline (parse → normalize → extract → match → score)
executes for real with the test fixtures.

Authentication token is supplied via the conftest.py env-var fixture
(INTERNAL_API_KEY = "test_api_key_123").
"""

import pathlib

import httpx2 as httpx
import pytest
import pytest_asyncio

from main import app

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENDPOINT = "/v1/ats/analyze"
_API_KEY = "test_api_key_123"        # matches conftest.py INTERNAL_API_KEY
_AUTH_HEADERS = {"X-Internal-API-Key": _API_KEY}

_SAMPLE_JD = (
    "We are looking for a Software Engineer proficient in Python, React, "
    "Docker, and Node.js. Experience with cloud platforms (AWS) is a plus. "
    "The candidate should have strong experience building scalable REST APIs "
    "and working in an Agile environment."
)

_FIXTURES_DIR = pathlib.Path(__file__).parent.parent / "fixtures"

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Shared async client fixture
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def client() -> httpx.AsyncClient:
    """Async ASGI test client wrapping the FastAPI app."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


@pytest.fixture
def text_payload() -> dict:
    """Valid text-mode request payload using the sample resume fixture."""
    resume_txt = (_FIXTURES_DIR / "sample_resume.txt").read_text(encoding="utf-8")
    return {
        "resume": {"text": resume_txt},
        "job_description": {"text": _SAMPLE_JD},
    }


# ---------------------------------------------------------------------------
# Happy-path — text mode
# ---------------------------------------------------------------------------


class TestAnalyzeTextMode:
    async def test_status_200(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        response = await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)
        assert response.status_code == 200, response.text

    async def test_response_is_json(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        response = await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)
        assert response.headers["content-type"].startswith("application/json")

    async def test_overall_score_present(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        body = (await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)).json()
        assert "overall_score" in body

    async def test_all_score_fields_present(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        body = (await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)).json()
        for field in (
            "overall_score", "keyword_score", "experience_score",
            "skills_score", "education_score", "summary_score", "formatting_score",
        ):
            assert field in body, f"Missing score field: {field}"

    async def test_all_scores_in_valid_range(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        body = (await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)).json()
        for field in (
            "overall_score", "keyword_score", "experience_score",
            "skills_score", "education_score", "summary_score", "formatting_score",
        ):
            val = body[field]
            assert 0.0 <= val <= 100.0, f"{field} = {val} out of [0, 100]"

    async def test_matched_keywords_is_list(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        body = (await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)).json()
        assert isinstance(body["matched_keywords"], list)

    async def test_missing_keywords_is_list(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        body = (await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)).json()
        assert isinstance(body["missing_keywords"], list)

    async def test_matched_keywords_have_required_fields(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        body = (await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)).json()
        for kw in body["matched_keywords"]:
            assert "keyword" in kw
            assert "matchType" in kw
            assert kw["matchType"] in ("EXACT", "SYNONYM", "FUZZY", "SEMANTIC")

    async def test_matched_skills_is_list(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        body = (await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)).json()
        assert isinstance(body["matched_skills"], list)

    async def test_missing_skills_is_list(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        body = (await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)).json()
        assert isinstance(body["missing_skills"], list)

    async def test_experience_summary_present(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        body = (await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)).json()
        exp = body.get("experience_summary", {})
        assert "total_entries" in exp
        assert "total_years" in exp
        assert "has_metrics" in exp

    async def test_education_summary_present(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        body = (await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)).json()
        edu = body.get("education_summary", {})
        assert "highest_degree" in edu
        assert "certifications" in edu

    async def test_processing_time_ms_present_and_positive(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        body = (await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)).json()
        assert "processing_time_ms" in body
        assert body["processing_time_ms"] > 0

    async def test_version_present(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        body = (await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)).json()
        assert body.get("version") == "1.2"

    async def test_deterministic_same_input_same_output(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        """Running the same request twice must produce identical scores."""
        body1 = (await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)).json()
        body2 = (await client.post(_ENDPOINT, json=text_payload, headers=_AUTH_HEADERS)).json()
        for field in (
            "overall_score", "keyword_score", "experience_score",
            "skills_score", "education_score", "summary_score", "formatting_score",
        ):
            assert body1[field] == body2[field], f"{field} is not deterministic"


# ---------------------------------------------------------------------------
# Authentication tests
# ---------------------------------------------------------------------------


class TestAuthentication:
    async def test_missing_api_key_returns_401(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        response = await client.post(_ENDPOINT, json=text_payload)
        assert response.status_code == 401

    async def test_wrong_api_key_returns_401(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        response = await client.post(
            _ENDPOINT,
            json=text_payload,
            headers={"X-Internal-API-Key": "wrong-key"},
        )
        assert response.status_code == 401

    async def test_bearer_token_accepted(
        self, client: httpx.AsyncClient, text_payload: dict
    ) -> None:
        response = await client.post(
            _ENDPOINT,
            json=text_payload,
            headers={"Authorization": f"Bearer {_API_KEY}"},
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


class TestValidation:
    async def test_empty_body_returns_422(self, client: httpx.AsyncClient) -> None:
        response = await client.post(_ENDPOINT, json={}, headers=_AUTH_HEADERS)
        assert response.status_code == 422

    async def test_blank_resume_text_returns_422(self, client: httpx.AsyncClient) -> None:
        payload = {
            "resume": {"text": "   "},
            "job_description": {"text": _SAMPLE_JD},
        }
        response = await client.post(_ENDPOINT, json=payload, headers=_AUTH_HEADERS)
        assert response.status_code == 422

    async def test_blank_jd_returns_422(self, client: httpx.AsyncClient) -> None:
        payload = {
            "resume": {"text": "John Doe - Software Engineer"},
            "job_description": {"text": ""},
        }
        response = await client.post(_ENDPOINT, json=payload, headers=_AUTH_HEADERS)
        assert response.status_code == 422

    async def test_resume_with_no_text_or_file_returns_422(
        self, client: httpx.AsyncClient
    ) -> None:
        payload = {
            "resume": {},
            "job_description": {"text": _SAMPLE_JD},
        }
        response = await client.post(_ENDPOINT, json=payload, headers=_AUTH_HEADERS)
        assert response.status_code == 422
