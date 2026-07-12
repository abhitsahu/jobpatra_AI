"""Unit tests for internal authentication middleware."""

import pytest
from fastapi.testclient import TestClient

from main import app

from app.core.config import settings

VALID_TOKEN = settings.INTERNAL_API_KEY.get_secret_value()


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)


def test_health_bypasses_auth(client):
    """GET /health must be accessible without any token."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_token_via_api_key_header(client):
    """Valid token in X-Internal-API-Key should bypass auth (returns 404 not 401)."""
    response = client.get("/api/v1/nonexistent", headers={"X-Internal-API-Key": VALID_TOKEN})
    assert response.status_code == 404


def test_valid_token_via_bearer_header(client):
    """Valid token in Authorization: Bearer should bypass auth (returns 404 not 401)."""
    response = client.get(
        "/api/v1/nonexistent", headers={"Authorization": f"Bearer {VALID_TOKEN}"}
    )
    assert response.status_code == 404


def test_missing_token_returns_401(client):
    """Request to a non-public path without token should return 401."""
    response = client.get("/api/v1/nonexistent")
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "UNAUTHORIZED"
    assert "missing" in body["error"]["message"].lower()


def test_invalid_token_returns_401(client):
    """Request with wrong token should return 401."""
    response = client.get(
        "/api/v1/nonexistent", headers={"X-Internal-API-Key": "wrong_token"}
    )
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "UNAUTHORIZED"
    assert "invalid" in body["error"]["message"].lower()
