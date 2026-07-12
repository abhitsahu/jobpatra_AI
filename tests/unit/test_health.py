"""Unit tests for the health check endpoint."""

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)


def test_health_returns_200(client):
    """GET /health should return HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok_status(client):
    """GET /health should return {"status": "ok"}."""
    response = client.get("/health")
    assert response.json() == {"status": "ok"}
