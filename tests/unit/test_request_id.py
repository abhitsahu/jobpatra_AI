"""Unit tests for request ID middleware."""

import pytest
from fastapi.testclient import TestClient

from main import app

REQUEST_ID_HEADER = "X-Request-ID"


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)


def test_request_id_generated_when_missing(client):
    """Response must include a generated X-Request-ID when none is sent."""
    response = client.get("/health")
    assert REQUEST_ID_HEADER in response.headers
    rid = response.headers[REQUEST_ID_HEADER]
    assert len(rid) == 36  # UUID4 format: 8-4-4-4-12


def test_request_id_propagated_when_provided(client):
    """Client-provided X-Request-ID must be echoed back."""
    custom_id = "my-custom-request-id-123"
    response = client.get("/health", headers={REQUEST_ID_HEADER: custom_id})
    assert response.headers[REQUEST_ID_HEADER] == custom_id


def test_unique_request_ids_per_request(client):
    """Each request without a provided ID should get a unique generated ID."""
    r1 = client.get("/health")
    r2 = client.get("/health")
    assert r1.headers[REQUEST_ID_HEADER] != r2.headers[REQUEST_ID_HEADER]


def test_request_id_present_on_error_responses(client):
    """Even 401 responses should include the X-Request-ID header."""
    response = client.get("/api/v1/nonexistent")
    assert response.status_code == 401
    assert REQUEST_ID_HEADER in response.headers
