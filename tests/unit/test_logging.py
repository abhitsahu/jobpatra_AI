"""Unit tests for logging middleware."""

import logging

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)


def test_request_is_logged(client, caplog):
    """Each request should produce a log entry with method and path."""
    with caplog.at_level(logging.INFO, logger="jobpatra"):
        client.get("/health")

    log_messages = [r.message for r in caplog.records if r.name == "jobpatra"]
    matching = [m for m in log_messages if "GET" in m and "/health" in m]
    assert len(matching) >= 1


def test_status_code_logged(client, caplog):
    """Log entry should contain the HTTP status code."""
    with caplog.at_level(logging.INFO, logger="jobpatra"):
        client.get("/health")

    log_messages = [r.message for r in caplog.records if r.name == "jobpatra"]
    matching = [m for m in log_messages if "200" in m]
    assert len(matching) >= 1


def test_latency_logged(client, caplog):
    """Log entry should contain response time in milliseconds."""
    with caplog.at_level(logging.INFO, logger="jobpatra"):
        client.get("/health")

    log_messages = [r.message for r in caplog.records if r.name == "jobpatra"]
    # Latency appears as e.g. "0.5ms" or "1.2ms"
    matching = [m for m in log_messages if "ms)" in m]
    assert len(matching) >= 1


def test_request_id_in_logs(client, caplog):
    """Log entry should include the request ID prefix."""
    custom_id = "abcdef12-test-id"
    with caplog.at_level(logging.INFO, logger="jobpatra"):
        client.get("/health", headers={"X-Request-ID": custom_id})

    log_messages = [r.message for r in caplog.records if r.name == "jobpatra"]
    # The middleware logs first 8 chars of request ID
    matching = [m for m in log_messages if "abcdef12" in m]
    assert len(matching) >= 1
