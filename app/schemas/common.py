"""Common response schemas used across all API endpoints."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
