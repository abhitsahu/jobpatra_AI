"""Common response schemas used across all API endpoints."""

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """Standard error detail structure."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """Standardized error response envelope.

    Example:
        {
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Resume file is missing."
            }
        }
    """

    error: ErrorDetail


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
