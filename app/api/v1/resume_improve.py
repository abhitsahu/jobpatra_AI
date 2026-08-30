"""Resume improvement route — POST /v1/resume/improve

This module is intentionally thin. It:
  1. Validates the request body (Pydantic).
  2. Validates that section_type is known.
  3. Delegates streaming to ``resume_improve_service``.
  4. Returns a ``StreamingResponse``.

Authentication is enforced by ``InternalAuthMiddleware`` — this route does
not duplicate that logic. Any request that reaches here has already been
authenticated.

This module does NOT:
  * Implement prompt logic
  * Implement LLM calls directly
  * Check subscription limits (that is the Next.js proxy's responsibility)
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from app.ai.prompts.resume_improve import VALID_SECTION_TYPES
from app.core.errors import InvalidInputError
from app.services import resume_improve_service

router = APIRouter(prefix="/v1/resume", tags=["Resume AI"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ImproveRequest(BaseModel):
    """Request body for the resume improvement endpoint."""

    section_type: str
    """Which section to improve: ``summary``, ``experience``, ``projects``, ``objective``."""

    current_text: str
    """The original section text to rewrite. Must be non-empty."""

    resume_context: Optional[str] = None
    """Optional full-resume text for additional LLM context."""

    @field_validator("section_type")
    @classmethod
    def validate_section_type(cls, v: str) -> str:
        normalised = v.lower().strip()
        if normalised not in VALID_SECTION_TYPES:
            raise ValueError(
                f"Unknown section_type '{v}'. "
                f"Valid values: {sorted(VALID_SECTION_TYPES)}"
            )
        return normalised

    @field_validator("current_text")
    @classmethod
    def validate_current_text(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("current_text must not be empty.")
        if len(stripped) > 8_000:
            raise ValueError("current_text exceeds maximum length of 8 000 characters.")
        return stripped


# ---------------------------------------------------------------------------
# Route handler
# ---------------------------------------------------------------------------


@router.post(
    "/improve",
    summary="Stream AI-improved text for a resume section",
    description=(
        "Accepts a resume section type and its current text, then streams an "
        "AI-improved version as Server-Sent Events. "
        "Each event: ``data: {\"token\": \"<text>\"}``. "
        "Terminal event: ``data: [DONE]``."
    ),
)
async def improve(request: ImproveRequest, req: Request) -> StreamingResponse:
    """Stream LLM-improved text for the supplied resume section.

    Args:
        request: Validated ``ImproveRequest`` body.
        req:     Raw FastAPI request (for request-id extraction).

    Returns:
        ``StreamingResponse`` with SSE token events.
    """
    request_id: str = req.headers.get("X-Request-ID", "")

    return StreamingResponse(
        resume_improve_service.stream_improve(
            section_type=request.section_type,
            current_text=request.current_text,
            resume_context=request.resume_context,
            request_id=request_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering for SSE
        },
    )
