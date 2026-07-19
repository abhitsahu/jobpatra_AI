"""ATS route — POST /v1/ats/analyze

This module is intentionally thin.  It:
  1. Accepts the validated request body.
  2. Delegates to ``ats_service.analyze()``.
  3. Returns the typed response.
  4. Registers an exception handler for ``InvalidInputError`` (→ HTTP 422).

Authentication is enforced by ``InternalAuthMiddleware`` — this route does
not duplicate that logic.  Any request that reaches here has already been
authenticated.

This module does NOT:
  - implement parsing logic
  - implement scoring logic
  - implement matching logic
  - call AI or external services
"""

from typing import Any
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.core.errors import InvalidInputError
from app.schemas.ats import ATSAnalyzeRequest, ATSAnalyzeResponse
from app.services import ats_service

router = APIRouter(prefix="/v1/ats", tags=["ATS"])


@router.post(
    "/analyze",
    response_model=ATSAnalyzeResponse,
    summary="Run deterministic ATS analysis with AI explanation",
    description=(
        "Execute the full ATS pipeline (parse → normalize → extract → match → score) "
        "and return a structured ATS report. "
        "An AI explanation is appended when available. "
        "Input guardrails reject oversized or injected inputs (HTTP 422). "
        "AI failures degrade gracefully — the deterministic report is always returned."
    ),
)
async def analyze(request: ATSAnalyzeRequest, req: Request) -> Any:
    """Run the deterministic ATS pipeline for the supplied resume and JD.

    Args:
        request: Validated ``ATSAnalyzeRequest`` with resume and job description.
        req: Raw request object for checking accept headers.

    Returns:
        ``ATSAnalyzeResponse`` with all scores, matched/missing keywords,
        skill coverage, extracted resume metadata, and optional AI explanation.
        Or a ``StreamingResponse`` if streaming is requested.
    """
    is_stream = request.stream or "text/event-stream" in req.headers.get("accept", "").lower()

    if is_stream:
        # Before yielding, parse and run input validation to fail-fast with HTTP 422
        # directly if input is invalid. This keeps error handling clean.
        from app.analysis.normalization import text_cleaner
        from app.analysis.normalization.jd_normalizer import normalize as normalize_jd
        from app.ai.guardrails.input_guardrails import validate_all as validate_input

        resume_raw = ats_service._parse_resume(request)
        resume_clean = text_cleaner.clean(resume_raw)
        jd_clean = normalize_jd(request.job_description.text)

        validate_input(resume_clean, jd_clean)

        return StreamingResponse(
            ats_service.analyze_stream(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    return ats_service.analyze(request)
