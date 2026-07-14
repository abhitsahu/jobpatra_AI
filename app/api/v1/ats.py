"""ATS route — POST /v1/ats/analyze

This module is intentionally thin.  It:
  1. Accepts the validated request body.
  2. Delegates to ``ats_service.analyze()``.
  3. Returns the typed response.

Authentication is enforced by ``InternalAuthMiddleware`` — this route does
not duplicate that logic.  Any request that reaches here has already been
authenticated.

This module does NOT:
  - implement parsing logic
  - implement scoring logic
  - implement matching logic
  - call AI or external services
"""

from fastapi import APIRouter

from app.schemas.ats import ATSAnalyzeRequest, ATSAnalyzeResponse
from app.services import ats_service

router = APIRouter(prefix="/v1/ats", tags=["ATS"])


@router.post(
    "/analyze",
    response_model=ATSAnalyzeResponse,
    summary="Run deterministic ATS analysis",
    description=(
        "Execute the full ATS pipeline (parse → normalize → extract → match → score) "
        "and return a structured ATS report. "
        "No AI. No LLM. Fully deterministic."
    ),
)
async def analyze(request: ATSAnalyzeRequest) -> ATSAnalyzeResponse:
    """Run the deterministic ATS pipeline for the supplied resume and JD.

    Args:
        request: Validated ``ATSAnalyzeRequest`` with resume and job description.

    Returns:
        ``ATSAnalyzeResponse`` containing all scores, matched/missing keywords,
        skill coverage, and extracted resume metadata.
    """
    return ats_service.analyze(request)
