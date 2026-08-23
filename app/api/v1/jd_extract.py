"""JD URL extraction route — POST /v1/jd/extract

Thin controller: validate → delegate to jd_extract_service → return response.
Authentication is enforced by InternalAuthMiddleware; no duplication here.
"""

from fastapi import APIRouter
from app.schemas.jd_extract import JDExtractRequest, JDExtractResponse
from app.services import jd_extract_service

router = APIRouter(prefix="/v1/jd", tags=["JD Extraction"])


@router.post(
    "/extract",
    response_model=JDExtractResponse,
    summary="Extract job description text from a public URL",
    description=(
        "Fetches the given URL and returns the main job description text. "
        "Uses a 2-tier strategy: httpx + trafilatura for static pages, "
        "playwright headless Chromium for JS-rendered pages (LinkedIn, Indeed, etc.). "
        "Returns HTTP 422 if neither tier can extract usable text."
    ),
)
async def extract_jd(request: JDExtractRequest) -> JDExtractResponse:
    """Extract the job description from the supplied public URL.

    Args:
        request: Validated JDExtractRequest with the target URL.

    Returns:
        JDExtractResponse with extracted text, source tier, and char count.

    Raises:
        AppError (422): If both scraping tiers fail to produce usable text.
    """
    return await jd_extract_service.extract_from_url(str(request.url))
