"""Resume improvement service.

Orchestrates the resume improvement pipeline and produces an SSE-encoded
async generator suitable for FastAPI's ``StreamingResponse``.

SSE wire format emitted:
  * Per token:    ``data: {"token": "<text>"}\n\n``
  * On success:   ``data: [DONE]\n\n``
  * On error:     ``event: error\ndata: {"code": "...", "message": "..."}\n\n``

This module does NOT:
* Know about HTTP (no FastAPI imports).
* Validate section type — the caller (route layer) is responsible.
* Interact with the database.
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

from app.ai.chains.resume_improve_chain import stream_improve_section
from app.ai.streaming.sse_encoder import encode_error
from app.core.logging import logger


async def stream_improve(
    section_type: str,
    current_text: str,
    resume_context: str | None = None,
    *,
    request_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Generate SSE-encoded token events for a resume section improvement.

    Args:
        section_type:   Resume section identifier (``"summary"``, ``"experience"``, etc.).
        current_text:   The original section text to improve.
        resume_context: Optional full-resume string for extra LLM context.
        request_id:     Optional request ID for logging/tracing.

    Yields:
        SSE-formatted strings, terminated by ``[DONE]``.
    """
    rid = request_id or "unknown"
    logger.info(
        "[ResumeImproveService] [%s] Streaming improvement for section=%s",
        rid[:8],
        section_type,
    )

    try:
        async for token in stream_improve_section(
            section_type=section_type,
            current_text=current_text,
            resume_context=resume_context,
            request_id=request_id,
        ):
            # Emit one SSE data event per token
            payload = json.dumps({"token": token}, ensure_ascii=False, separators=(",", ":"))
            yield f"data: {payload}\n\n"

        # Signal completion
        yield "data: [DONE]\n\n"
        logger.info("[ResumeImproveService] [%s] Stream complete", rid[:8])

    except Exception as exc:
        logger.exception(
            "[ResumeImproveService] [%s] LLM error: %s",
            rid[:8],
            exc,
        )
        yield encode_error(
            message="AI service encountered an error. Please try again.",
            code="LLM_ERROR",
        )
