"""Centralized application errors and FastAPI exception handlers.

All custom errors inherit from AppError. Routes raise these errors;
exception handlers convert them into standardized JSON responses.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logging import logger
from app.middleware.request_id_middleware import get_request_id


# ---------------------------------------------------------------------------
# Application error hierarchy
# ---------------------------------------------------------------------------


class AppError(Exception):
    """Base application error. All custom errors inherit from this."""

    def __init__(
        self,
        message: str = "An unexpected error occurred.",
        code: str = "INTERNAL_ERROR",
        status_code: int = 500,
    ) -> None:
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(message)


class ValidationError(AppError):
    """Raised when input validation fails."""

    def __init__(self, message: str = "Validation failed.") -> None:
        super().__init__(message=message, code="VALIDATION_ERROR", status_code=400)


class UnauthorizedError(AppError):
    """Raised when authentication is missing or invalid."""

    def __init__(self, message: str = "Unauthorized.") -> None:
        super().__init__(message=message, code="UNAUTHORIZED", status_code=401)


class NotFoundError(AppError):
    """Raised when a requested resource does not exist."""

    def __init__(self, message: str = "Resource not found.") -> None:
        super().__init__(message=message, code="NOT_FOUND", status_code=404)


class InternalServerError(AppError):
    """Raised for unrecoverable internal failures."""

    def __init__(self, message: str = "Internal server error.") -> None:
        super().__init__(message=message, code="INTERNAL_ERROR", status_code=500)


class UnparsableDocumentError(AppError):
    """Raised when a document cannot be converted to plain text.

    Examples:
        - Scanned image PDF with no selectable text layer.
        - Corrupt or password-protected DOCX file.
    """

    def __init__(self, message: str = "Document contains no extractable text.") -> None:
        super().__init__(message=message, code="UNPARSABLE_DOCUMENT", status_code=422)


class InvalidInputError(AppError):
    """Raised by AI input guardrails before any LLM call is made.

    Use cases:
        - Resume or JD text exceeds the configured size limit.
        - Input is empty or blank.
        - Basic prompt injection pattern detected.

    The LLM is NEVER called when this error is raised.
    The HTTP response is 422 Unprocessable Entity.
    """

    def __init__(
        self,
        message: str = "Input validation failed.",
        metadata: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message=message, code="INVALID_INPUT", status_code=422)
        self.metadata: dict[str, object] = metadata or {}


class AIGenerationError(Exception):
    """Raised when the AI layer fails to produce a valid structured response.

    This is NOT an ``AppError`` — it must NOT propagate through the FastAPI
    exception handler.  It is caught inside ``ats_service.analyze()`` and
    converted into ``ai_status="unavailable"`` while the deterministic ATS
    report is still returned with HTTP 200.

    Use cases:
        - Output guardrail validation and local repair fails.
        - LLM returns malformed JSON that cannot be parsed into ``ATSExplanation``.

    This error NEVER causes the endpoint to fail.
    """

    def __init__(
        self,
        message: str = "AI generation failed.",
        code: str = "AI_GENERATION_ERROR",
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.message = message
        self.code = code
        self.metadata: dict[str, object] = metadata or {}
        super().__init__(message)


# ---------------------------------------------------------------------------
# Exception handlers — registered on the FastAPI app
# ---------------------------------------------------------------------------


def _build_error_response(error: AppError) -> JSONResponse:
    """Convert an AppError into a standardized JSON response."""
    return JSONResponse(
        status_code=error.status_code,
        content={
            "error": {
                "code": error.code,
                "message": error.message,
            }
        },
    )


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Handle all AppError subclasses."""
    rid = get_request_id()
    logger.error("[%s] %s: %s", rid[:8] if rid else "-", exc.code, exc.message)
    return _build_error_response(exc)


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unexpected exceptions."""
    rid = get_request_id()
    logger.exception("[%s] Unhandled exception: %s", rid[:8] if rid else "-", str(exc))
    return _build_error_response(InternalServerError())


def register_error_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the FastAPI application."""
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
