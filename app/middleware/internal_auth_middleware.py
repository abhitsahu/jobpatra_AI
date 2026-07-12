"""Internal authentication middleware — service-to-service token validation.

This middleware validates that requests originate from the trusted Next.js backend.
It does NOT authenticate users, decode JWTs, check sessions, or perform OAuth.

Next.js owns all user authentication. FastAPI only verifies the internal API key.
"""

import secrets

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.config import settings
from app.core.logging import logger
from app.middleware.request_id_middleware import get_request_id

# Paths that bypass internal authentication
PUBLIC_PATHS: frozenset[str] = frozenset({"/health", "/docs", "/openapi.json", "/redoc"})


class InternalAuthMiddleware(BaseHTTPMiddleware):
    """Validate internal service token on every request except public paths.

    Accepts the token via either:
      - X-Internal-API-Key header
      - Authorization: Bearer <token> header

    Rejects with 401 if the token is missing or invalid.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        token = self._extract_token(request)

        if not token:
            logger.warning(
                "[%s] Unauthorized — missing API key for %s %s",
                get_request_id()[:8],
                request.method,
                request.url.path,
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "code": "UNAUTHORIZED",
                        "message": "Missing API key.",
                    }
                },
            )

        if not secrets.compare_digest(token, settings.INTERNAL_API_KEY.get_secret_value()):
            logger.warning(
                "[%s] Unauthorized — invalid API key for %s %s",
                get_request_id()[:8],
                request.method,
                request.url.path,
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "code": "UNAUTHORIZED",
                        "message": "Invalid internal API key.",
                    }
                },
            )

        return await call_next(request)

    @staticmethod
    def _extract_token(request: Request) -> str | None:
        """Extract token from X-Internal-API-Key or Authorization header."""
        api_key = request.headers.get("X-Internal-API-Key")
        if api_key:
            return api_key

        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]

        return None
