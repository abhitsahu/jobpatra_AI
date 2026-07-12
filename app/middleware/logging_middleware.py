"""Logging middleware — structured request/response logging with latency tracking."""

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import logger
from app.middleware.request_id_middleware import get_request_id


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request with method, path, status code, and latency.

    Integrates with RequestIDMiddleware — must be registered AFTER it
    in the middleware stack so the request ID is already available.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start = time.perf_counter()
        method = request.method
        path = request.url.path

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start) * 1000
        rid = get_request_id()

        logger.info(
            "[%s] %s %s → %s (%.1fms)",
            rid[:8] if rid else "-",
            method,
            path,
            response.status_code,
            elapsed_ms,
        )

        return response
