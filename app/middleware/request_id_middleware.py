"""Request ID middleware — generates and propagates unique request identifiers."""

import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

# Context variable accessible anywhere in the request lifecycle
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")

REQUEST_ID_HEADER = "X-Request-ID"


def get_request_id() -> str:
    """Return the current request ID from context."""
    return request_id_ctx.get()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assign a unique request ID to every request.

    - Accepts client-provided X-Request-ID if present.
    - Generates a UUID4 if not provided.
    - Stores the ID in a ContextVar for downstream access.
    - Returns the ID in the X-Request-ID response header.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        incoming_id = request.headers.get(REQUEST_ID_HEADER)
        rid = incoming_id if incoming_id else str(uuid.uuid4())

        token = request_id_ctx.set(rid)
        try:
            response = await call_next(request)
            response.headers[REQUEST_ID_HEADER] = rid
            return response
        finally:
            request_id_ctx.reset(token)
