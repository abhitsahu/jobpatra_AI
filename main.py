"""JobPatra AI — Application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.ats import router as ats_router
from app.api.v1.health import router as health_router
from app.api.v1.jd_extract import router as jd_extract_router
from app.core.config import settings
from app.core.errors import register_error_handlers
from app.core.logging import logger, setup_logging
from app.analysis.matching.semantic_matcher import (
    clear_shared_provider,
    initialize_shared_provider,
)
from app.services.taxonomy_service import initialize_taxonomy_service
from app.middleware.internal_auth_middleware import InternalAuthMiddleware
from app.middleware.logging_middleware import LoggingMiddleware
from app.middleware.request_id_middleware import RequestIDMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    setup_logging()
    logger.info(
        "%s starting — env=%s host=%s port=%s",
        settings.APP_NAME,
        settings.ENV,
        settings.HOST,
        settings.PORT,
    )
    initialize_taxonomy_service()
    initialize_shared_provider()
    yield
    clear_shared_provider()
    logger.info("%s shutting down", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Middleware — registered in REVERSE order (last added = first executed)
#
# Execution order for incoming requests:
#   1. RequestIDMiddleware   — assign/propagate request ID
#   2. LoggingMiddleware     — log request with timing
#   3. InternalAuthMiddleware — validate service token
#   4. Route handler
# ---------------------------------------------------------------------------

app.add_middleware(InternalAuthMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RequestIDMiddleware)

# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

register_error_handlers(app)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app.include_router(health_router)
app.include_router(ats_router)
app.include_router(jd_extract_router)
