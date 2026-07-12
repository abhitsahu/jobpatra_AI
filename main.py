"""JobPatra AI — Application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.health import router as health_router
from app.core.config import settings
from app.core.logging import logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    setup_logging()
    logger.info(
        "JobPatra AI starting — env=%s host=%s port=%s",
        settings.ENV,
        settings.HOST,
        settings.PORT,
    )
    yield
    logger.info("JobPatra AI shutting down")


app = FastAPI(
    title="JobPatra AI",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
