# syntax=docker/dockerfile:1.7

# ── Stage 1: install dependencies ───────────────────────────────────────────
FROM python:3.14-slim AS base

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    PLAYWRIGHT_BROWSERS_PATH=/home/appuser/.cache/ms-playwright

# System libraries required by Playwright Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libnss3 libnspr4 libdbus-1-3 libatk1.0-0 \
    libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 libxcomposite1 \
    libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2 \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: install Python dependencies ────────────────────────────────────
# Copy lock files first so Docker reuses this layer when only app code changes.
FROM base AS dependencies

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ── Stage 3: final runtime image ─────────────────────────────────────────────
FROM base AS runner

# Copy installed packages from the dependency stage
COPY --from=dependencies /app/.venv /app/.venv

# Copy application code
COPY . .

# Create non-root user and fix ownership
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

# Install Playwright Chromium browser binary for the appuser
RUN uv run playwright install chromium

EXPOSE 8000

# Healthcheck — /health is a public path (bypasses InternalAuthMiddleware)
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Use uvicorn directly (single process — t3.micro has 1 vCPU, no benefit from Gunicorn workers)
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]