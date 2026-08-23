"""Pydantic schemas for the JD URL extraction endpoint (POST /v1/jd/extract).

Only plain URL input → extracted plain text output.
No ATS logic lives here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, HttpUrl, field_validator


class JDExtractRequest(BaseModel):
    """Request body for POST /v1/jd/extract."""

    url: HttpUrl
    """The public URL of the job posting page."""


class JDExtractResponse(BaseModel):
    """Response body for POST /v1/jd/extract."""

    text: str
    """Extracted plain text of the job description."""

    source: Literal["httpx", "playwright"]
    """Which tier was used — 'httpx' (static page) or 'playwright' (JS-rendered page)."""

    char_count: int
    """Length of the extracted text in characters."""

    url: str
    """The original URL that was scraped (stringified)."""
