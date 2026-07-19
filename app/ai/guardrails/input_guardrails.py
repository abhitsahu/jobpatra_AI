"""Input guardrails.

Validates all AI input BEFORE any LLM call is made.

Design
------
* Purely deterministic — no LLM involved.
* Raises ``InvalidInputError`` on any violation.
* Keeps the LLM completely safe from malformed, oversized, or injected input.

This file does NOT:
  - Call any LLM or external service.
  - Implement output validation.
  - Implement retry logic.
  - Implement sophisticated NLP-based injection detection.

Limits (all configurable at module level)
-----------------------------------------
``MAX_RESUME_CHARS``  — absolute upper bound on raw resume text length.
``MAX_JD_CHARS``      — absolute upper bound on job-description text length.
``MAX_COMBINED_CHARS``— combined limit (resume + JD) to protect prompt size.
``MIN_CONTENT_CHARS`` — minimum meaningful content length.
"""

from __future__ import annotations

import re

from app.core.errors import InvalidInputError
from app.core.logging import logger

# Configuration constants

MAX_RESUME_CHARS: int = 30_000
"""Maximum allowed resume text length in characters."""

MAX_JD_CHARS: int = 15_000
"""Maximum allowed job-description text length in characters."""

MAX_COMBINED_CHARS: int = 40_000
"""Maximum combined (resume + JD) character count fed to the prompt."""

MIN_CONTENT_CHARS: int = 20
"""Minimum characters required to be considered non-empty."""


# Prompt injection patterns


# Each pattern is a compiled regex.  Matching any single pattern is enough
# to reject the input.  These are intentionally conservative — they only
# catch the most obvious injection attempts and known jailbreak phrases.
# A false-positive is always recoverable (user resubmits); a missed injection
# could poison the trace.
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(previous|prior|above|all)\s+instructions?", re.IGNORECASE),
    re.compile(r"you\s+are\s+(now\s+)?(chat ?gpt|gpt-?4|an?\s+ai|an?\s+assistant)", re.IGNORECASE),
    re.compile(r"forget\s+(your|the|all|previous)\s+(system\s+)?prompt", re.IGNORECASE),
    re.compile(r"\bact\s+as\b.{0,30}(ai|bot|assistant|model|gpt)", re.IGNORECASE),
    re.compile(r"\bdeveloper\s+mode\b", re.IGNORECASE),
    re.compile(r"\breveal\s+(your\s+)?(system\s+)?prompt", re.IGNORECASE),
    re.compile(r"\bexecute\s+(command|shell|bash|python|code)\b", re.IGNORECASE),
    re.compile(r"^system\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"DAN\s*mode", re.IGNORECASE),
]


# Public API


def validate_resume(text: str) -> None:
    """Validate resume text before it reaches the LLM.

    Checks (in order):
    1. Not empty / blank.
    2. Does not exceed ``MAX_RESUME_CHARS``.
    3. Does not contain basic prompt injection patterns.

    Args:
        text: Raw resume text (may be noisy, not yet normalized).

    Raises:
        InvalidInputError: On any validation failure.
    """
    _check_not_empty(text, field="resume")
    _check_max_length(text, limit=MAX_RESUME_CHARS, field="resume")
    _check_injection(text, field="resume")


def validate_job_description(text: str) -> None:
    """Validate job-description text before it reaches the LLM.

    Checks (in order):
    1. Not empty / blank.
    2. Does not exceed ``MAX_JD_CHARS``.
    3. Does not contain basic prompt injection patterns.

    Args:
        text: Raw job description text.

    Raises:
        InvalidInputError: On any validation failure.
    """
    _check_not_empty(text, field="job_description")
    _check_max_length(text, limit=MAX_JD_CHARS, field="job_description")
    _check_injection(text, field="job_description")


def validate_combined_length(resume_text: str, jd_text: str) -> None:
    """Validate that the combined input fits within the prompt size limit.

    Even if each piece is individually within its own limit, the sum may
    still exceed what the LLM can handle without degraded output quality.

    Args:
        resume_text: Clean resume text.
        jd_text:     Clean job-description text.

    Raises:
        InvalidInputError: If combined length exceeds ``MAX_COMBINED_CHARS``.
    """
    combined = len(resume_text) + len(jd_text)
    if combined > MAX_COMBINED_CHARS:
        logger.warning(
            "[InputGuardrail] Combined input too large: %d chars (limit %d)",
            combined,
            MAX_COMBINED_CHARS,
        )
        raise InvalidInputError(
            message=(
                f"Combined resume and job description length ({combined} chars) "
                f"exceeds the limit of {MAX_COMBINED_CHARS} chars."
            ),
            metadata={"combined_chars": combined, "limit": MAX_COMBINED_CHARS},
        )


def validate_all(resume_text: str, jd_text: str) -> None:
    """Run all input guardrail checks in one call.

    This is the primary entry point used by ``explain_score_chain``.

    Order of checks:
    1. Resume: emptiness, max-length, injection.
    2. JD: emptiness, max-length, injection.
    3. Combined: total length.

    Args:
        resume_text: Raw resume text.
        jd_text:     Raw job-description text.

    Raises:
        InvalidInputError: On the first failed check.
    """
    validate_resume(resume_text)
    validate_job_description(jd_text)
    validate_combined_length(resume_text, jd_text)


# Private helpers


def _check_not_empty(text: str, field: str) -> None:
    """Raise ``InvalidInputError`` if ``text`` is blank."""
    if not text or not text.strip():
        logger.warning("[InputGuardrail] Empty %s rejected.", field)
        raise InvalidInputError(
            message=f"{field} must not be empty.",
            metadata={"field": field},
        )
    if len(text.strip()) < MIN_CONTENT_CHARS:
        logger.warning(
            "[InputGuardrail] %s too short: %d chars (min %d).",
            field,
            len(text.strip()),
            MIN_CONTENT_CHARS,
        )
        raise InvalidInputError(
            message=(
                f"{field} is too short ({len(text.strip())} chars). "
                f"Minimum is {MIN_CONTENT_CHARS} chars."
            ),
            metadata={"field": field, "length": len(text.strip()), "min": MIN_CONTENT_CHARS},
        )


def _check_max_length(text: str, limit: int, field: str) -> None:
    """Raise ``InvalidInputError`` if ``text`` exceeds ``limit`` characters."""
    if len(text) > limit:
        logger.warning(
            "[InputGuardrail] %s too large: %d chars (limit %d).",
            field,
            len(text),
            limit,
        )
        raise InvalidInputError(
            message=(
                f"{field} is too large ({len(text)} chars). "
                f"Maximum allowed is {limit} chars."
            ),
            metadata={"field": field, "length": len(text), "limit": limit},
        )


def _check_injection(text: str, field: str) -> None:
    """Raise ``InvalidInputError`` if a basic injection pattern is detected."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning(
                "[InputGuardrail] Prompt injection pattern detected in %s: %r",
                field,
                pattern.pattern,
            )
            raise InvalidInputError(
                message=f"{field} contains disallowed content.",
                metadata={"field": field, "pattern": pattern.pattern},
            )
