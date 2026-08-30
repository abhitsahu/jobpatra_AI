"""Entity extraction chain — LCEL pipeline for AI-driven keyword extraction.

This module provides two public functions consumed exclusively by
``ats_service._extract_entities_hybrid()``:

    extract_resume_entities(resume_text) -> ResumeExtraction
    extract_jd_entities(jd_text)         -> JDExtraction

Pipeline (per function)
-----------------------
1. Size guard    — reject inputs that would blow the LLM context window.
2. LCEL chain    — Prompt | LiteLLM (via existing ``get_chat_model``).
3. Tracing       — ``invoke_with_tracing`` attaches LangSmith callbacks.
4. JSON repair   — ``_parse_extraction_json`` strips markdown fences, repairs
                   truncated JSON, falls back to brace-balancing.
5. Pydantic      — validates repaired dict against ``ResumeExtraction`` /
                   ``JDExtraction`` with lenient defaults.

On ANY failure (network, timeout, bad JSON, schema mismatch) this module raises
``AIGenerationError`` and nothing else.  The service layer catches that error
and falls back to the naive extractors — so the caller never receives a partial
or corrupt extraction result.

This module does NOT:
  - implement retry logic
  - call the explain-score chain
  - import from FastAPI
  - access the database or file system
"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from app.ai.chains.base_chain import invoke_with_tracing
from app.ai.prompts.extract_entities_v1 import (
    JD_EXTRACTION_PROMPT,
    RESUME_EXTRACTION_PROMPT,
)
from app.ai.providers.litellm_client import get_chat_model
from app.core.errors import AIGenerationError
from app.core.logging import logger
from app.schemas.extraction import JDExtraction, ResumeExtraction


# ---------------------------------------------------------------------------
# Size limits — deliberately softer than the full input guardrail.
# Extraction prompts are cheaper to run; we don't need to truncate aggressively.
# ---------------------------------------------------------------------------

_MAX_RESUME_CHARS: int = 25_000
"""Hard ceiling on resume text fed to the extraction prompt."""

_MAX_JD_CHARS: int = 12_000
"""Hard ceiling on JD text fed to the extraction prompt."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_resume_entities(resume_text: str) -> ResumeExtraction:
    """Run the AI resume extraction chain and return structured entities.

    Args:
        resume_text: Cleaned, normalised resume text (output of text_cleaner).

    Returns:
        ``ResumeExtraction`` with hard_skills, soft_skills, domain_terms, etc.

    Raises:
        AIGenerationError: On any extraction or validation failure. The caller
            is expected to catch this and fall back to the naive extractor.
    """
    _guard_size(resume_text, limit=_MAX_RESUME_CHARS, field="resume")
    logger.info("[HybridExtract] Running AI resume extraction")

    try:
        llm = get_chat_model()
        chain = RESUME_EXTRACTION_PROMPT | llm
        raw_message = invoke_with_tracing(
            chain,
            {"resume_text": resume_text[:_MAX_RESUME_CHARS]},
            tags=["extract_resume_v1"],
            metadata={"extractor": "resume"},
        )
        parsed = _parse_extraction_json(raw_message.content, label="resume")
        result = ResumeExtraction(**parsed)
        logger.info(
            "[HybridExtract] Resume extraction OK — hard_skills=%d soft_skills=%d domain_terms=%d",
            len(result.hard_skills),
            len(result.soft_skills),
            len(result.domain_terms),
        )
        return result
    except AIGenerationError:
        raise
    except (ValidationError, TypeError, ValueError) as exc:
        raise AIGenerationError(
            message=f"Resume extraction schema validation failed: {exc}",
            metadata={"exc": str(exc)},
        ) from exc
    except Exception as exc:
        raise AIGenerationError(
            message=f"Resume extraction chain failed: {exc}",
            metadata={"exc": str(exc)},
        ) from exc


def extract_jd_entities(jd_text: str) -> JDExtraction:
    """Run the AI JD extraction chain and return structured hiring requirements.

    Args:
        jd_text: Cleaned, normalised job description text.

    Returns:
        ``JDExtraction`` with required_hard_skills, domain_terms, etc.

    Raises:
        AIGenerationError: On any extraction or validation failure.
    """
    _guard_size(jd_text, limit=_MAX_JD_CHARS, field="job_description")
    logger.info("[HybridExtract] Running AI JD extraction")

    try:
        llm = get_chat_model()
        chain = JD_EXTRACTION_PROMPT | llm
        raw_message = invoke_with_tracing(
            chain,
            {"jd_text": jd_text[:_MAX_JD_CHARS]},
            tags=["extract_jd_v1"],
            metadata={"extractor": "jd"},
        )
        parsed = _parse_extraction_json(raw_message.content, label="jd")
        result = JDExtraction(**parsed)
        logger.info(
            "[HybridExtract] JD extraction OK — required=%d preferred=%d domain=%d",
            len(result.required_hard_skills),
            len(result.preferred_hard_skills),
            len(result.domain_terms),
        )
        return result
    except AIGenerationError:
        raise
    except (ValidationError, TypeError, ValueError) as exc:
        raise AIGenerationError(
            message=f"JD extraction schema validation failed: {exc}",
            metadata={"exc": str(exc)},
        ) from exc
    except Exception as exc:
        raise AIGenerationError(
            message=f"JD extraction chain failed: {exc}",
            metadata={"exc": str(exc)},
        ) from exc


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _guard_size(text: str, limit: int, field: str) -> None:
    """Raise ``AIGenerationError`` if ``text`` is blank or over the size limit.

    Args:
        text:  Input text to validate.
        limit: Maximum allowed character count.
        field: Human-readable field name for error messages.
    """
    if not text or not text.strip():
        raise AIGenerationError(
            message=f"Extraction input '{field}' must not be empty.",
            metadata={"field": field},
        )
    if len(text) > limit:
        logger.warning(
            "[HybridExtract] '%s' exceeds extraction limit (%d > %d chars). "
            "Truncating to limit.",
            field,
            len(text),
            limit,
        )
        # We truncate rather than hard-fail — the extraction will still run on
        # the first `limit` chars. Hard failures here would bypass the fallback.


def _clean_json_text(text: str) -> str:
    """Strip markdown fences and leading/trailing whitespace from LLM output."""
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _balance_braces(text: str) -> dict | None:
    """Attempt to close open strings and braces in a truncated JSON string.

    Mirrors the repair logic in ``output_guardrails.py``.
    """
    in_string = False
    escape = False
    stack: list[str] = []
    chars: list[str] = []

    for char in text:
        chars.append(char)
        if char == '"' and not escape:
            in_string = not in_string
        escape = (char == "\\") and in_string and not escape

        if not in_string:
            if char in ("{", "["):
                stack.append(char)
            elif char in ("}", "]"):
                if stack:
                    top = stack[-1]
                    if (char == "}" and top == "{") or (char == "]" and top == "["):
                        stack.pop()

    if in_string:
        chars.append('"')
    for op in reversed(stack):
        chars.append("}" if op == "{" else "]")

    try:
        return json.loads("".join(chars))
    except Exception:
        return None


def _parse_extraction_json(raw_text: str, label: str) -> dict:
    """Parse raw LLM output into a dict, applying repair on JSON failures.

    Args:
        raw_text: Raw string content from the LLM message.
        label:    'resume' or 'jd' — used only for log messages.

    Returns:
        Parsed dict ready for Pydantic model instantiation.

    Raises:
        AIGenerationError: If JSON cannot be parsed even after repair.
    """
    cleaned = _clean_json_text(raw_text)

    # Fast path — clean, valid JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Repair path — try to extract the first JSON object from surrounding text
    logger.warning(
        "[HybridExtract] JSON decode failed for '%s'. Attempting repair.", label
    )
    obj_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if obj_match:
        try:
            return json.loads(obj_match.group())
        except json.JSONDecodeError:
            pass

    # Last resort — brace balancer
    repaired = _balance_braces(cleaned)
    if repaired is not None:
        logger.info(
            "[HybridExtract] JSON repair succeeded for '%s' via brace balancer.", label
        )
        return repaired

    raise AIGenerationError(
        message=f"Extraction JSON could not be parsed or repaired for '{label}'.",
        metadata={"raw_text": raw_text[:500]},
    )
