"""Retry policy.

Centralizes retry behavior for the AI explain-score chain.

Design
* Attempts to invoke raw LLM up to 2 times:
  - Attempt 1: Initial call.
  - Attempt 2: Corrective prompt run (appends correction message to inputs).
* Truncated JSON is repaired using clean/balance/salvage helpers.
* Missing recommendation fields are repaired locally without extra LLM calls.
* Retries only trigger on output validation/schema failures. Provider failures
  are propagated immediately.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from app.ai.chains.base_chain import invoke_with_tracing
from app.ai.guardrails.output_guardrails import OutputValidationError, validate_output
from app.core.errors import AIGenerationError
from app.core.logging import logger
from app.schemas.ai import ATSExplanation

# ---------------------------------------------------------------------------
# Retry correction message
# ---------------------------------------------------------------------------

_CORRECTION_MESSAGE: str = (
    "\n\n---\nCORRECTION REQUIRED: Your previous response did not conform to the "
    "required JSON schema or was incomplete. Return ONLY the valid JSON object. "
    "Do not include any text, markdown, or explanation outside the JSON structure. "
    "All required fields (strengths, weaknesses, section_explanations, suggestions, summary, recommendations) "
    "must be present and non-empty. Every recommendation must include every required field: "
    "priority, issue, why, copy_paste_content, placement, ats_impact."
)


# ---------------------------------------------------------------------------
# Helpers for JSON cleaning, balancing, and repairing
# ---------------------------------------------------------------------------


def clean_json_text(text: str) -> str:
    """Strip markdown formatting and whitespace around JSON text."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _balance_braces(text: str) -> dict | None:
    """Close any open strings and braces/brackets in truncated JSON."""
    in_string = False
    escape = False
    stack = []
    clean_chars = []

    for char in text:
        clean_chars.append(char)
        if char == '"' and not escape:
            in_string = not in_string
        if char == '\\' and in_string:
            escape = not escape
        else:
            escape = False

        if not in_string:
            if char in ("{", "["):
                stack.append(char)
            elif char in ("}", "]"):
                if stack:
                    top = stack[-1]
                    if (char == "}" and top == "{") or (char == "]" and top == "["):
                        stack.pop()

    reconstructed = list(clean_chars)
    if in_string:
        reconstructed.append('"')

    for op in reversed(stack):
        if op == "{":
            reconstructed.append("}")
        elif op == "[":
            reconstructed.append("]")

    final_str = "".join(reconstructed)
    try:
        return json.loads(final_str)
    except Exception:
        return None


def repair_truncated_json(raw_text: str) -> dict | None:
    """Salvage complete elements from a truncated JSON recommendations list."""
    text = clean_json_text(raw_text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r'"recommendations"\s*:\s*\[', text)
    if not match:
        return _balance_braces(text)

    recs_start = match.end()
    prefix = text[:recs_start]
    recs_part = text[recs_start:]

    complete_objects = []
    current_obj = []
    depth = 0
    in_string = False
    escape = False

    for char in recs_part:
        current_obj.append(char)
        if char == '"' and not escape:
            in_string = not in_string
        if char == "\\" and in_string:
            escape = not escape
        else:
            escape = False

        if not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    obj_str = "".join(current_obj).strip()
                    if obj_str.startswith(","):
                        obj_str = obj_str[1:].strip()
                    try:
                        json.loads(obj_str)
                        complete_objects.append(obj_str)
                    except Exception:
                        pass
                    current_obj = []

    reconstructed_recs = ", ".join(complete_objects)
    reconstructed_json_str = prefix + reconstructed_recs + "]}"
    try:
        return json.loads(reconstructed_json_str)
    except Exception:
        return _balance_braces(text)


def _get_fallback_value(field: str, rec: dict) -> str:
    """Return an appropriate default fallback value for a missing recommendation field."""
    fallback_map = {
        "priority": "Medium",
        "issue": "Resume optimization suggestion.",
        "why": "Improves ATS parsing and keyword index matches.",
        "copy_paste_content": "Add relevant certifications and professional achievements.",
        "placement": "In the relevant section.",
        "ats_impact": "+5 points",
    }
    return rec.get(field) or fallback_map.get(field, "")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_with_retry(
    prompt: ChatPromptTemplate,
    llm: Any,
    inputs: dict[str, Any],
    resume_text: str,
    jd_text: str,
    *,
    tags: list[str] | None = None,
    metadata: dict[str, str] | None = None,
) -> ATSExplanation:
    """Invoke the raw LLM with automatic retry and JSON repair capability.

    Attempts:
        1. Original prompt run.
        2. Corrective prompt run (appends correction message to inputs).
    """
    raw_llm_chain = prompt | llm
    attempts = 2
    last_error = None

    for attempt in range(1, attempts + 1):
        logger.info(
            "[RetryPolicy] Executing LLM prompt invocation: attempt %d of %d",
            attempt,
            attempts,
        )

        current_inputs = inputs
        if attempt == 2:
            current_inputs = _build_retry_inputs(inputs)

        attempt_tags = (tags or []) + [f"attempt_{attempt}"]

        try:
            # 1. Invoke raw LLM (Propagate provider network/timeout failures immediately)
            raw_message = invoke_with_tracing(
                raw_llm_chain,
                current_inputs,
                tags=attempt_tags,
                metadata=metadata,
            )
        except Exception as exc:
            # Do NOT retry on provider timeouts, rate limits, or network errors
            logger.error(
                "[RetryPolicy] LiteLLM provider or network failure encountered during LLM invocation. "
                "Propagating immediately without application retry. Exception: %s",
                exc,
            )
            raise exc

        try:
            raw_text = raw_message.content
            response_metadata = raw_message.response_metadata or {}
            token_usage = response_metadata.get("token_usage", {})
            completion_tokens = (
                token_usage.get("completion_tokens")
                or response_metadata.get("completion_tokens")
            )
            finish_reason = response_metadata.get("finish_reason")

            logger.info(
                "[RetryPolicy] Attempt %d - Raw Response Length: %d chars, "
                "Completion Tokens: %s, Finish Reason: %s",
                attempt,
                len(raw_text),
                completion_tokens,
                finish_reason,
            )

            if finish_reason and finish_reason != "stop":
                logger.warning(
                    "[RetryPolicy] Warning: LLM output was truncated. Finish reason: %s",
                    finish_reason,
                )

            # 2. JSON Pre-validation & Truncation Repair
            cleaned_text = clean_json_text(raw_text)
            parsed_dict = None
            try:
                parsed_dict = json.loads(cleaned_text)
            except json.JSONDecodeError as json_err:
                logger.warning(
                    "[RetryPolicy] JSON decode failed: %s. Attempting truncation repair.",
                    json_err,
                )
                parsed_dict = repair_truncated_json(cleaned_text)
                if not parsed_dict:
                    logger.warning("[RetryPolicy] JSON truncation repair failed.")
                    raise OutputValidationError(
                        message=f"Invalid JSON format and truncation repair failed: {json_err}",
                        field="json",
                    ) from json_err
                logger.info("[RetryPolicy] JSON truncation repair succeeded.")

            # 3. Verify recommendations and perform local repair if needed
            recs = parsed_dict.get("recommendations", [])
            if isinstance(recs, list):
                repaired_recs = []
                for rec_idx, rec in enumerate(recs):
                    if not isinstance(rec, dict):
                        continue
                    missing_fields = [
                        f
                        for f in [
                            "priority",
                            "issue",
                            "why",
                            "copy_paste_content",
                            "placement",
                            "ats_impact",
                        ]
                        if not rec.get(f)
                    ]
                    if missing_fields:
                        logger.warning(
                            "[RetryPolicy] Recommendation at index %d missing fields %s. "
                            "Repairing locally using fallback defaults.",
                            rec_idx,
                            missing_fields,
                        )
                        repaired_rec = dict(rec)
                        for field in ["priority", "issue", "why", "copy_paste_content", "placement", "ats_impact"]:
                            if not repaired_rec.get(field):
                                repaired_rec[field] = _get_fallback_value(field, rec)
                        repaired_recs.append(repaired_rec)
                    else:
                        repaired_recs.append(rec)
                parsed_dict["recommendations"] = repaired_recs

            logger.debug("[RetryPolicy] Parsed JSON output dict: %s", parsed_dict)

            # 4. Pydantic validation
            explanation = ATSExplanation(**parsed_dict)

            # 5. Output Guardrails check
            validate_output(explanation)

            logger.info("[RetryPolicy] Attempt %d validation succeeded.", attempt)
            return explanation

        except (json.JSONDecodeError, ValidationError, OutputValidationError, TypeError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "[RetryPolicy] Attempt %d failed output schema/validation check: %s. Retry reason: %s",
                attempt,
                exc,
                type(exc).__name__,
            )

    logger.error("[RetryPolicy] All %d attempts failed. Raising AIGenerationError.", attempts)
    raise AIGenerationError(
        message=f"AI explanation could not be generated after {attempts} attempts. Last error: {last_error}",
        metadata={"attempts": attempts, "last_error": str(last_error)},
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_retry_inputs(original_inputs: dict[str, Any]) -> dict[str, Any]:
    """Append the correction instruction to the JD context field."""
    retry_inputs = dict(original_inputs)
    original_context = str(retry_inputs.get("jd_context", ""))
    retry_inputs["jd_context"] = original_context + _CORRECTION_MESSAGE
    return retry_inputs
