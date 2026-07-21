"""Output guardrails.

Validates LLM output AFTER the chain returns a result.

* Validates the structured ``ATSExplanation`` Pydantic model for logical
  completeness beyond what Pydantic's type system alone enforces.
* Does NOT retry on failure — retry responsibility is completely removed.
* Returns the validated response unchanged if all checks pass.

Checks
------
* strengths:            non-empty list.
* weaknesses:           non-empty list.
* section_explanations: non-empty list; each entry has a non-blank explanation.
* suggestions:          non-empty list.
* summary:              non-blank string.

This file does NOT:
  - Call any LLM or external service.
  - Implement input validation.
  - Implement retry logic.
"""

from __future__ import annotations

import json
import re
from typing import Any
from pydantic import ValidationError

from app.core.errors import AIGenerationError
from app.core.logging import logger
from app.schemas.ai import ATSExplanation


class OutputValidationError(ValueError):
    """Raised when LLM output fails semantic validation.

    This is NOT an ``AppError``. It is internal to the guardrail pipeline.
    """

    def __init__(self, message: str, field: str = "") -> None:
        self.field = field
        super().__init__(message)


# Public API


def validate_output(explanation: ATSExplanation) -> ATSExplanation:
    """Validate an ``ATSExplanation`` for semantic completeness.

    Pydantic already enforces the type structure.  This function adds
    logical rules that Pydantic cannot express:

    * strengths must contain at least one item.
    * weaknesses must contain at least one item.
    * section_explanations must contain at least one item.
    * Every section explanation must have a non-blank explanation string.
    * suggestions must contain at least one item.
    * summary must be a non-blank string.

    Args:
        explanation: The ``ATSExplanation`` returned by the LLM chain.

    Returns:
        The same ``explanation`` object, unmodified, if all checks pass.

    Raises:
        OutputValidationError: If any semantic check fails.
    """
    _check_non_empty_list(explanation.strengths, field="strengths")
    _check_non_empty_list(explanation.weaknesses, field="weaknesses")
    _check_non_empty_list(explanation.section_explanations, field="section_explanations")
    _check_section_explanations(explanation)
    _check_non_empty_list(explanation.suggestions, field="suggestions")
    _check_non_blank_string(explanation.summary, field="summary")
    _check_non_empty_list(explanation.recommendations, field="recommendations")
    _check_recommendations(explanation)

    return explanation


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


def parse_and_repair_response(raw_text: str) -> ATSExplanation:
    """Parse raw LLM response text into ATSExplanation, applying local repairs on failure."""
    logger.info("[OutputGuardrail] Executing response parsing and local repair checks.")

    # 1. JSON Pre-validation & Truncation Repair
    cleaned_text = clean_json_text(raw_text)
    parsed_dict = None
    try:
        parsed_dict = json.loads(cleaned_text)
    except json.JSONDecodeError as json_err:
        logger.warning(
            "[OutputGuardrail] JSON decode failed: %s. Attempting truncation repair.",
            json_err,
        )
        parsed_dict = repair_truncated_json(cleaned_text)
        if not parsed_dict:
            logger.warning("[OutputGuardrail] JSON truncation repair failed.")
            raise AIGenerationError(
                message=f"Invalid JSON format and truncation repair failed: {json_err}",
                metadata={"raw_text": raw_text},
            ) from json_err
        logger.info("[OutputGuardrail] JSON truncation repair succeeded.")

    # 2. Verify recommendations and perform local repair if needed
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
                    "[OutputGuardrail] Recommendation at index %d missing fields %s. "
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

    logger.debug("[OutputGuardrail] Parsed JSON output dict: %s", parsed_dict)

    # 3. Pydantic validation
    try:
        explanation = ATSExplanation(**parsed_dict)
    except (ValidationError, TypeError, ValueError) as val_err:
        logger.error("[OutputGuardrail] Pydantic schema validation failed: %s", val_err)
        raise AIGenerationError(
            message=f"Pydantic schema validation failed: {val_err}",
            metadata={"parsed_dict": parsed_dict},
        ) from val_err

    # 4. Output Guardrails check
    try:
        validate_output(explanation)
    except OutputValidationError as out_err:
        logger.error("[OutputGuardrail] Semantic output guardrail check failed: %s", out_err)
        raise AIGenerationError(
            message=f"Semantic validation failed: {out_err}",
            metadata={"parsed_dict": parsed_dict},
        ) from out_err

    logger.info("[OutputGuardrail] Response validation succeeded.")
    return explanation


# Private helpers


def _check_non_empty_list(items: list, field: str) -> None:
    """Raise ``OutputValidationError`` if ``items`` is empty."""
    if not items:
        logger.warning("[OutputGuardrail] '%s' is empty in LLM response.", field)
        raise OutputValidationError(
            message=f"LLM response field '{field}' must not be empty.",
            field=field,
        )


def _check_non_blank_string(value: str, field: str) -> None:
    """Raise ``OutputValidationError`` if ``value`` is blank."""
    if not value or not value.strip():
        logger.warning("[OutputGuardrail] '%s' is blank in LLM response.", field)
        raise OutputValidationError(
            message=f"LLM response field '{field}' must not be blank.",
            field=field,
        )


def _check_section_explanations(explanation: ATSExplanation) -> None:
    """Validate that every section explanation has a non-blank explanation."""
    for i, section in enumerate(explanation.section_explanations):
        if not section.explanation or not section.explanation.strip():
            logger.warning(
                "[OutputGuardrail] section_explanations[%d] has blank explanation (section=%r).",
                i,
                section.section,
            )
            raise OutputValidationError(
                message=(
                    f"section_explanations[{i}] (section={section.section!r}) "
                    "has a blank explanation."
                ),
                field="section_explanations",
            )


def _check_recommendations(explanation: ATSExplanation) -> None:
    """Validate that every recommendation has non-blank required fields."""
    for i, rec in enumerate(explanation.recommendations):
        if not rec.priority or not rec.priority.strip():
            raise OutputValidationError(message=f"recommendations[{i}] has blank priority.", field="recommendations")
        if not rec.issue or not rec.issue.strip():
            raise OutputValidationError(message=f"recommendations[{i}] has blank issue.", field="recommendations")
        if not rec.why or not rec.why.strip():
            raise OutputValidationError(message=f"recommendations[{i}] has blank why.", field="recommendations")
        if not rec.copy_paste_content or not rec.copy_paste_content.strip():
            raise OutputValidationError(message=f"recommendations[{i}] has blank copy_paste_content.", field="recommendations")
        if not rec.placement or not rec.placement.strip():
            raise OutputValidationError(message=f"recommendations[{i}] has blank placement.", field="recommendations")
        if not rec.ats_impact or not rec.ats_impact.strip():
            raise OutputValidationError(message=f"recommendations[{i}] has blank ats_impact.", field="recommendations")
