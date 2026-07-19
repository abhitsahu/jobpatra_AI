"""Output guardrails.

Validates LLM output AFTER the chain returns a result.

* Validates the structured ``ATSExplanation`` Pydantic model for logical
  completeness beyond what Pydantic's type system alone enforces.
* Does NOT retry on failure — retry responsibility lives in ``retry_policy.py``.
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

from app.core.logging import logger
from app.schemas.ai import ATSExplanation


class OutputValidationError(ValueError):
    """Raised when LLM output fails semantic validation.

    This is NOT an ``AppError``.  It is internal to the guardrail / retry
    pipeline.  ``retry_policy.py`` catches it and triggers a single retry.
    After retry, ``AIGenerationError`` is raised if it still fails.
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
