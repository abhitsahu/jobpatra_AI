"""Unit tests for Phase 10.3 — Output guardrails.

Tests verify:
- Valid structured explanation responses are accepted.
- Missing/blank fields (like summary or empty strengths/weaknesses) are rejected.
- Empty section explanations or blank section explanations are rejected.
- Empty suggestions list is rejected.
"""

from __future__ import annotations

import pytest

from app.ai.guardrails.output_guardrails import OutputValidationError, validate_output
from app.schemas.ai import ATSExplanation, SectionExplanation, RecommendationSchema

_VALID_RECOMMENDATIONS = [
    RecommendationSchema(
        priority="High",
        issue="Missing professional summary.",
        why="A summary hooks the recruiter.",
        copy_paste_content="Motivated engineer.",
        placement="Top of resume.",
        ats_impact="+10 points"
    )
]


class TestOutputGuardrails:
    def test_valid_explanation_passes(self) -> None:
        """A fully populated and semantically valid ATSExplanation should pass."""
        valid_explanation = ATSExplanation(
            strengths=["Strong programming skills in Python.", "Docker experience is solid."],
            weaknesses=["Missing Kubernetes expertise.", "No cloud certification."],
            section_explanations=[
                SectionExplanation(
                    section="Keywords",
                    score=75.0,
                    explanation="Matched 7 out of 10 keywords.",
                )
            ],
            suggestions=["Add Kubernetes to your resume.", "List your AWS certification."],
            summary="Overall good fit for the python developer role.",
            recommendations=_VALID_RECOMMENDATIONS,
        )
        result = validate_output(valid_explanation)
        assert result is valid_explanation

    def test_empty_strengths_rejected(self) -> None:
        """If strengths list is empty, it should raise OutputValidationError."""
        invalid = ATSExplanation(
            strengths=[],
            weaknesses=["Missing Kubernetes expertise."],
            section_explanations=[
                SectionExplanation(
                    section="Keywords",
                    score=75.0,
                    explanation="Matched 7 out of 10.",
                )
            ],
            suggestions=["Add Kubernetes to your resume."],
            summary="Summary text.",
            recommendations=_VALID_RECOMMENDATIONS,
        )
        with pytest.raises(OutputValidationError, match="field 'strengths' must not be empty"):
            validate_output(invalid)

    def test_empty_weaknesses_rejected(self) -> None:
        """If weaknesses list is empty, it should raise OutputValidationError."""
        invalid = ATSExplanation(
            strengths=["Python experience."],
            weaknesses=[],
            section_explanations=[
                SectionExplanation(
                    section="Keywords",
                    score=75.0,
                    explanation="Matched 7 out of 10.",
                )
            ],
            suggestions=["Add Kubernetes to your resume."],
            summary="Summary text.",
            recommendations=_VALID_RECOMMENDATIONS,
        )
        with pytest.raises(OutputValidationError, match="field 'weaknesses' must not be empty"):
            validate_output(invalid)

    def test_empty_section_explanations_rejected(self) -> None:
        """If section_explanations list is empty, it should raise OutputValidationError."""
        invalid = ATSExplanation(
            strengths=["Python experience."],
            weaknesses=["No Kubernetes."],
            section_explanations=[],
            suggestions=["Add Kubernetes to your resume."],
            summary="Summary text.",
            recommendations=_VALID_RECOMMENDATIONS,
        )
        with pytest.raises(OutputValidationError, match="field 'section_explanations' must not be empty"):
            validate_output(invalid)

    def test_blank_section_explanation_field_rejected(self) -> None:
        """If any SectionExplanation has a blank or empty explanation string, it should be rejected."""
        invalid = ATSExplanation(
            strengths=["Python experience."],
            weaknesses=["No Kubernetes."],
            section_explanations=[
                SectionExplanation(
                    section="Keywords",
                    score=75.0,
                    explanation="   ",
                )
            ],
            suggestions=["Add Kubernetes to your resume."],
            summary="Summary text.",
            recommendations=_VALID_RECOMMENDATIONS,
        )
        with pytest.raises(OutputValidationError, match="has a blank explanation"):
            validate_output(invalid)

    def test_empty_suggestions_rejected(self) -> None:
        """If suggestions list is empty, it should raise OutputValidationError."""
        invalid = ATSExplanation(
            strengths=["Python experience."],
            weaknesses=["No Kubernetes."],
            section_explanations=[
                SectionExplanation(
                    section="Keywords",
                    score=75.0,
                    explanation="Matched 7 out of 10.",
                )
            ],
            suggestions=[],
            summary="Summary text.",
            recommendations=_VALID_RECOMMENDATIONS,
        )
        with pytest.raises(OutputValidationError, match="field 'suggestions' must not be empty"):
            validate_output(invalid)

    def test_blank_summary_rejected(self) -> None:
        """If the summary string is empty or blank, it should raise OutputValidationError."""
        invalid = ATSExplanation(
            strengths=["Python experience."],
            weaknesses=["No Kubernetes."],
            section_explanations=[
                SectionExplanation(
                    section="Keywords",
                    score=75.0,
                    explanation="Matched 7 out of 10.",
                )
            ],
            suggestions=["Add Kubernetes to your resume."],
            summary="   ",
            recommendations=_VALID_RECOMMENDATIONS,
        )
        with pytest.raises(OutputValidationError, match="field 'summary' must not be blank"):
            validate_output(invalid)

    def test_empty_recommendations_rejected(self) -> None:
        """If recommendations list is empty, it should raise OutputValidationError."""
        invalid = ATSExplanation(
            strengths=["Python experience."],
            weaknesses=["No Kubernetes."],
            section_explanations=[
                SectionExplanation(
                    section="Keywords",
                    score=75.0,
                    explanation="Matched 7 out of 10.",
                )
            ],
            suggestions=["Add Kubernetes to your resume."],
            summary="Summary text.",
            recommendations=[],
        )
        with pytest.raises(OutputValidationError, match="field 'recommendations' must not be empty"):
            validate_output(invalid)

    def test_invalid_recommendation_fields_rejected(self) -> None:
        """If any field in recommendations is blank, it should raise OutputValidationError."""
        invalid_rec = [
            RecommendationSchema(
                priority="High",
                issue="Missing professional summary.",
                why="   ",  # Blank
                copy_paste_content="Highly motivated Software Engineer...",
                placement="At the very top.",
                ats_impact="+10 points"
            )
        ]
        invalid = ATSExplanation(
            strengths=["Python experience."],
            weaknesses=["No Kubernetes."],
            section_explanations=[
                SectionExplanation(
                    section="Keywords",
                    score=75.0,
                    explanation="Matched 7 out of 10.",
                )
            ],
            suggestions=["Add Kubernetes to your resume."],
            summary="Summary text.",
            recommendations=invalid_rec,
        )
        with pytest.raises(OutputValidationError, match="has blank why"):
            validate_output(invalid)
