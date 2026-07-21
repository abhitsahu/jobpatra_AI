"""Unit tests for output parsing, local JSON repair, and validation.

These tests prove:
- Happy path: Valid JSON is parsed successfully.
- JSON Truncation Repair: Truncated JSON in recommendations array is repaired and salvaged.
- Local Recommendation Repair: Missing fields in recommendations are repaired locally.
- Invalid JSON: Non-repairable JSON raises AIGenerationError.
- Schema/Semantic failure: Invalid/incomplete schema raises AIGenerationError.
All operations happen locally; NO LLM call is triggered on failure/repair.
"""

from __future__ import annotations

import json
import pytest

from app.ai.guardrails.output_guardrails import parse_and_repair_response
from app.core.errors import AIGenerationError
from app.schemas.ai import ATSExplanation


class TestOutputRepair:
    @pytest.fixture()
    def valid_json_dict(self) -> dict:
        return {
            "strengths": ["Strong programming skills in Python."],
            "weaknesses": ["Missing Kubernetes expertise."],
            "section_explanations": [
                {
                    "section": "Keywords",
                    "score": 75.0,
                    "explanation": "Matched 7 out of 10 keywords.",
                }
            ],
            "suggestions": ["Add Kubernetes to skills."],
            "summary": "Overall good fit.",
            "recommendations": [
                {
                    "priority": "High",
                    "issue": "Missing Kubernetes.",
                    "why": "Crucial for this role.",
                    "copy_paste_content": "Kubernetes core concepts.",
                    "placement": "Skills section.",
                    "ats_impact": "+10 points",
                }
            ],
        }

    def test_happy_path_parsing(self, valid_json_dict: dict) -> None:
        """Verify that a valid JSON string parses successfully."""
        raw_text = json.dumps(valid_json_dict)
        result = parse_and_repair_response(raw_text)

        assert isinstance(result, ATSExplanation)
        assert result.summary == "Overall good fit."
        assert len(result.recommendations) == 1

    def test_truncated_json_repair_salvages_complete_items(self, valid_json_dict: dict) -> None:
        """Verify that truncated JSON gets repaired and parsed successfully."""
        valid_json_dict["recommendations"] = [
            {
                "priority": "High",
                "issue": "Missing Kubernetes.",
                "why": "Crucial for this role.",
                "copy_paste_content": "Kubernetes core concepts.",
                "placement": "Skills section.",
                "ats_impact": "+10 points",
            }
        ]
        raw_json = json.dumps(valid_json_dict)
        # Cut it off in the middle of a second recommendation
        truncated_json = (
            raw_json[:-2]
            + ', {"priority": "Medium", "issue": "Missing Docker", "why": "Need containers'
        )

        result = parse_and_repair_response(truncated_json)

        assert isinstance(result, ATSExplanation)
        # Should have salvaged the first recommendation and ignored the incomplete second one
        assert len(result.recommendations) == 1
        assert result.recommendations[0].issue == "Missing Kubernetes."

    def test_incomplete_recommendation_local_repair(self, valid_json_dict: dict) -> None:
        """If a recommendation is missing required fields, it is repaired locally."""
        valid_json_dict["recommendations"] = [
            {
                "priority": "High",
                "issue": "Missing Kubernetes.",
                "why": "Crucial for this role.",
                "copy_paste_content": "Kubernetes core.",
                "placement": "",
                "ats_impact": "",
            }
        ]

        raw_text = json.dumps(valid_json_dict)
        result = parse_and_repair_response(raw_text)

        assert isinstance(result, ATSExplanation)
        assert len(result.recommendations) == 1
        # Should be repaired locally to fallback values
        assert result.recommendations[0].placement == "In the relevant section."
        assert result.recommendations[0].ats_impact == "+5 points"

    def test_unrepairable_json_raises_error(self) -> None:
        """If JSON is completely invalid and repair fails, raise AIGenerationError."""
        raw_text = "{ invalid json..."
        with pytest.raises(AIGenerationError, match="Invalid JSON format and truncation repair failed"):
            parse_and_repair_response(raw_text)

    def test_semantic_validation_failure_raises_error(self, valid_json_dict: dict) -> None:
        """If Pydantic parsing succeeds but semantic guardrails fail, raise AIGenerationError."""
        # Empty strengths list is a semantic violation
        valid_json_dict["strengths"] = []
        raw_text = json.dumps(valid_json_dict)

        with pytest.raises(AIGenerationError, match="Semantic validation failed"):
            parse_and_repair_response(raw_text)
