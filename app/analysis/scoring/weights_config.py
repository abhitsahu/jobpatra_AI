"""Scoring weights configuration.

This file is the SINGLE source of truth for all ATS scoring weights.

Rules
-----
- Never hardcode weights inside individual scoring modules.
- Only ``scoring_engine.py`` reads and applies these weights.
- All weights are floats in the range [0.0, 1.0].
- Weights must sum to exactly 1.0 — a runtime assertion enforces this.
- To tune scoring, change values here ONLY.

Default weights
---------------
Category           Weight
---------          ------
Keyword Score       0.40   (most signal from keyword presence)
Experience Score    0.25
Skills Score        0.15
Formatting Score    0.10
Education Score     0.05
Summary Score       0.05
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringWeights:
    """Immutable container for all ATS scoring weights.

    All fields are floats. ``scoring_engine`` verifies they sum to 1.0 on
    startup so misconfiguration is caught immediately.
    """

    keyword_score: float
    experience_score: float
    skills_score: float
    formatting_score: float
    education_score: float
    summary_score: float

    def total(self) -> float:
        """Return the sum of all weights."""
        return (
            self.keyword_score
            + self.experience_score
            + self.skills_score
            + self.formatting_score
            + self.education_score
            + self.summary_score
        )


# ---------------------------------------------------------------------------
# Active weights — edit here to change scoring behaviour
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = ScoringWeights(
    keyword_score=0.30,
    experience_score=0.25,
    skills_score=0.25,
    formatting_score=0.10,
    education_score=0.05,
    summary_score=0.05,
)

assert abs(DEFAULT_WEIGHTS.total() - 1.0) < 1e-9, (
    f"ScoringWeights must sum to 1.0, got {DEFAULT_WEIGHTS.total()}"
)
