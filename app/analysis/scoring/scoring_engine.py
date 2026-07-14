"""Scoring engine — the single public entry point for ATS scoring.

This module is the ONLY file that external code should import from the
``scoring`` package.  It orchestrates all sub-scorers and applies the
weights defined in ``weights_config``.

Responsibilities
----------------
- Accept the structured outputs of previous pipeline phases.
- Delegate to each sub-scorer for its specific domain.
- Apply weighted average using ``weights_config.DEFAULT_WEIGHTS``.
- Return a single ``ATSReport`` dataclass.

This module does NOT:
  - implement any scoring logic itself
  - call AI, LangChain, or external services
  - import from FastAPI
  - access a database or cache

Weighted formula
----------------
    overall_score = (
        keyword_score    × weights.keyword_score
        + experience_score × weights.experience_score
        + skills_score     × weights.skills_score
        + formatting_score × weights.formatting_score
        + education_score  × weights.education_score
        + summary_score    × weights.summary_score
    )

All inputs and the overall score live in the range [0.0, 100.0].
"""

from dataclasses import dataclass

from app.analysis.extraction.education_extractor import EducationExtractionResult
from app.analysis.extraction.experience_extractor import ExperienceEntry
from app.analysis.matching.keyword_matcher import MatchResult
from app.analysis.normalization.section_splitter import ResumeSection
from app.analysis.scoring import (
    education_score,
    experience_score,
    formatting_score,
    keyword_score,
    skills_score,
    summary_score,
)
from app.analysis.scoring.weights_config import DEFAULT_WEIGHTS, ScoringWeights


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ATSReport:
    """Complete ATS scoring report for one resume + JD pair."""

    keyword_score: float
    """Keyword coverage score [0–100]."""
    experience_score: float
    """Work experience quality score [0–100]."""
    skills_score: float
    """Skill coverage score [0–100]."""
    formatting_score: float
    """Resume structure completeness score [0–100]."""
    education_score: float
    """Education credential score [0–100]."""
    summary_score: float
    """Summary quality score [0–100]."""
    overall_score: float
    """Weighted average of all sub-scores, rounded to 2 decimal places."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score(
    match_result: MatchResult,
    experience_entries: list[ExperienceEntry],
    resume_skills: list[str],
    required_skills: list[str],
    education_result: EducationExtractionResult,
    sections: ResumeSection,
    weights: ScoringWeights = DEFAULT_WEIGHTS,
) -> ATSReport:
    """Compute the full ATS report for a resume + JD pair.

    Args:
        match_result: Output of ``keyword_matcher.match()``.
        experience_entries: Output of ``experience_extractor.extract()``.
        resume_skills: Canonical skill names from ``skill_extractor``.
        required_skills: Skill names required by the JD.
        education_result: Output of ``education_extractor.extract()``.
        sections: Output of ``section_splitter.split()``.
        weights: Scoring weights. Defaults to ``DEFAULT_WEIGHTS``.

    Returns:
        ``ATSReport`` with all sub-scores and the weighted ``overall_score``.
    """
    kw   = keyword_score.calculate(match_result)
    exp  = experience_score.calculate(experience_entries)
    sk   = skills_score.calculate(resume_skills, required_skills)
    fmt  = formatting_score.calculate(sections)
    edu  = education_score.calculate(education_result)
    summ = summary_score.calculate(sections.summary)

    overall = _weighted_average(kw, exp, sk, fmt, edu, summ, weights)

    return ATSReport(
        keyword_score=round(kw, 2),
        experience_score=round(exp, 2),
        skills_score=round(sk, 2),
        formatting_score=round(fmt, 2),
        education_score=round(edu, 2),
        summary_score=round(summ, 2),
        overall_score=round(overall, 2),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _weighted_average(
    kw: float,
    exp: float,
    sk: float,
    fmt: float,
    edu: float,
    summ: float,
    weights: ScoringWeights,
) -> float:
    """Compute the weighted average of all sub-scores.

    Args:
        kw, exp, sk, fmt, edu, summ: Individual sub-scores [0–100].
        weights: Weight configuration.

    Returns:
        Weighted average in [0.0, 100.0].
    """
    return (
        kw   * weights.keyword_score
        + exp  * weights.experience_score
        + sk   * weights.skills_score
        + fmt  * weights.formatting_score
        + edu  * weights.education_score
        + summ * weights.summary_score
    )
