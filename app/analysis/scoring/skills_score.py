"""Skills score — measure skill coverage against JD requirements.

Single responsibility: given resume skills and required JD skills, compute
a coverage score [0–100] using the multi-pass KeywordMatcher (Exact, Synonym,
Fuzzy, and optional Semantic).

Formula
-------
    score = (matched_skills_count / required_skills_count) × 100

All functions are pure. No I/O. No AI. No FastAPI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from app.analysis.matching import keyword_matcher
from app.analysis.matching.keyword_matcher import MatchResult

if TYPE_CHECKING:
    from app.analysis.matching.semantic_matcher import EmbeddingProvider


@dataclass(frozen=True)
class SkillScoreResult:
    """The score and exact match data used to calculate it."""

    score: float
    match_result: MatchResult
    required_skill_count: int


def calculate(
    resume_skills: list[str],
    required_skills: list[str],
    embedding_provider: EmbeddingProvider | None = None,
    semantic_threshold: float | None = 0.60,
) -> float:
    """Compute skill coverage score using KeywordMatcher.

    Args:
        resume_skills: Canonical skill names from the resume.
        required_skills: Skills required by the JD.
        embedding_provider: Optional provider for semantic matching pass.
        semantic_threshold: Cosine similarity threshold for semantic matching (default 0.60).

    Returns:
        Score in [0.0, 100.0]. Returns 0.0 when ``required_skills`` is empty.
    """
    return evaluate(
        resume_skills=resume_skills,
        required_skills=required_skills,
        embedding_provider=embedding_provider,
        semantic_threshold=semantic_threshold,
    ).score


def evaluate(
    resume_skills: list[str],
    required_skills: list[str],
    embedding_provider: EmbeddingProvider | None = None,
    semantic_threshold: float | None = 0.60,
) -> SkillScoreResult:
    """Match required technical skills once and return its score and evidence."""
    if not required_skills:
        return SkillScoreResult(0.0, MatchResult(), 0)

    if not resume_skills:
        return SkillScoreResult(
            0.0,
            MatchResult(missing=list(required_skills)),
            len(required_skills),
        )

    match_result = keyword_matcher.match(
        resume_keywords=resume_skills,
        jd_keywords=required_skills,
        embedding_provider=embedding_provider,
        semantic_threshold=semantic_threshold,
    )

    matched_count = len(match_result.matched)
    score = (matched_count / len(required_skills)) * 100.0
    return SkillScoreResult(
        score=_clamp(score),
        match_result=match_result,
        required_skill_count=len(required_skills),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp ``value`` to [lo, hi]."""
    return max(lo, min(hi, value))
