"""Skills score — measure skill coverage against JD requirements.

Single responsibility: given resume skills and required JD skills, compute
a coverage score [0–100] using the multi-pass KeywordMatcher (Exact, Synonym,
Fuzzy, and optional Semantic).

Formula
-------
    score = (sum(matched skill weights) / sum(required skill weights)) × 100

All functions are pure. No I/O. No AI. No FastAPI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from app.analysis.matching import keyword_matcher
from app.analysis.matching.keyword_matcher import MatchResult
from app.services.taxonomy_service import get_taxonomy_service

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
    """Match weighted, score-bearing requirements once and return evidence."""
    taxonomy = get_taxonomy_service()
    scoreable_required = list(
        dict.fromkeys(
            taxonomy.normalize(skill)
            for skill in required_skills
            if taxonomy.get_weight(skill) >= 0.3
        )
    )
    if not scoreable_required:
        return SkillScoreResult(0.0, MatchResult(), 0)

    if not resume_skills:
        return SkillScoreResult(
            0.0,
            MatchResult(missing=list(scoreable_required)),
            len(scoreable_required),
        )

    match_result = keyword_matcher.match(
        resume_keywords=resume_skills,
        jd_keywords=scoreable_required,
        embedding_provider=embedding_provider,
        semantic_threshold=semantic_threshold,
    )

    denominator = sum(taxonomy.get_weight(skill) for skill in scoreable_required)
    matched_requirements = {
        match.matched_jd_keyword
        for match in match_result.matched
        if match.matched_jd_keyword is not None
    }
    numerator = sum(
        taxonomy.get_weight(skill)
        for skill in scoreable_required
        if skill in matched_requirements
    )
    score = (numerator / denominator) * 100.0 if denominator else 0.0
    return SkillScoreResult(
        score=_clamp(score),
        match_result=match_result,
        required_skill_count=len(scoreable_required),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp ``value`` to [lo, hi]."""
    return max(lo, min(hi, value))
