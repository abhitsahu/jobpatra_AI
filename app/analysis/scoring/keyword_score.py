"""Keyword score — measure JD keyword coverage in the resume.

Single responsibility: given a ``MatchResult``, compute a score [0–100]
based on the fraction of JD keywords that were found in the resume.

Formula
-------
    score = (matched_count / total_jd_keywords) × 100

where:
    matched_count     = len(match_result.matched)
    total_jd_keywords = matched_count + len(match_result.missing)

Edge cases
----------
- If the JD has no keywords → score is 0.0 (no signal).
- Score is clamped to [0, 100].

All functions are pure. No I/O. No AI. No FastAPI.
"""

from app.analysis.matching.keyword_matcher import MatchResult


def calculate(match_result: MatchResult) -> float:
    """Compute the keyword coverage score.

    Args:
        match_result: Output of ``keyword_matcher.match()``.

    Returns:
        Score in [0.0, 100.0].  Returns 0.0 when the JD had no keywords.
    """
    matched_count = len(match_result.matched)
    missing_count = len(match_result.missing)
    total_jd = matched_count + missing_count

    if total_jd == 0:
        return 0.0

    raw = (matched_count / total_jd) * 100.0
    return _clamp(raw)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp ``value`` to [lo, hi]."""
    return max(lo, min(hi, value))
