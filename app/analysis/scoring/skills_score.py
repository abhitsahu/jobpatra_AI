"""Skills score — measure skill coverage against JD requirements.

Single responsibility: given resume skills and required JD skills, compute
a coverage score [0–100].

Formula
-------
    score = (matched_skills / required_skills) × 100

If required_skills is 0, returns 0.0 (no signal).

All functions are pure. No I/O. No AI. No FastAPI.
"""


def calculate(resume_skills: list[str], required_skills: list[str]) -> float:
    """Compute skill coverage score.

    Comparison is case-insensitive.  A resume skill counts as matched if its
    lowercased value appears in the lowercased set of required skills.

    Args:
        resume_skills: Canonical skill names from the resume
            (from ``skill_extractor``).
        required_skills: Skills that the JD requires.

    Returns:
        Score in [0.0, 100.0].  Returns 0.0 when ``required_skills`` is empty.

    Example:
        >>> calculate(["Python", "Docker", "React", "AWS"], ["Python", "Docker", "React", "Redis", "AWS"])
        80.0
    """
    if not required_skills:
        return 0.0

    required_lower = {s.lower() for s in required_skills}
    resume_lower = {s.lower() for s in resume_skills}

    matched = len(required_lower & resume_lower)
    score = (matched / len(required_lower)) * 100.0
    return _clamp(score)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp ``value`` to [lo, hi]."""
    return max(lo, min(hi, value))
