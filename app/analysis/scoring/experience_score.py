"""Experience score — evaluate work experience quality deterministically.

Single responsibility: given a list of ``ExperienceEntry`` objects (from
``experience_extractor``), compute a score [0–100] using four sub-signals:

1. Duration score  (max 40 pts) — total years of experience.
2. Continuity score (max 20 pts) — number of distinct jobs (breadth signal).
3. Bullet density  (max 20 pts) — description quality: average bullets/entry.
4. Metrics score   (max 20 pts) — quantifiable achievements (%, $, x).

All thresholds are deterministic rule-based constants, not AI judgements.

All functions are pure. No I/O. No AI. No FastAPI.
"""

from app.analysis.extraction.experience_extractor import ExperienceEntry


# ---------------------------------------------------------------------------
# Rule-based thresholds (tune here, never inside scoring_engine)
# ---------------------------------------------------------------------------

# Duration: full score at or above this many total years
_FULL_DURATION_YEARS: float = 10.0

# Continuity: full score at or above this many jobs
_FULL_JOB_COUNT: int = 4

# Bullet density: full score when avg bullets per entry >= this
_FULL_BULLET_DENSITY: float = 4.0

# Metrics: full score when total quantified achievements >= this
_FULL_METRIC_COUNT: int = 6


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate(entries: list[ExperienceEntry]) -> float:
    """Compute the experience score.

    Args:
        entries: Work experience entries from ``experience_extractor.extract()``.

    Returns:
        Score in [0.0, 100.0].  Returns 0.0 for an empty entry list.
    """
    if not entries:
        return 0.0

    duration_pts = _duration_score(entries)
    continuity_pts = _continuity_score(entries)
    bullet_pts = _bullet_density_score(entries)
    metrics_pts = _metrics_score(entries)

    total = duration_pts + continuity_pts + bullet_pts + metrics_pts
    return _clamp(total)


# ---------------------------------------------------------------------------
# Sub-signal calculators
# ---------------------------------------------------------------------------


def _duration_score(entries: list[ExperienceEntry]) -> float:
    """Award up to 40 points for total years of experience.

    Args:
        entries: All experience entries.

    Returns:
        Points in [0.0, 40.0].
    """
    total_years = sum(e.duration_years for e in entries if e.duration_years is not None)
    ratio = min(total_years / _FULL_DURATION_YEARS, 1.0)
    return round(ratio * 40.0, 4)


def _continuity_score(entries: list[ExperienceEntry]) -> float:
    """Award up to 20 points for number of distinct jobs.

    More jobs = broader experience signal (up to a cap).

    Args:
        entries: All experience entries.

    Returns:
        Points in [0.0, 20.0].
    """
    ratio = min(len(entries) / _FULL_JOB_COUNT, 1.0)
    return round(ratio * 20.0, 4)


def _bullet_density_score(entries: list[ExperienceEntry]) -> float:
    """Award up to 20 points for description quality (bullets per entry).

    Args:
        entries: All experience entries.

    Returns:
        Points in [0.0, 20.0].
    """
    if not entries:
        return 0.0
    avg_bullets = sum(len(e.bullets) for e in entries) / len(entries)
    ratio = min(avg_bullets / _FULL_BULLET_DENSITY, 1.0)
    return round(ratio * 20.0, 4)


def _metrics_score(entries: list[ExperienceEntry]) -> float:
    """Award up to 20 points for quantified achievements (%, $, x, +).

    Args:
        entries: All experience entries.

    Returns:
        Points in [0.0, 20.0].
    """
    total_metrics = sum(len(e.metrics) for e in entries)
    ratio = min(total_metrics / _FULL_METRIC_COUNT, 1.0)
    return round(ratio * 20.0, 4)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp ``value`` to [lo, hi]."""
    return max(lo, min(hi, value))
