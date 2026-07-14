"""Education score — evaluate education credentials deterministically.

Single responsibility: given an ``EducationExtractionResult``, compute a
score [0–100] that reflects degree level and presence of certifications.

Scoring rules (deterministic)
------------------------------
Degree presence and level
  PhD / Doctorate               → 100 pts (base)
  Master / M.Sc / M.Tech / MBA  →  90 pts
  Bachelor / B.Sc / B.Tech / BE →  80 pts
  Associate / Diploma            →  60 pts
  Any degree detected            →  50 pts (fallback, unknown level)
  No degree detected             →   0 pts

Bonus
  +10 pts for each certification (capped at 10 pts total bonus)

Final score is clamped to [0, 100].

All functions are pure. No I/O. No AI. No FastAPI.
"""

from app.analysis.extraction.education_extractor import EducationExtractionResult


# ---------------------------------------------------------------------------
# Degree level → base points
# ---------------------------------------------------------------------------

_PHD_KEYWORDS: frozenset[str] = frozenset(
    {"phd", "ph.d", "doctorate", "doctoral", "dphil", "d.phil"}
)
_MASTERS_KEYWORDS: frozenset[str] = frozenset(
    {"master", "m.sc", "msc", "m.tech", "mtech", "mba", "m.eng", "meng",
     "m.s.", "ms", "m.a.", "ma", "postgraduate", "pg", "pgdip"}
)
_BACHELORS_KEYWORDS: frozenset[str] = frozenset(
    {"bachelor", "b.sc", "bsc", "b.tech", "btech", "be", "b.e.", "b.eng",
     "b.a.", "ba", "b.s.", "bs", "undergraduate", "ug"}
)
_ASSOCIATE_KEYWORDS: frozenset[str] = frozenset(
    {"associate", "diploma", "hnd", "hnc", "foundation"}
)

_DEGREE_BASE_SCORE: dict[str, int] = {
    "phd": 100,
    "masters": 90,
    "bachelors": 80,
    "associate": 60,
    "other": 50,
}

# Certification bonus per cert (capped at 10 total)
_CERT_BONUS_PER: int = 5
_CERT_BONUS_CAP: int = 10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate(education_result: EducationExtractionResult) -> float:
    """Compute the education score.

    Args:
        education_result: Output of ``education_extractor.extract()``.

    Returns:
        Score in [0.0, 100.0].
    """
    base = _degree_base_score(education_result)
    cert_bonus = min(
        len(education_result.certifications) * _CERT_BONUS_PER,
        _CERT_BONUS_CAP,
    )
    return _clamp(base + cert_bonus)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _degree_base_score(result: EducationExtractionResult) -> int:
    """Return the base score for the highest detected degree.

    Args:
        result: Education extraction result.

    Returns:
        Integer base score.
    """
    if not result.entries:
        return 0

    best = 0
    for entry in result.entries:
        level = _classify_degree(entry.degree)
        pts = _DEGREE_BASE_SCORE.get(level, 0)
        if pts > best:
            best = pts
    return best


def _classify_degree(degree: str | None) -> str:
    """Classify a degree string into a level label.

    Args:
        degree: Raw degree string from the extractor, or ``None``.

    Returns:
        One of ``'phd'``, ``'masters'``, ``'bachelors'``, ``'associate'``,
        ``'other'``, or ``'unknown'``.
    """
    if not degree:
        return "unknown"

    lower = degree.lower()

    if any(kw in lower for kw in _PHD_KEYWORDS):
        return "phd"
    if any(kw in lower for kw in _MASTERS_KEYWORDS):
        return "masters"
    if any(kw in lower for kw in _BACHELORS_KEYWORDS):
        return "bachelors"
    if any(kw in lower for kw in _ASSOCIATE_KEYWORDS):
        return "associate"
    return "other"


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp ``value`` to [lo, hi]."""
    return max(lo, min(hi, value))
