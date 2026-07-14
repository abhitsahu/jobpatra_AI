"""Formatting score — evaluate resume structure completeness.

Single responsibility: given a ``ResumeSection``, award points for the
presence and non-emptiness of expected resume sections.

Scoring rules (deterministic)
------------------------------
Each section is worth a fixed number of points.  If the section is present
and non-empty it earns its full allocation.

Section             Points
---------           ------
Summary / Objective   15
Experience            30
Education             20
Skills                20
Projects               5
Certifications         5
Languages              5

Total possible: 100

Missing or empty sections score 0 for that slot.

All functions are pure. No I/O. No AI. No FastAPI.
"""

from app.analysis.normalization.section_splitter import ResumeSection


# ---------------------------------------------------------------------------
# Section → point allocation
# ---------------------------------------------------------------------------

_SECTION_POINTS: list[tuple[str, int]] = [
    ("summary",          15),
    ("experience",       30),
    ("education",        20),
    ("skills",           20),
    ("projects",          5),
    ("certifications",    5),
    ("languages",         5),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate(sections: ResumeSection) -> float:
    """Compute the formatting / structure score.

    Args:
        sections: Output of ``section_splitter.split()``.

    Returns:
        Score in [0.0, 100.0].
    """
    total = 0
    for attr, points in _SECTION_POINTS:
        value: str | None = getattr(sections, attr, None)
        if value and value.strip():
            total += points
    return float(_clamp(total))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _clamp(value: int | float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp ``value`` to [lo, hi]."""
    return max(lo, min(hi, float(value)))
