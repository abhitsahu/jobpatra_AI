"""Summary score — evaluate the quality of the resume summary section.

Single responsibility: given the raw text of the summary section, award
points for presence, length, and basic content signals.

Scoring rules (deterministic)
------------------------------
Check                              Points
-----                              ------
Summary present and non-empty         40
Word count ≥ 20                       20
Word count ≥ 50                       20   (additional, cumulative)
Contains at least one action word     10
Contains a numeric metric (%, $, x)  10

Total possible: 100

No AI. No writing-quality judgement. Pure structural checks.

All functions are pure. No I/O. No AI. No FastAPI.
"""

import re


# ---------------------------------------------------------------------------
# Rule constants
# ---------------------------------------------------------------------------

_MIN_WORDS_BASIC: int = 20
_MIN_WORDS_DETAILED: int = 50

_ACTION_WORDS: frozenset[str] = frozenset(
    {
        "led", "built", "created", "designed", "developed", "implemented",
        "architected", "managed", "delivered", "launched", "scaled",
        "reduced", "increased", "improved", "optimized", "automated",
        "established", "migrated", "deployed", "spearheaded",
    }
)

_METRIC_RE: re.Pattern[str] = re.compile(
    r"\b\d+(?:\.\d+)?%"         # 40%
    r"|\b\d+(?:\.\d+)?[xX]\b"   # 5x
    r"|\$\s*\d[\d,]*"            # $200K
    r"|\b\d[\d,]+\+",            # 100+
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate(summary_text: str | None) -> float:
    """Compute the summary quality score.

    Args:
        summary_text: The raw text of the resume summary / objective section,
            or ``None`` if the section is absent.

    Returns:
        Score in [0.0, 100.0].  Returns 0.0 when ``summary_text`` is absent
        or empty.
    """
    if not summary_text or not summary_text.strip():
        return 0.0

    text = summary_text.strip()
    score = 0

    # Presence (40 pts)
    score += 40

    # Length (up to 40 additional pts)
    words = text.split()
    if len(words) >= _MIN_WORDS_BASIC:
        score += 20
    if len(words) >= _MIN_WORDS_DETAILED:
        score += 20

    # Action word (10 pts)
    lower_words = {w.strip(".,;:!?()[]\"'").lower() for w in words}
    if lower_words & _ACTION_WORDS:
        score += 10

    # Numeric metric (10 pts)
    if _METRIC_RE.search(text):
        score += 10

    return _clamp(float(score))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp ``value`` to [lo, hi]."""
    return max(lo, min(hi, value))
