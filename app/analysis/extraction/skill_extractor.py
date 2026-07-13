"""Skill extractor — match text tokens against the reference skills dataset.

This module has ONE responsibility: given a piece of text, identify which
known skills (from ``reference_data/skills_list.py``) appear in that text,
and return them grouped by category.

It does NOT:
  - hardcode skill names (all data lives in skills_list.py)
  - compare skills between resume and JD (that is the Matcher's job)
  - score or rank skills
  - call AI

Algorithm:
  1. Build a reverse lookup: alias → (category, canonical) at import time.
  2. Tokenise the input text into candidate tokens (single words + bigrams).
  3. For each token check the lookup table.
  4. Return matched skills grouped by category, deduplicated.

All functions are pure after module initialisation. No I/O. No FastAPI.
"""

import re
from dataclasses import dataclass, field

from app.analysis.extraction.reference_data.skills_list import SKILLS

# ---------------------------------------------------------------------------
# Build reverse lookup at import time  alias → (category, canonical)
# ---------------------------------------------------------------------------

_ALIAS_MAP: dict[str, tuple[str, str]] = {}

for _category, _skills in SKILLS.items():
    for _canonical, _aliases in _skills.items():
        for _alias in _aliases:
            _ALIAS_MAP[_alias] = (_category, _canonical)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class SkillMatch:
    """A single identified skill with its category and canonical display name."""

    canonical: str
    """Display name as defined in skills_list.py, e.g. ``'Node.js'``."""
    category: str
    """Category label, e.g. ``'Backend'``."""


@dataclass
class SkillExtractionResult:
    """Result of skill extraction on one piece of text."""

    skills: list[SkillMatch] = field(default_factory=list)
    """All matched skills, deduplicated, in first-seen order."""
    by_category: dict[str, list[str]] = field(default_factory=dict)
    """Skills grouped by category: {category: [canonical, ...]}."""


# Tokenisation — same liberal pattern as keyword extractor
_TOKEN_RE: re.Pattern[str] = re.compile(r"[A-Za-z][A-Za-z0-9.#+\-_]*")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract(text: str) -> SkillExtractionResult:
    """Identify known skills in ``text`` using the reference skills dataset.

    Generates single-word and two-word candidate phrases from the text and
    looks each one up in the alias map built from skills_list.py.

    Args:
        text: Any plain text — resume skills section, full resume, JD, etc.

    Returns:
        ``SkillExtractionResult`` with all matched skills deduplicated and
        grouped by category.  Unknown tokens are silently ignored.
    """
    candidates = _generate_candidates(text)
    seen_canonical: set[str] = set()
    matches: list[SkillMatch] = []

    for candidate in candidates:
        entry = _ALIAS_MAP.get(candidate.lower())
        if entry is None:
            continue
        category, canonical = entry
        if canonical in seen_canonical:
            continue
        seen_canonical.add(canonical)
        matches.append(SkillMatch(canonical=canonical, category=category))

    return _build_result(matches)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _generate_candidates(text: str) -> list[str]:
    """Return single tokens and adjacent bigrams from ``text``.

    Bigrams allow matching multi-word skills like ``"machine learning"``
    or ``"spring boot"`` without needing NLP dependencies.

    Args:
        text: Raw input text.

    Returns:
        List of candidate strings (may contain duplicates across single/bi).
    """
    tokens = _TOKEN_RE.findall(text)
    candidates: list[str] = list(tokens)
    # Bigrams
    for i in range(len(tokens) - 1):
        candidates.append(f"{tokens[i]} {tokens[i + 1]}")
    return candidates


def _build_result(matches: list[SkillMatch]) -> SkillExtractionResult:
    """Group a flat list of SkillMatch objects into a SkillExtractionResult.

    Args:
        matches: Deduplicated list of matched skills.

    Returns:
        ``SkillExtractionResult`` with both the flat list and category map.
    """
    by_category: dict[str, list[str]] = {}
    for m in matches:
        by_category.setdefault(m.category, []).append(m.canonical)
    return SkillExtractionResult(skills=matches, by_category=by_category)
