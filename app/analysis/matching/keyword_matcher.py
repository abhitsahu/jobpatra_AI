"""Keyword matcher — the single public entry point for the matching pipeline.

This module is the ONLY file that external modules should import from the
``matching`` package.  It orchestrates the three-step deterministic pipeline:

  Step 1 — Exact matching    (app.analysis.matching.exact_matcher)
  Step 2 — Synonym matching  (app.analysis.matching.synonym_map)
  Step 3 — Fuzzy matching    (app.analysis.matching.fuzzy_matcher)

The order is always Exact → Synonym → Fuzzy and can never be changed.
A keyword resolved at an earlier step is removed from further consideration.

Public API
----------
``match(resume_keywords, jd_keywords) -> MatchResult``

Result schema
-------------
``MatchResult`` contains:
  - ``matched``:    list of ``MatchedKeyword`` (keyword + matchType)
  - ``missing``:    JD keywords not found in the resume at all
  - ``unresolved``: resume keywords not found in the JD at all

matchType values: ``"EXACT"`` | ``"SYNONYM"`` | ``"FUZZY"``

This module does NOT:
  - implement semantic matching
  - calculate ATS scores
  - call AI or external services
  - import from FastAPI
"""

from dataclasses import dataclass, field
from typing import Literal

from app.analysis.matching import exact_matcher, fuzzy_matcher
from app.analysis.matching.exact_matcher import normalise
from app.analysis.matching.synonym_map import SYNONYMS

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

MatchType = Literal["EXACT", "SYNONYM", "FUZZY"]


@dataclass
class MatchedKeyword:
    """A resume keyword that was resolved against the JD."""

    keyword: str
    """The resume keyword as originally provided."""
    matchType: MatchType
    """How the match was achieved: EXACT, SYNONYM, or FUZZY."""


@dataclass
class MatchResult:
    """Complete result of the keyword matching pipeline."""

    matched: list[MatchedKeyword] = field(default_factory=list)
    """All resume keywords that were matched against a JD keyword."""
    missing: list[str] = field(default_factory=list)
    """JD keywords that were NOT found in the resume (resume is missing them)."""
    unresolved: list[str] = field(default_factory=list)
    """Resume keywords that did NOT match any JD keyword."""


# ---------------------------------------------------------------------------
# Build synonym lookup at import time
# ---------------------------------------------------------------------------

# alias_to_canonical: lowercase alias → canonical key
# Allows O(1) lookup in the synonym pass.
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canonical, _aliases in SYNONYMS.items():
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL[_alias] = _canonical


def _synonym_canonical(keyword: str) -> str | None:
    """Return the canonical form of ``keyword`` if it exists in the synonym map.

    Args:
        keyword: Raw keyword string.

    Returns:
        Canonical name (e.g. ``'Node.js'``) or ``None``.
    """
    return _ALIAS_TO_CANONICAL.get(normalise(keyword))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def match(
    resume_keywords: list[str],
    jd_keywords: list[str],
    fuzzy_threshold: int = fuzzy_matcher.DEFAULT_THRESHOLD,
) -> MatchResult:
    """Match resume keywords against JD keywords using a three-step pipeline.

    The pipeline always runs in this order:
      1. Exact matching   — case-insensitive, whitespace-normalised equality.
      2. Synonym matching — canonical-form lookup via ``synonym_map.SYNONYMS``.
      3. Fuzzy matching   — rapidfuzz WRatio with configurable threshold.

    Keywords resolved at an earlier step are removed from subsequent steps.

    Args:
        resume_keywords: Keywords extracted from the resume.
        jd_keywords: Keywords extracted from the job description.
        fuzzy_threshold: Minimum rapidfuzz similarity score [0–100] for the
            fuzzy pass.  Defaults to ``fuzzy_matcher.DEFAULT_THRESHOLD`` (85).

    Returns:
        ``MatchResult`` containing matched, missing, and unresolved keywords.
    """
    result = MatchResult()

    # Working copies — consumed as matches are found
    remaining_resume = list(resume_keywords)
    remaining_jd = list(jd_keywords)

    # ── Step 1: Exact matching ──────────────────────────────────────────────
    exact_matched, remaining_resume, remaining_jd = exact_matcher.match_all(
        remaining_resume, remaining_jd
    )
    for rk, _ in exact_matched:
        result.matched.append(MatchedKeyword(keyword=rk, matchType="EXACT"))

    # ── Step 2: Synonym matching ────────────────────────────────────────────
    still_unmatched_resume: list[str] = []
    for rk in remaining_resume:
        jd_match = _synonym_match(rk, remaining_jd)
        if jd_match is not None:
            result.matched.append(MatchedKeyword(keyword=rk, matchType="SYNONYM"))
            remaining_jd.remove(jd_match)
        else:
            still_unmatched_resume.append(rk)
    remaining_resume = still_unmatched_resume

    # ── Step 3: Fuzzy matching ──────────────────────────────────────────────
    fuzzy_matched, remaining_resume, remaining_jd = fuzzy_matcher.match_all(
        remaining_resume, remaining_jd, fuzzy_threshold
    )
    for rk, _ in fuzzy_matched:
        result.matched.append(MatchedKeyword(keyword=rk, matchType="FUZZY"))

    # ── Collect missing and unresolved ──────────────────────────────────────
    result.missing = remaining_jd          # JD keywords the resume lacks
    result.unresolved = remaining_resume   # resume keywords not in JD

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _synonym_match(resume_keyword: str, jd_keywords: list[str]) -> str | None:
    """Check whether ``resume_keyword`` and any JD keyword share a synonym group.

    Both the resume keyword and each JD keyword are resolved to their
    canonical forms.  If both resolve to the same canonical, it is a synonym
    match.

    Args:
        resume_keyword: A single resume keyword (already failed exact match).
        jd_keywords: JD keywords still available for matching.

    Returns:
        The JD keyword string that synonym-matched, or ``None``.
    """
    resume_canonical = _synonym_canonical(resume_keyword)
    if resume_canonical is None:
        return None

    for jd_kw in jd_keywords:
        jd_canonical = _synonym_canonical(jd_kw)
        if jd_canonical is not None and jd_canonical == resume_canonical:
            return jd_kw

    return None
