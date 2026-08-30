"""Keyword matcher — the single public entry point for the matching pipeline.

This module orchestrates the four-step keyword matching pipeline:

  Step 1 — Exact matching    (exact_matcher)
  Step 2 — Synonym matching  (synonym_map)
  Step 3 — Fuzzy matching    (fuzzy_matcher)
  Step 4 — Semantic matching (semantic_matcher)  [optional, requires provider]

The order is strict and can never be changed.  A keyword resolved at an
earlier step is removed from further consideration.

After all four steps every resume keyword belongs to exactly one category:
  - ``matched``  — found in the JD (via any match type)
  - ``missing``  — JD keywords absent from the resume
  - ``unresolved`` — resume keywords not in the JD *when semantic step is
                     skipped* (no provider supplied).  When a provider IS
                     supplied this list is always empty.

Public API
----------
``match(resume_keywords, jd_keywords, ...) -> MatchResult``

Other modules must NEVER import exact_matcher, fuzzy_matcher, or
semantic_matcher directly.

matchType values: ``"EXACT"`` | ``"SYNONYM"`` | ``"FUZZY"`` | ``"SEMANTIC"``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.analysis.matching import exact_matcher, fuzzy_matcher
from app.analysis.matching.exact_matcher import normalise
from app.analysis.matching.semantic_matcher import (
    EmbeddingProvider,
    SemanticMatchResult,
    match_unresolved,
)
from app.analysis.matching.synonym_map import SYNONYMS

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

MatchType = Literal["EXACT", "SYNONYM", "FUZZY", "RELATED"]


@dataclass
class MatchedKeyword:
    """A resume keyword resolved against the JD."""

    keyword: str
    """The resume keyword as originally provided."""
    matchType: MatchType
    """How the match was achieved."""
    similarity: float | None = None
    """Cosine similarity score for related-concept matches."""
    matched_jd_keyword: str | None = None
    """The JD term satisfied or related to this resume term."""
    is_related_concept: bool = False
    """Whether this is advisory semantic evidence rather than a direct match."""


@dataclass
class MatchResult:
    """Complete result of the keyword matching pipeline."""

    matched: list[MatchedKeyword] = field(default_factory=list)
    """All resume keywords matched against a JD keyword."""
    missing: list[str] = field(default_factory=list)
    """JD keywords not found in the resume."""
    unresolved: list[str] = field(default_factory=list)
    """Resume keywords not matched to any JD keyword.

    This list is empty when an ``EmbeddingProvider`` is supplied because the
    semantic pass classifies every remaining keyword as either SEMANTIC-matched
    or MISSING.  Without a provider, any keywords that survive fuzzy matching
    land here.
    """
    related: list[MatchedKeyword] = field(default_factory=list)
    """Embedding-derived related concepts; never score-bearing."""


# ---------------------------------------------------------------------------
# Build synonym lookup at import time
# ---------------------------------------------------------------------------

_ALIAS_TO_CANONICAL: dict[str, set[str]] = {}
for _canonical, _aliases in SYNONYMS.items():
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL.setdefault(_alias, set()).add(_canonical)


def _synonym_canonicals(keyword: str) -> set[str]:
    """Return every synonym group containing ``keyword``.

    Some aliases, such as ``terraform`` and ``github actions``, are valid in
    more than one group. Keeping every group prevents later map entries from
    silently overwriting earlier groups during module import.
    """
    return _ALIAS_TO_CANONICAL.get(normalise(keyword), set())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def match(
    resume_keywords: list[str],
    jd_keywords: list[str],
    fuzzy_threshold: int = fuzzy_matcher.DEFAULT_THRESHOLD,
    embedding_provider: EmbeddingProvider | None = None,
    semantic_threshold: float | None = None,
) -> MatchResult:
    """Match resume keywords against JD keywords using a four-step pipeline.

    Steps run in strict order:
      1. Exact — case-insensitive, whitespace-normalised equality.
      2. Synonym — canonical-form lookup via ``synonym_map.SYNONYMS``.
      3. Fuzzy — rapidfuzz WRatio with ``fuzzy_threshold``.
      4. Semantic — cosine similarity via ``embedding_provider`` (optional).

    When ``embedding_provider`` is ``None`` the semantic step is skipped and
    unresolved keywords are returned in ``MatchResult.unresolved``.

    Args:
        resume_keywords: Keywords extracted from the resume.
        jd_keywords: Keywords extracted from the job description.
        fuzzy_threshold: Minimum rapidfuzz score [0–100].
        embedding_provider: An ``EmbeddingProvider`` instance for the semantic
            pass.  If ``None``, Step 4 is skipped.
        semantic_threshold: Override for the semantic similarity threshold.
            When ``None``, ``semantic_matcher.SIMILARITY_THRESHOLD`` is used.

    Returns:
        ``MatchResult`` with matched, missing, and optionally unresolved lists.
    """
    result = MatchResult()
    remaining_resume = list(resume_keywords)
    remaining_jd = list(jd_keywords)

    # ── Step 1: Exact ───────────────────────────────────────────────────────
    exact_matched, remaining_resume, remaining_jd = exact_matcher.match_all(
        remaining_resume, remaining_jd
    )
    for rk, jd_kw in exact_matched:
        result.matched.append(
            MatchedKeyword(keyword=rk, matchType="EXACT", matched_jd_keyword=jd_kw)
        )

    # ── Step 2: Synonym ─────────────────────────────────────────────────────
    still_unmatched: list[str] = []
    for rk in remaining_resume:
        jd_match = _synonym_match(rk, remaining_jd)
        if jd_match is not None:
            result.matched.append(
                MatchedKeyword(keyword=rk, matchType="SYNONYM", matched_jd_keyword=jd_match)
            )
            remaining_jd.remove(jd_match)
        else:
            still_unmatched.append(rk)
    remaining_resume = still_unmatched

    # ── Step 3: Fuzzy ───────────────────────────────────────────────────────
    fuzzy_matched, remaining_resume, remaining_jd = fuzzy_matcher.match_all(
        remaining_resume, remaining_jd, fuzzy_threshold
    )
    for rk, jd_kw in fuzzy_matched:
        result.matched.append(
            MatchedKeyword(keyword=rk, matchType="FUZZY", matched_jd_keyword=jd_kw)
        )

    # ── Step 4: Semantic (optional) ─────────────────────────────────────────
    if embedding_provider is not None and remaining_resume:
        kwargs: dict = {}
        if semantic_threshold is not None:
            kwargs["threshold"] = semantic_threshold

        semantic_results, remaining_jd = match_unresolved(
            resume_keywords=remaining_resume,
            jd_keywords=remaining_jd,
            provider=embedding_provider,
            consume_matches=False,
            **kwargs,
        )
        _apply_semantic_results(result, semantic_results)
        remaining_resume = []  # semantic pass classifies every keyword
    else:
        # No provider — unresolved keywords are returned as-is
        result.unresolved = remaining_resume
        remaining_resume = []

    result.missing = remaining_jd
    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _synonym_match(resume_keyword: str, jd_keywords: list[str]) -> str | None:
    """Return the JD keyword that shares a synonym group with ``resume_keyword``.

    Args:
        resume_keyword: A single resume keyword (already failed exact match).
        jd_keywords: JD keywords still available for matching.

    Returns:
        Matching JD keyword string, or ``None``.
    """
    resume_canonicals = _synonym_canonicals(resume_keyword)
    if not resume_canonicals:
        return None
    for jd_kw in jd_keywords:
        jd_canonicals = _synonym_canonicals(jd_kw)
        if resume_canonicals.intersection(jd_canonicals):
            return jd_kw
    return None


def _apply_semantic_results(
    result: MatchResult,
    semantic_results: list[SemanticMatchResult],
) -> None:
    """Merge semantic match results into ``result`` in place.

    Related items are separate from direct matches. They never increase
    keyword or skills coverage and leave the corresponding JD term missing.

    Args:
        result: The ``MatchResult`` being built.
        semantic_results: Output from ``semantic_matcher.match_unresolved()``.
    """
    for sr in semantic_results:
        if sr.matched:
            result.related.append(
                MatchedKeyword(
                    keyword=sr.keyword,
                    matchType="RELATED",
                    similarity=sr.similarity,
                    matched_jd_keyword=sr.matched_jd_keyword,
                    is_related_concept=True,
                )
            )
