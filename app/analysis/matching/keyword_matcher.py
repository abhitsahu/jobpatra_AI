"""Keyword matcher — the single public entry point for the matching pipeline.

This module orchestrates the four-step keyword matching pipeline:

  Step 1 — Exact matching    (exact_matcher)
  Step 2 — Taxonomy relationship matching
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

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Literal

from app.analysis.matching import exact_matcher, fuzzy_matcher
from app.analysis.matching.exact_matcher import normalise
from app.analysis.matching.semantic_matcher import (
    EmbeddingProvider,
    SemanticMatchResult,
    match_unresolved,
)
from app.services.taxonomy_service import TaxonomyService, get_taxonomy_service

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


@dataclass(frozen=True)
class _TaxonomyTerm:
    """A caller-visible term paired with its taxonomy-normalized value."""

    original: str
    normalized: str


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
      2. Taxonomy relationship — parent/child and related-skill graph lookup.
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
    taxonomy = get_taxonomy_service()
    result = MatchResult()
    remaining_resume = _normalise_terms(resume_keywords, taxonomy.normalize)
    remaining_jd = _normalise_terms(jd_keywords, taxonomy.normalize)

    # ── Step 1: Exact ───────────────────────────────────────────────────────
    exact_matched, _, _ = exact_matcher.match_all(
        [term.normalized for term in remaining_resume],
        [term.normalized for term in remaining_jd],
    )
    exact_pairs, remaining_resume, remaining_jd = _consume_pairs(
        remaining_resume, remaining_jd, exact_matched
    )
    for resume_term, jd_term in exact_pairs:
        result.matched.append(
            MatchedKeyword(
                keyword=resume_term.original,
                matchType="EXACT",
                matched_jd_keyword=jd_term.original,
            )
        )

    # ── Step 2: Taxonomy relationship ───────────────────────────────────────
    still_unmatched: list[_TaxonomyTerm] = []
    for resume_term in remaining_resume:
        jd_match = _taxonomy_match(resume_term, remaining_jd, taxonomy)
        if jd_match is not None:
            result.matched.append(
                MatchedKeyword(
                    keyword=resume_term.original,
                    matchType="SYNONYM",
                    matched_jd_keyword=jd_match.original,
                )
            )
            remaining_jd.remove(jd_match)
        else:
            still_unmatched.append(resume_term)
    remaining_resume = still_unmatched

    # ── Step 3: Fuzzy ───────────────────────────────────────────────────────
    fuzzy_matched, _, _ = fuzzy_matcher.match_all(
        [term.normalized for term in remaining_resume],
        [term.normalized for term in remaining_jd],
        fuzzy_threshold,
    )
    fuzzy_pairs, remaining_resume, remaining_jd = _consume_pairs(
        remaining_resume, remaining_jd, fuzzy_matched
    )
    for resume_term, jd_term in fuzzy_pairs:
        result.matched.append(
            MatchedKeyword(
                keyword=resume_term.original,
                matchType="FUZZY",
                matched_jd_keyword=jd_term.original,
            )
        )

    # ── Step 4: Semantic (optional) ─────────────────────────────────────────
    if embedding_provider is not None and remaining_resume:
        semantic_jd_terms = list(remaining_jd)
        kwargs: dict = {}
        if semantic_threshold is not None:
            kwargs["threshold"] = semantic_threshold

        semantic_results, semantic_remaining_jd = match_unresolved(
            resume_keywords=[term.normalized for term in remaining_resume],
            jd_keywords=[term.normalized for term in semantic_jd_terms],
            provider=embedding_provider,
            consume_matches=False,
            **kwargs,
        )
        _apply_semantic_results(result, semantic_results, remaining_resume, semantic_jd_terms)
        remaining_jd = _remaining_terms(semantic_jd_terms, semantic_remaining_jd)
        remaining_resume = []  # semantic pass classifies every keyword
    else:
        # No provider — unresolved keywords are returned as-is
        result.unresolved = [term.original for term in remaining_resume]
        remaining_resume = []

    result.missing = [term.original for term in remaining_jd]
    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _normalise_terms(
    terms: list[str], normalise_taxonomy: Callable[[str], str]
) -> list[_TaxonomyTerm]:
    """Preserve display values while canonicalizing matcher inputs."""
    return [
        _TaxonomyTerm(original=term, normalized=normalise_taxonomy(term))
        for term in terms
    ]


def _remaining_terms(
    terms: list[_TaxonomyTerm], remaining_values: list[str]
) -> list[_TaxonomyTerm]:
    """Map a matcher remainder back to its original display values."""
    remaining_counts = Counter(normalise(value) for value in remaining_values)
    result: list[_TaxonomyTerm] = []
    for term in terms:
        key = normalise(term.normalized)
        if remaining_counts[key] > 0:
            remaining_counts[key] -= 1
            result.append(term)
    return result


def _consume_pairs(
    resume_terms: list[_TaxonomyTerm],
    jd_terms: list[_TaxonomyTerm],
    pairs: list[tuple[str, str]],
) -> tuple[list[tuple[_TaxonomyTerm, _TaxonomyTerm]], list[_TaxonomyTerm], list[_TaxonomyTerm]]:
    """Map exact/fuzzy canonical pairs back to originals without changing pairs."""
    remaining_resume = list(resume_terms)
    remaining_jd = list(jd_terms)
    resolved: list[tuple[_TaxonomyTerm, _TaxonomyTerm]] = []

    for resume_value, jd_value in pairs:
        resume_index = next(
            index
            for index, term in enumerate(remaining_resume)
            if normalise(term.normalized) == normalise(resume_value)
        )
        jd_index = next(
            index
            for index, term in enumerate(remaining_jd)
            if normalise(term.normalized) == normalise(jd_value)
        )
        resolved.append((remaining_resume.pop(resume_index), remaining_jd.pop(jd_index)))

    return resolved, remaining_resume, remaining_jd


def _taxonomy_match(
    resume_term: _TaxonomyTerm,
    jd_terms: list[_TaxonomyTerm],
    taxonomy: TaxonomyService,
) -> _TaxonomyTerm | None:
    """Return a deterministic parent/child or related taxonomy match."""
    for jd_term in jd_terms:
        if taxonomy.are_related(resume_term.normalized, jd_term.normalized):
            return jd_term
    return None


def _apply_semantic_results(
    result: MatchResult,
    semantic_results: list[SemanticMatchResult],
    resume_terms: list[_TaxonomyTerm],
    jd_terms: list[_TaxonomyTerm],
) -> None:
    """Merge semantic match results into ``result`` in place.

    Related items are separate from direct matches. They never increase
    keyword or skills coverage and leave the corresponding JD term missing.

    Args:
        result: The ``MatchResult`` being built.
        semantic_results: Output from ``semantic_matcher.match_unresolved()``.
    """
    resume_originals = {term.normalized: term.original for term in resume_terms}
    jd_originals = {term.normalized: term.original for term in jd_terms}
    for sr in semantic_results:
        if sr.matched:
            result.related.append(
                MatchedKeyword(
                    keyword=resume_originals.get(sr.keyword, sr.keyword),
                    matchType="RELATED",
                    similarity=sr.similarity,
                    matched_jd_keyword=jd_originals.get(
                        sr.matched_jd_keyword or "", sr.matched_jd_keyword
                    ),
                    is_related_concept=True,
                )
            )
