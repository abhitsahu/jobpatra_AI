"""Fuzzy matcher — Step 3 of the keyword matching pipeline.

This module has ONE responsibility: catch spelling mistakes and minor
variations that escaped exact matching (Step 1) and synonym matching (Step 2).

It MUST only be called AFTER exact and synonym matching have already been
attempted.  It does NOT call exact or synonym matchers — orchestration lives
in keyword_matcher.py.

Dependency: rapidfuzz
  - Chosen because it is significantly faster than python-Levenshtein and
    fuzzywuzzy, ships with pre-built wheels for all major platforms, and
    provides a clean, type-annotated API.
  - rapidfuzz.fuzz.WRatio is used because it handles substring and token-
    order differences gracefully, which matters for multi-word tech terms
    (e.g. "machine learning" vs "learning machine").

Similarity threshold (default: 85)
  - Values ≥ 85 have low false-positive rates for tech keywords in practice.
  - Deliberate typos like "Docekr" → "Docker" (score ≈ 92) pass.
  - Large differences like "Python" → "PHP" (score ≈ 57) are rejected.
  - The threshold is configurable via the ``threshold`` parameter.

All functions are pure with respect to the rapidfuzz library. No I/O.
No state. No FastAPI imports.
"""

from rapidfuzz import fuzz

from app.analysis.matching.exact_matcher import normalise

# Default similarity threshold. Must be in range [0, 100].
# Calibrated at 82 so that single-transposition typos like 'Docekr' → 'Docker'
# (WRatio ≈ 83) are accepted, while large differences like 'Python' → 'Java'
# (WRatio ≈ 57) are safely rejected.  Callers can override via ``threshold``.
DEFAULT_THRESHOLD: int = 82


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def match(
    resume_keyword: str,
    jd_keywords: list[str],
    threshold: int = DEFAULT_THRESHOLD,
) -> str | None:
    """Find the best fuzzy match for ``resume_keyword`` within ``jd_keywords``.

    Uses ``rapidfuzz.fuzz.WRatio`` which combines multiple fuzzy strategies
    (simple ratio, partial ratio, token-sort, token-set) and returns the
    highest score.  Only matches above ``threshold`` are accepted.

    Args:
        resume_keyword: A single keyword from the resume (already failed exact
            and synonym matching).
        jd_keywords: JD keywords that have not yet been matched.
        threshold: Minimum similarity score [0–100] to accept a match.
            Defaults to ``DEFAULT_THRESHOLD`` (85).

    Returns:
        The JD keyword string that fuzzy-matched best, or ``None`` if no
        candidate exceeds the threshold.
    """
    best_score: float = 0.0
    best_match: str | None = None

    norm_resume = normalise(resume_keyword)

    for jd_kw in jd_keywords:
        score = fuzz.WRatio(norm_resume, normalise(jd_kw))
        if score >= threshold and score > best_score:
            best_score = score
            best_match = jd_kw

    return best_match


def match_all(
    resume_keywords: list[str],
    jd_keywords: list[str],
    threshold: int = DEFAULT_THRESHOLD,
) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """Run fuzzy matching across all remaining resume keywords.

    Args:
        resume_keywords: Resume keywords that are still unmatched after exact
            and synonym passes.
        jd_keywords: JD keywords that are still unmatched after exact and
            synonym passes.
        threshold: Minimum similarity score to accept.

    Returns:
        A 3-tuple of:
          - matched: (resume_kw, jd_kw) pairs that matched via fuzzy scoring.
          - unmatched_resume: resume keywords that still have no match.
          - unmatched_jd: JD keywords still unmatched after this pass.
    """
    matched: list[tuple[str, str]] = []
    unmatched_resume: list[str] = []
    remaining_jd = list(jd_keywords)

    for rk in resume_keywords:
        jd_match = match(rk, remaining_jd, threshold)
        if jd_match is not None:
            matched.append((rk, jd_match))
            remaining_jd.remove(jd_match)
        else:
            unmatched_resume.append(rk)

    return matched, unmatched_resume, remaining_jd
