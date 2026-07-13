"""Exact matcher — Step 1 of the keyword matching pipeline.

This module has ONE responsibility: determine whether a resume keyword
matches a JD keyword through case-insensitive, whitespace-normalised
exact string comparison.

It does NOT:
  - perform synonym lookups
  - perform fuzzy matching
  - call AI or external services

All functions are pure. No I/O. No state. No FastAPI imports.
"""

import re


# Public API


def normalise(keyword: str) -> str:
    """Normalise a keyword for exact comparison.

    Steps:
      1. Lowercase.
      2. Collapse any run of whitespace to a single space.
      3. Strip leading/trailing whitespace.
      4. Normalise common punctuation variants (e.g. strip trailing dots).

    This function is intentionally exposed so that other matchers and tests
    can apply the same normalisation without reimplementing it.

    Args:
        keyword: Raw keyword string.

    Returns:
        Normalised lowercase string.
    """
    lowered = keyword.lower()
    collapsed = re.sub(r"\s+", " ", lowered).strip()
    return collapsed


def match(resume_keyword: str, jd_keywords: list[str]) -> str | None:
    """Find an exact match for ``resume_keyword`` within ``jd_keywords``.

    Comparison is case-insensitive and whitespace-normalised (via
    ``normalise()``).  Punctuation is preserved — "Node.js" will NOT match
    "Nodejs" here; that belongs to synonym or fuzzy matching.

    Args:
        resume_keyword: A single keyword from the resume.
        jd_keywords: All keywords extracted from the job description.

    Returns:
        The original JD keyword string that matched, or ``None`` if no exact
        match was found.
    """
    normalised_resume = normalise(resume_keyword)
    for jd_kw in jd_keywords:
        if normalise(jd_kw) == normalised_resume:
            return jd_kw
    return None


def match_all(
    resume_keywords: list[str],
    jd_keywords: list[str],
) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """Run exact matching across all resume keywords against all JD keywords.

    Args:
        resume_keywords: Keywords extracted from the resume.
        jd_keywords: Keywords extracted from the job description.

    Returns:
        A 3-tuple of:
          - matched: list of (resume_kw, jd_kw) pairs that matched exactly.
          - unmatched_resume: resume keywords that did NOT match any JD keyword.
          - unmatched_jd: JD keywords that were NOT matched by any resume keyword.
    """
    matched: list[tuple[str, str]] = []
    unmatched_resume: list[str] = []
    remaining_jd = list(jd_keywords)

    for rk in resume_keywords:
        jd_match = match(rk, remaining_jd)
        if jd_match is not None:
            matched.append((rk, jd_match))
            remaining_jd.remove(jd_match)
        else:
            unmatched_resume.append(rk)

    return matched, unmatched_resume, remaining_jd
