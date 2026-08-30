"""Classify extracted ATS entities into score-bearing requirement groups.

The LLM extraction layer supplies candidate entities. This module makes the
scoring boundary deterministic: explicit technical requirements are scored,
while culture signals are retained for feedback without diluting technical
coverage scores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.analysis.extraction import skill_extractor
from app.analysis.matching.synonym_map import SYNONYMS
from app.schemas.extraction import JDExtraction, ResumeExtraction


_CULTURE_SIGNALS = frozenset(
    {
        "first principles thinking",
        "passion for reliability",
        "enthusiasm for framework architecture",
        "problem solving",
        "problem-solving",
    }
)

_PREFERRED_CONTEXT = re.compile(
    r"\b(?:preferred|nice\s+to\s+have|bonus|plus|desirable)\b", re.IGNORECASE
)
_EXPERIENCE_REQUIREMENT = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b", re.IGNORECASE
)
_UNSAFE_SHORT_ALIASES = frozenset({"ai", "dl", "es", "go", "js", "ml", "np", "pd", "rb", "sh", "tf", "ts"})


@dataclass(frozen=True)
class JDRequirementTaxonomy:
    """Score-bearing and feedback-only requirements extracted from a JD."""

    required_technical_skills: list[str]
    preferred_technical_skills: list[str]
    domain_terms: list[str]
    culture_signals: list[str]

    @property
    def keyword_requirements(self) -> list[str]:
        """Return technical/domain keywords used by general keyword coverage."""
        return _unique_terms(
            self.required_technical_skills,
            self.preferred_technical_skills,
            self.domain_terms,
        )


def classify_jd_requirements(extraction: JDExtraction) -> JDRequirementTaxonomy:
    """Split an LLM JD extraction into technical and feedback-only groups."""
    required_technical_skills: list[str] = []
    preferred_technical_skills: list[str] = []
    domain_terms: list[str] = []
    culture_signals = list(extraction.required_soft_skills)

    for term in extraction.required_hard_skills:
        if _normalise(term) in _CULTURE_SIGNALS:
            culture_signals.append(term)
        else:
            required_technical_skills.append(term)

    for term in extraction.preferred_hard_skills:
        if _normalise(term) in _CULTURE_SIGNALS:
            culture_signals.append(term)
        else:
            preferred_technical_skills.append(term)

    for term in extraction.domain_terms:
        if _normalise(term) in _CULTURE_SIGNALS:
            culture_signals.append(term)
        else:
            domain_terms.append(term)

    return JDRequirementTaxonomy(
        required_technical_skills=_unique_terms(required_technical_skills),
        preferred_technical_skills=_unique_terms(preferred_technical_skills),
        domain_terms=_unique_terms(domain_terms),
        culture_signals=_unique_terms(culture_signals),
    )


def resume_technical_evidence(extraction: ResumeExtraction) -> list[str]:
    """Return explicit technical skills and domain concepts from a resume."""
    return _unique_terms(extraction.hard_skills, extraction.domain_terms)


def fallback_resume_extraction(text: str) -> ResumeExtraction:
    """Build explicit technical resume evidence without an LLM.

    This fallback deliberately uses only the maintained skills dictionary and
    synonym dictionary. It never turns general prose into ATS requirements.
    """
    skills = [match.canonical for match in skill_extractor.extract(text).skills]
    terms = _detect_technical_terms(text)
    return ResumeExtraction(hard_skills=_unique_terms(skills, terms))


def fallback_jd_extraction(text: str) -> JDExtraction:
    """Build a technical-only JD extraction when Hybrid AI is unavailable."""
    required: list[str] = []
    preferred: list[str] = []
    domain_terms: list[str] = []

    for line in text.splitlines() or [text]:
        line_terms = _detect_technical_terms(line)
        if _PREFERRED_CONTEXT.search(line):
            preferred.extend(line_terms)
        else:
            required.extend(line_terms)

    # A one-line JD has no meaningful line boundaries; preserve the detected
    # technical terms as required rather than falling back to arbitrary tokens.
    if not required and not preferred:
        required.extend(_detect_technical_terms(text))

    for term in _unique_terms(required, preferred):
        if term.lower() in {"distributed systems", "payment orchestration", "multi-dc architecture", "self-healing systems", "traffic routing", "anomaly detection", "payment tokenization", "fraud & risk management", "edge computing", "low-code/no-code", "api integrations", "infrastructure as code"}:
            domain_terms.append(term)

    return JDExtraction(
        required_hard_skills=_unique_terms(required),
        preferred_hard_skills=_unique_terms(preferred),
        domain_terms=_unique_terms(domain_terms),
        min_experience=_fallback_min_experience(text),
        required_education_level=_fallback_education_requirement(text),
    )


def _unique_terms(*groups: list[str]) -> list[str]:
    """Return non-empty terms in stable, case-insensitive unique order."""
    result: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for term in group:
            cleaned = term.strip()
            normalised = _normalise(cleaned)
            if cleaned and normalised not in seen:
                seen.add(normalised)
                result.append(cleaned)
    return result


def _normalise(term: str) -> str:
    """Normalise a requirement for taxonomy membership and de-duplication."""
    return " ".join(term.lower().split())


def _detect_technical_terms(text: str) -> list[str]:
    """Find only explicitly named terms from maintained technical dictionaries."""
    detected: list[str] = []
    lowered = text.lower()
    for canonical, aliases in SYNONYMS.items():
        canonical_normalised = _normalise(canonical)
        if canonical_normalised in _CULTURE_SIGNALS:
            continue
        for alias in aliases:
            normalised_alias = _normalise(alias)
            if normalised_alias in _UNSAFE_SHORT_ALIASES:
                continue
            if re.search(rf"(?<![a-z0-9+#.]){re.escape(normalised_alias)}(?![a-z0-9+#.])", lowered):
                detected.append(canonical)
                break
    return _unique_terms(detected)


def _fallback_min_experience(text: str) -> float:
    """Return an explicit years requirement from deterministic JD fallback text."""
    matches = [float(match.group(1)) for match in _EXPERIENCE_REQUIREMENT.finditer(text)]
    return max(matches, default=0.0)


def _fallback_education_requirement(text: str) -> str:
    """Return the highest explicit degree requirement found in fallback JD text."""
    lowered = text.lower()
    if re.search(r"\b(?:ph\.?d|doctorate|doctoral)\b", lowered):
        return "phd"
    if re.search(r"\b(?:master'?s|m\.?(?:sc|tech|s|eng)|mba)\b", lowered):
        return "masters"
    if re.search(r"\b(?:bachelor'?s|b\.?(?:sc|tech|s|eng|e))\b", lowered):
        return "bachelors"
    if re.search(r"\b(?:associate|diploma)\b", lowered):
        return "associate"
    return "none"
