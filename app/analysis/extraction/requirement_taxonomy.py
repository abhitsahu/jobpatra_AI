"""Route extracted entities through the taxonomy-backed ATS scoring boundary.

Only recognized, score-bearing technical skills enter the required-skill and
keyword denominators. Tools are retained as preferred evidence and unknown,
soft-skill, culture, and business-language terms remain feedback-only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.analysis.extraction import keyword_extractor
from app.schemas.extraction import JDExtraction, ResumeExtraction
from app.services.taxonomy_service import get_taxonomy_service


_PREFERRED_CONTEXT = re.compile(
    r"\b(?:preferred|nice\s+to\s+have|bonus|plus|desirable)\b", re.IGNORECASE
)
_EXPERIENCE_REQUIREMENT = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class JDRequirementTaxonomy:
    """Taxonomy-routed JD requirements with non-scoring feedback preserved."""

    required_technical_skills: list[str]
    preferred_technical_skills: list[str]
    feedback_only: list[str]
    culture_signals: list[str]

    @property
    def domain_terms(self) -> list[str]:
        """Compatibility alias for the former score-bearing domain bucket."""
        return []

    @property
    def keyword_requirements(self) -> list[str]:
        """Return the only JD terms allowed into keyword-score coverage."""
        return self.required_technical_skills


def classify_jd_requirements(extraction: JDExtraction) -> JDRequirementTaxonomy:
    """Classify JD entities using taxonomy categories rather than heuristics."""
    required: list[str] = []
    preferred: list[str] = []
    feedback: list[str] = []

    for term in extraction.required_hard_skills:
        _route_term(term, required, preferred, feedback)
    for term in extraction.preferred_hard_skills:
        _route_term(term, required, preferred, feedback)
    for term in extraction.domain_terms:
        _route_term(term, required, preferred, feedback)
    for term in extraction.required_soft_skills:
        feedback.append(term.strip())

    feedback = _unique_terms(feedback)
    return JDRequirementTaxonomy(
        required_technical_skills=_unique_terms(required),
        preferred_technical_skills=_unique_terms(preferred),
        feedback_only=feedback,
        culture_signals=feedback,
    )


def resume_technical_evidence(extraction: ResumeExtraction) -> list[str]:
    """Return normalized, recognized technical evidence from a resume."""
    taxonomy = get_taxonomy_service()
    terms = [*extraction.hard_skills, *extraction.domain_terms]
    return _unique_terms(
        [taxonomy.normalize(term) for term in terms if taxonomy.get_category(term) != "unknown"]
    )


def fallback_resume_extraction(text: str) -> ResumeExtraction:
    """Build resume evidence exclusively from taxonomy-recognized skills."""
    return ResumeExtraction(hard_skills=keyword_extractor.extract(text))


def fallback_jd_extraction(text: str) -> JDExtraction:
    """Build a no-fluff JD extraction from taxonomy-recognized skills only."""
    taxonomy = get_taxonomy_service()
    required: list[str] = []
    preferred: list[str] = []

    for line in text.splitlines() or [text]:
        destination = preferred if _PREFERRED_CONTEXT.search(line) else required
        destination.extend(taxonomy.recognized_terms(line))

    if not required and not preferred:
        required.extend(taxonomy.recognized_terms(text))

    return JDExtraction(
        required_hard_skills=_unique_terms(required),
        preferred_hard_skills=_unique_terms(preferred),
        min_experience=_fallback_min_experience(text),
        required_education_level=_fallback_education_requirement(text),
    )


def _route_term(
    term: str,
    required: list[str],
    preferred: list[str],
    feedback: list[str],
) -> None:
    """Route one LLM entity without allowing unknown values into scoring."""
    cleaned = term.strip()
    if not cleaned:
        return
    taxonomy = get_taxonomy_service()
    if taxonomy.is_required_skill(cleaned):
        required.append(taxonomy.normalize(cleaned))
    elif taxonomy.is_preferred_skill(cleaned):
        preferred.append(taxonomy.normalize(cleaned))
    else:
        feedback.append(cleaned)


def _unique_terms(*groups: list[str]) -> list[str]:
    """Return non-empty terms in stable, case-insensitive unique order."""
    result: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for term in group:
            cleaned = term.strip()
            normalized = " ".join(cleaned.casefold().split())
            if cleaned and normalized not in seen:
                seen.add(normalized)
                result.append(cleaned)
    return result


def _fallback_min_experience(text: str) -> float:
    """Return an explicit years requirement from deterministic JD fallback text."""
    matches = [float(match.group(1)) for match in _EXPERIENCE_REQUIREMENT.finditer(text)]
    return max(matches, default=0.0)


def _fallback_education_requirement(text: str) -> str:
    """Return the highest explicit degree requirement found in fallback JD text."""
    lowered = text.casefold()
    if re.search(r"\b(?:ph\.?d|doctorate|doctoral)\b", lowered):
        return "phd"
    if re.search(r"\b(?:master'?s|m\.?(?:sc|tech|s|eng)|mba)\b", lowered):
        return "masters"
    if re.search(r"\b(?:bachelor'?s|b\.?(?:sc|tech|s|eng|e))\b", lowered):
        return "bachelors"
    if re.search(r"\b(?:associate|diploma)\b", lowered):
        return "associate"
    return "none"
