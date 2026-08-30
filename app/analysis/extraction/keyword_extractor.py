"""Extract taxonomy-recognized skills, prioritizing structured Skills sections.

The deterministic fallback may receive imperfect PDF text. It first scans a
``SKILLS`` block and labelled rows such as ``Frontend: HTML, CSS`` so explicit
resume skills survive parsing. Unknown prose is still excluded by taxonomy
normalization and cannot leak into ATS score denominators.
"""

from __future__ import annotations

import re

from app.services.taxonomy_service import get_taxonomy_service


_SKILLS_HEADER = re.compile(
    r"^\s*(?:technical\s+)?skills?(?:\s*(?:&|and)\s*(?:technologies|tools))?\s*:?\s*$",
    re.IGNORECASE,
)
_SECTION_HEADER = re.compile(r"^\s*[A-Z][A-Z\s&/-]{2,}:?\s*$")
_CATEGORY_SKILL_ROW = re.compile(
    r"^\s*(?:[-*•◦]\s*)?(?:frontend|backend|full[ -]?stack|programming(?:\s+languages)?|"
    r"databases?|cloud(?:\s*&\s*devops)?|devops|integrations?(?:\s*&\s*tools?)?|"
    r"tools?|frameworks?|libraries|ai|ml)\s*(?:&|and)?\s*[\w -]*\s*:\s*(.+)$",
    re.IGNORECASE,
)
_BULLET = re.compile(r"^\s*(?:[-*•◦]\s*)+")
_ITEM_SEPARATOR = re.compile(r"\s*(?:,|\||;|•)\s*")


def extract(text: str) -> list[str]:
    """Return canonical skills, including comma-separated Skills-section items."""
    taxonomy = get_taxonomy_service()
    structured_items = _extract_structured_skill_items(text)
    structured_skills = _normalize_items(structured_items)
    # Keep broad text scanning as a secondary source for unstructured resumes.
    return _unique_terms(structured_skills, taxonomy.recognized_terms(text))


def extract_from_skills_section(skills_text: str) -> list[str]:
    """Extract canonical skills from a raw ``ResumeSection.skills`` value.

    ``section_splitter`` removes the heading itself, so this function treats
    every line as Skills-section content. Comma-separated items and labelled
    category rows are both supported.
    """
    structured_items = _extract_structured_skill_items(skills_text, in_skills_section=True)
    return _unique_terms(_normalize_items(structured_items))


def _extract_structured_skill_items(
    text: str,
    in_skills_section: bool = False,
) -> list[str]:
    """Read Skills-section lines and category rows from parsed resume text."""
    items: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _SKILLS_HEADER.match(stripped):
            in_skills_section = True
            continue
        if in_skills_section and _SECTION_HEADER.match(stripped):
            in_skills_section = False

        category_match = _CATEGORY_SKILL_ROW.match(stripped)
        if category_match is not None:
            items.extend(_split_items(category_match.group(1)))
            continue
        if in_skills_section:
            items.extend(_split_items(_BULLET.sub("", stripped)))

    return items


def _normalize_items(items: list[str]) -> list[str]:
    """Keep only taxonomy-recognized explicit Skills-section items."""
    taxonomy = get_taxonomy_service()
    return [
        taxonomy.normalize(item)
        for item in items
        if taxonomy.get_category(item) != "unknown"
    ]


def _split_items(value: str) -> list[str]:
    """Split a skills row while preserving multi-word skill names."""
    return [item.strip() for item in _ITEM_SEPARATOR.split(value) if item.strip()]


def _unique_terms(*groups: list[str]) -> list[str]:
    """Return stable canonical terms without duplicates."""
    result: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for term in group:
            key = " ".join(term.casefold().split())
            if term and key not in seen:
                seen.add(key)
                result.append(term)
    return result
