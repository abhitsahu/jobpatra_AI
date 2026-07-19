"""Section splitter — divide cleaned resume text into named sections.

This module has ONE responsibility: use heuristic heading detection to split
a cleaned resume into its logical sections (Summary, Experience, Education,
Skills, etc.) and return a typed, structured result.
"""

import re
from dataclasses import dataclass, field
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ResumeSection:
    """Structured representation of a resume's named sections.

    Each field holds the raw text body of that section, or ``None`` if the
    section was not found in the resume.
    """

    summary: str | None = field(default=None)
    experience: str | None = field(default=None)
    education: str | None = field(default=None)
    skills: str | None = field(default=None)
    projects: str | None = field(default=None)
    certifications: str | None = field(default=None)
    languages: str | None = field(default=None)
    other: str | None = field(default=None)
    """Content that appears before the first recognized heading, if any."""


# ---------------------------------------------------------------------------
# Heading vocabulary
# ---------------------------------------------------------------------------

_HEADING_MAP: dict[str, tuple[str, ...]] = {
    "summary": (
        "summary",
        "professional summary",
        "profile",
        "about me",
        "objective",
        "career objective",
        "overview",
    ),
    "experience": (
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "employment",
        "work history",
        "career history",
    ),
    "education": (
        "education",
        "academic background",
        "academic history",
        "qualifications",
        "educational background",
    ),
    "skills": (
        "skills",
        "technical skills",
        "core competencies",
        "competencies",
        "technologies",
        "tools",
        "key skills",
    ),
    "projects": (
        "projects",
        "personal projects",
        "key projects",
        "notable projects",
        "portfolio",
    ),
    "certifications": (
        "certifications",
        "certificates",
        "certification",
        "professional certifications",
        "licenses",
        "accreditations",
    ),
    "languages": (
        "languages",
        "language skills",
        "spoken languages",
    ),
}

# Flat list of all aliases for sorting and regex creation
_ALL_ALIASES = sorted(
    (alias for aliases in _HEADING_MAP.values() for alias in aliases),
    key=len,
    reverse=True,
)


def _match_heading(line: str) -> str | None:
    """Return the canonical section key if ``line`` is a known heading, else None.

    First tries an exact match on cleaned lowercase text, and falls back to
    fuzzy matching via rapidfuzz.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 40:
        return None

    # Clean punctuation and trailing separators commonly found in headings
    lower = stripped.lower().rstrip(':.- ')

    # Fast path: exact match
    for canonical, aliases in _HEADING_MAP.items():
        if lower in aliases:
            return canonical

    # Fuzzy match fallback using rapidfuzz ratio
    for canonical, aliases in _HEADING_MAP.items():
        for alias in aliases:
            if fuzz.ratio(lower, alias) >= 88.0:
                return canonical

    return None



def split(text: str) -> ResumeSection:
    """Split cleaned resume text into named sections for compatibility.

    Uses _match_heading to split text. Any text before the first detected
    heading is placed under 'other' to satisfy compatibility constraints.
    """
    sections: dict[str, str] = {}
    current_key: str | None = None
    buffer: list[str] = []

    for line in text.splitlines():
        canonical = _match_heading(line)
        if canonical is not None:
            _flush(sections, current_key, buffer)
            buffer = []
            current_key = canonical
        else:
            buffer.append(line)

    _flush(sections, current_key, buffer)

    return ResumeSection(
        summary=sections.get("summary"),
        experience=sections.get("experience"),
        education=sections.get("education"),
        skills=sections.get("skills"),
        projects=sections.get("projects"),
        certifications=sections.get("certifications"),
        languages=sections.get("languages"),
        other=sections.get(None),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _flush(
    sections: dict[str, str],
    key: str | None,
    buffer: list[str],
) -> None:
    """Write the accumulated buffer into ``sections`` under ``key``."""
    content = "\n".join(buffer).strip()
    if content:
        sections[key] = content  # type: ignore[index]
