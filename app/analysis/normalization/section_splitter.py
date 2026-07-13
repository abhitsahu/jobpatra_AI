"""Section splitter — divide cleaned resume text into named sections.

This module has ONE responsibility: use heuristic heading detection to split
a cleaned resume into its logical sections (Summary, Experience, Education,
Skills, etc.) and return a typed, structured result.

Limitations (by design):
  - Detection is heuristic. A heading is recognized by matching it against a
    dictionary of known section names. Exotic or highly personalized headings
    may not be detected.
  - If a section is absent from the resume the corresponding field is ``None``.
    Missing sections NEVER raise an exception.
  - The quality and completeness of the split depends directly on how well the
    text was cleaned before being passed here. Always run text_cleaner.clean()
    first.
  - Content between the last detected heading and end-of-document is assigned
    to that final section.

This module does NOT:
  - extract keywords, skills, or entities from section content
  - judge resume quality
  - calculate ATS scores
  - call any external service or AI

All functions are pure: same input always produces the same output.
No I/O. No file access. No FastAPI imports.
"""

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ResumeSection:
    """Structured representation of a resume's named sections.

    Each field holds the raw text body of that section, or ``None`` if the
    section was not found in the resume.  No extraction or analysis is
    performed on the content — that belongs to later phases.
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

# Each key is the canonical field name on ResumeSection.
# Values are all recognized heading aliases for that section.
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


def _build_heading_pattern() -> re.Pattern[str]:
    """Compile a regex that matches any known section heading on its own line.

    A heading line is defined as a line whose stripped, lowercased content
    exactly matches one of the aliases in ``_HEADING_MAP``.  The pattern
    is case-insensitive and anchored to the start/end of each line.
    """
    all_aliases = sorted(
        (alias for aliases in _HEADING_MAP.values() for alias in aliases),
        key=len,
        reverse=True,  # longer aliases first to avoid partial matches
    )
    escaped = [re.escape(a) for a in all_aliases]
    pattern = r"^(?:" + "|".join(escaped) + r")\s*$"
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE)


_HEADING_RE: re.Pattern[str] = _build_heading_pattern()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split(text: str) -> ResumeSection:
    """Split cleaned resume text into named sections.

    Scans the text line by line for recognized section headings.  Text
    between two consecutive headings is assigned to the first heading's
    section.  Any text before the first detected heading is stored in
    ``ResumeSection.other``.

    Args:
        text: Cleaned resume text (output of ``text_cleaner.clean()``).

    Returns:
        A ``ResumeSection`` dataclass with each recognized section populated.
        Unrecognized or absent sections are ``None``.  This function never
        raises.
    """
    sections: dict[str, str] = {}
    current_key: str | None = None
    buffer: list[str] = []

    for line in text.splitlines():
        canonical = _match_heading(line)
        if canonical is not None:
            # Flush buffer into previous section
            _flush(sections, current_key, buffer)
            buffer = []
            current_key = canonical
        else:
            buffer.append(line)

    # Flush final section
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


def _match_heading(line: str) -> str | None:
    """Return the canonical section key if ``line`` is a known heading, else None.

    Args:
        line: A single line of text.

    Returns:
        Canonical key (e.g. ``"experience"``) or ``None``.
    """
    stripped = line.strip()
    if not stripped:
        return None

    if not _HEADING_RE.fullmatch(stripped):
        return None

    lower = stripped.lower()
    for canonical, aliases in _HEADING_MAP.items():
        if lower in aliases:
            return canonical
    return None


def _flush(
    sections: dict[str, str],
    key: str | None,
    buffer: list[str],
) -> None:
    """Write the accumulated buffer into ``sections`` under ``key``.

    Strips leading/trailing blank lines from the buffered content.
    If the buffer is empty after stripping, nothing is written.

    Args:
        sections: Mutable dict of already-collected sections.
        key: The section key to write to (``None`` means pre-heading text).
        buffer: Lines accumulated since the last heading.
    """
    content = "\n".join(buffer).strip()
    if content:
        sections[key] = content  # type: ignore[index]
