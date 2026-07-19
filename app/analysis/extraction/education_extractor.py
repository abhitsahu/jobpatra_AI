"""Education extractor — parse structured education history from text.

This module has ONE responsibility: given the raw text of a resume's
Education section (as produced by section_splitter), extract structured
education and certification entries.
"""

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class EducationEntry:
    """One education entry from a resume."""

    degree: str | None = None
    """Degree level, e.g. ``'B.Sc'``, ``'Master of Science'``."""
    field: str | None = None
    """Field of study, e.g. ``'Computer Science'``."""
    institution: str | None = None
    """University or college name."""
    graduation_year: str | None = None
    """Graduation year as a string, e.g. ``'2021'``."""
    cgpa: str | None = None
    """Extracted GPA or CGPA score, e.g. ``'3.8'`` or ``'9.2'``."""


@dataclass
class EducationExtractionResult:
    """Result of education extraction on the education section."""

    entries: list[EducationEntry] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    """Any certification names detected (standalone lines not matching a degree)."""


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# Degree keyword pattern
_DEGREE_RE = re.compile(
    r"""
    \b(
        # Abbreviated forms
        B\.?Sc|B\.?Tech|B\.?E|B\.?A|B\.?Com|B\.?S|
        M\.?Sc|M\.?Tech|M\.?E|M\.?A|M\.?S|M\.?Com|MBA|MCA|BCA|
        Ph\.?D|DPhil|
        # Full forms
        Bachelor(?:'s)?(?:\s+of)?|
        Master(?:'s)?(?:\s+of)?|
        Doctor(?:ate)?(?:\s+of)?|
        Associate(?:'s)?(?:\s+of)?|
        # Diploma / certificate forms
        Diploma|Postgraduate\s+Diploma|PG\s+Diploma|
        High\s+School|Secondary\s+School
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Institution keywords
_INSTITUTION_RE = re.compile(
    r"\b(university|college|institute|school|academy|polytechnic|"
    r"iit|nit|bits|mit|stanford|oxford|cambridge)\b",
    re.IGNORECASE,
)

# Certification patterns — lines that look like certifications but not degrees
_CERT_KEYWORDS_RE = re.compile(
    r"\b(certified|certification|certificate|aws|gcp|azure|pmp|"
    r"cpa|cfa|cissp|ccna|ccnp|google|microsoft|oracle|comptia|"
    r"scrum|agile|itil|prince2)\b",
    re.IGNORECASE,
)

# Regular expression to extract GPA / CGPA / Grade percentages
_CGPA_RE = re.compile(
    r'\b(?:gpa|cgpa|grade|score|percentage|pct)?\s*:?\s*(\b[0-9]\.[0-9]{1,2}(?:\s*/\s*(?:4|5|10))?\b|\b10\.0\b|\b[789]\.[0-9]\b|\b[89]\d%\b)\b',
    re.IGNORECASE
)

_BLANK_LINE_RE = re.compile(r"^\s*$")

# Separator between degree and field: "in", "of", dash, comma
_DEGREE_FIELD_SEP_RE = re.compile(
    r"(?:,\s*|\s+in\s+|\s+of\s+|-\s*|\s+-\s*)", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract(education_text: str) -> EducationExtractionResult:
    """Parse education entries and certifications from the education section."""
    if not education_text or not education_text.strip():
        return EducationExtractionResult()

    blocks = _split_into_blocks(education_text)
    entries: list[EducationEntry] = []
    certifications: list[str] = []

    for block in blocks:
        result = _parse_block(block)
        if isinstance(result, EducationEntry):
            entries.append(result)
        elif isinstance(result, str):
            certifications.append(result)

    return EducationExtractionResult(entries=entries, certifications=certifications)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _split_into_blocks(text: str) -> list[list[str]]:
    """Split text on blank lines into non-blank line groups."""
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if _BLANK_LINE_RE.match(line):
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def _parse_block(lines: list[str]) -> EducationEntry | str | None:
    """Parse a block of lines as either an education entry, a certification, or None."""
    combined = " ".join(line.strip() for line in lines)
    degree_match = _DEGREE_RE.search(combined)

    if degree_match:
        return _build_education_entry(lines, combined, degree_match)

    # Check for certification
    if _CERT_KEYWORDS_RE.search(combined):
        return combined.strip()

    return None


def _build_education_entry(
    lines: list[str],
    combined: str,
    degree_match: re.Match[str],
) -> EducationEntry:
    """Build an EducationEntry from matched lines."""
    entry = EducationEntry()

    # Degree: the matched keyword
    entry.degree = degree_match.group(0).strip()

    # Field of study: text after the degree keyword on the same line
    degree_line = next(
        (l.strip() for l in lines if _DEGREE_RE.search(l)), combined
    )
    after_degree = degree_line[degree_match.start() + len(entry.degree):].strip()
    after_degree = _DEGREE_FIELD_SEP_RE.sub("", after_degree, count=1).strip()
    field_text = _YEAR_RE.sub("", after_degree).strip(" ,.-")
    field_text = _INSTITUTION_RE.sub("", field_text).strip(" ,.-")
    entry.field = field_text if field_text else None

    # Institution: line containing a university/college keyword
    for line in lines:
        if _INSTITUTION_RE.search(line):
            inst = _YEAR_RE.sub("", line).strip(" ,.-")
            # Filter CGPA remnants if they are present in institution line
            inst = _CGPA_RE.sub("", inst).strip(" ,.-")
            entry.institution = inst if inst else None
            break

    # Graduation year: first year found across all lines
    year_match = _YEAR_RE.search(combined)
    if year_match:
        entry.graduation_year = year_match.group()

    # CGPA/GPA extraction
    cgpa_match = _CGPA_RE.search(combined)
    if cgpa_match:
        entry.cgpa = cgpa_match.group(1).strip()

    return entry
