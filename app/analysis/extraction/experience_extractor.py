"""Experience extractor — parse structured work history from the experience section.

This module has ONE responsibility: given the raw text of a resume's
Experience section (as produced by section_splitter), extract structured
work experience entries.

It does NOT:
  - score experience relevance
  - compare against a job description
  - call AI or external services
  - import from FastAPI

Algorithm:
  A new experience entry is identified when a line contains a date range AND
  looks like a header (not a bullet).  Subsequent non-blank lines belonging to
  that entry are collected as bullet points.  A new header line triggers the
  flush of the current entry.

Limitations (by design):
  - Heuristic parsing.  Unusual formatting may not be fully parsed.
  - Duration is computed only when both start and end dates are parseable year
    values.
  - Missing fields are None, never exceptions.

All functions are pure. No I/O. No state. No FastAPI imports.
"""

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ExperienceEntry:
    """One work experience entry from a resume."""

    title: str | None = None
    """Job title, e.g. ``'Senior Software Engineer'``."""
    company: str | None = None
    """Employer name, e.g. ``'Acme Corp'``."""
    start_date: str | None = None
    """Start date string as found in text, e.g. ``'2021'``."""
    end_date: str | None = None
    """End date string as found in text, e.g. ``'2024'`` or ``'Present'``."""
    duration_years: float | None = None
    """Computed duration in years when both dates are numeric years."""
    bullets: list[str] = field(default_factory=list)
    """Descriptive lines from the entry body."""
    metrics: list[str] = field(default_factory=list)
    """Measurable values found in body lines, e.g. ``['40%', '$200K']``."""


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

_DATE_RANGE_RE = re.compile(
    r"((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?"
    r"|Dec(?:ember)?)\.?\s*)?"
    r"((?:19|20)\d{2})"
    r"\s*[-–—to]+\s*"
    r"((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?"
    r"|Dec(?:ember)?)\.?\s*)?"
    r"((?:19|20)\d{2}|[Pp]resent|[Cc]urrent)",
    re.IGNORECASE,
)

# Separator between title and company: "at", "@", "-", "–", "—", ","
_AT_SEP_RE = re.compile(r"\s+(?:at|@|-|–|—|,)\s+", re.IGNORECASE)

# Bullet line prefix
_BULLET_RE = re.compile(r"^\s*[-*•·▪▸>\d.]+\s+")

# Quantifiable metrics
_METRIC_RE = re.compile(
    r"\b\d+(?:\.\d+)?%"                              # 40%, 3.5%
    r"|\b\d+(?:\.\d+)?[xX]\b"                        # 5x, 2.5X
    r"|\$\s*\d[\d,]*(?:\.\d+)?[KMBkmb]?"             # $200K, $1.2M, $500
    r"|\b\d[\d,]+\+"                                  # 100+, 1,000+
    r"|\b\d+\s*(?:million|billion|thousand)\b",       # 5 million
    re.IGNORECASE,
)

_BLANK_LINE_RE = re.compile(r"^\s*$")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract(experience_text: str) -> list[ExperienceEntry]:
    """Parse work experience entries from the experience section text.

    Uses a line-by-line state machine: a line that contains a date range
    and is not a bullet is treated as a new entry header.  All following
    lines (including those separated by blank lines within the same entry)
    are collected as body lines until the next header is encountered.

    Args:
        experience_text: Text content of the Experience section, as returned
            by ``section_splitter.split().experience``.

    Returns:
        List of ``ExperienceEntry`` objects.  Returns an empty list if the
        text is empty.  Never raises.
    """
    if not experience_text or not experience_text.strip():
        return []

    entries: list[ExperienceEntry] = []
    current: ExperienceEntry | None = None
    body_lines: list[str] = []

    for line in experience_text.splitlines():
        stripped = line.strip()

        # Blank line — keep collecting for the current entry
        if not stripped:
            continue

        is_bullet = bool(_BULLET_RE.match(line))
        has_date = bool(_DATE_RANGE_RE.search(stripped))

        # A non-bullet line with a date range starts a new entry
        if has_date and not is_bullet:
            if current is not None:
                _finalise_entry(current, body_lines)
                entries.append(current)
            current = _parse_header(stripped)
            body_lines = []
        else:
            # Accumulate as a body line for the current entry
            body_lines.append(stripped)

    # Flush the last entry
    if current is not None:
        _finalise_entry(current, body_lines)
        entries.append(current)
    elif body_lines:
        # Text with no date headers at all — treat the whole thing as one entry
        entry = ExperienceEntry()
        _finalise_entry(entry, body_lines)
        if entry.bullets:
            entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_header(line: str) -> ExperienceEntry:
    """Extract title, company, and dates from a header line.

    Args:
        line: A stripped header line containing a date range.

    Returns:
        Partially populated ``ExperienceEntry`` (no bullets/metrics yet).
    """
    entry = ExperienceEntry()

    date_match = _DATE_RANGE_RE.search(line)
    if date_match:
        entry.start_date = date_match.group(2)
        end_raw = date_match.group(4)
        entry.end_date = "Present" if end_raw.lower() in ("present", "current") else end_raw
        entry.duration_years = _compute_duration(entry.start_date, entry.end_date)
        # Remove date range from the header text to isolate title/company
        header_text = _DATE_RANGE_RE.sub("", line).strip(" ,.-–—")
    else:
        header_text = line

    parts = _AT_SEP_RE.split(header_text, maxsplit=1)
    entry.title = parts[0].strip() or None
    entry.company = parts[1].strip() if len(parts) > 1 else None

    return entry


def _finalise_entry(entry: ExperienceEntry, body_lines: list[str]) -> None:
    """Populate bullets and metrics on ``entry`` from ``body_lines``.

    Modifies ``entry`` in place.

    Args:
        entry: The ExperienceEntry being built.
        body_lines: Lines collected after the header.
    """
    for raw in body_lines:
        clean = _BULLET_RE.sub("", raw).strip()
        if clean:
            entry.bullets.append(clean)
            found = _METRIC_RE.findall(raw)
            entry.metrics.extend(found)


def _compute_duration(start: str | None, end: str | None) -> float | None:
    """Compute duration in years between two year strings.

    Args:
        start: Start year string, e.g. ``'2021'``.
        end: End year string, e.g. ``'2024'`` or ``'Present'``.

    Returns:
        Float years, or ``None`` if either value is non-numeric.
    """
    if start is None or end is None:
        return None
    if end.lower() in ("present", "current"):
        import datetime
        end_year = datetime.datetime.now().year
    else:
        year_match = _YEAR_RE.search(end)
        if not year_match:
            return None
        end_year = int(year_match.group())

    start_match = _YEAR_RE.search(start)
    if not start_match:
        return None
    start_year = int(start_match.group())
    duration = end_year - start_year
    return float(duration) if duration >= 0 else None
