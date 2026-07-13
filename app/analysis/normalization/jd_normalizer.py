"""Job description normalizer.

This module has ONE responsibility: clean job description text so it is
consistently formatted before downstream analysis.

It handles:
  - HTML tag removal (job descriptions frequently come from web scrapers
    or copy-paste from online portals with residual markup)
  - Whitespace normalization
  - Line-ending standardization

It does NOT:
  - split the JD into sections
  - extract requirements or keywords
  - score or judge the job description
  - call any external service or AI

All functions are pure: same input always produces the same output.
No I/O. No file access. No FastAPI imports.
"""

import re

from app.analysis.normalization import text_cleaner


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize(jd_text: str) -> str:
    """Clean a job description string.

    Applies HTML stripping first (job descriptions often arrive with
    residual markup), then delegates to the shared text cleaner for
    whitespace and unicode normalization.

    Args:
        jd_text: Raw job description text, which may contain HTML tags,
            multiple blank lines, or irregular spacing.

    Returns:
        Clean plain text suitable for downstream keyword extraction or
        matching phases.
    """
    text = _strip_html(jd_text)
    return text_cleaner.clean(text)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode common HTML entities.

    Handles both well-formed and malformed tags (e.g. unclosed, nested).
    After tag removal, common entities are decoded so that ``&amp;`` →
    ``&``, ``&lt;`` → ``<``, etc.

    Args:
        text: Input string that may contain HTML markup.

    Returns:
        Plain text with all HTML tags removed.
    """
    # Remove HTML tags
    no_tags = re.sub(r"<[^>]+>", " ", text)

    # Decode the most common HTML entities
    entities: dict[str, str] = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
        "&nbsp;": " ",
        "&ndash;": "-",
        "&mdash;": "-",
        "&bull;": "-",
    }
    for entity, replacement in entities.items():
        no_tags = no_tags.replace(entity, replacement)

    return no_tags
