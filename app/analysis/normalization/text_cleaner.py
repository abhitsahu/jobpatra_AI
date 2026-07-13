"""Text cleaner — normalize messy plain text into consistently formatted output.

This module has ONE responsibility: take raw text (typically the output of a
document parser) and return clean, consistently structured plain text.

It does NOT:
  - identify skills, education, experience, or any resume sections
  - calculate ATS scores
  - split text into sections
  - call any external service or AI

All functions are pure: same input always produces the same output.
No I/O. No file access. No FastAPI imports.
"""

import re
import unicodedata


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clean(text: str) -> str:
    """Convert messy extracted text into consistently formatted plain text.

    Applies a deterministic sequence of cleaning steps:
      1. Decode/normalize Unicode to remove mojibake and special chars
      2. Remove control characters (keeping newlines and tabs)
      3. Normalize line endings to ``\\n``
      4. Collapse multiple spaces on each line to a single space
      5. Strip leading/trailing whitespace from every line
      6. Collapse three or more consecutive blank lines to two
      7. Strip the entire result

    Args:
        text: Raw text as returned by a document parser.

    Returns:
        Clean plain text, ready for section splitting or JD analysis.
    """
    text = _normalize_unicode(text)
    text = _remove_control_characters(text)
    text = _normalize_line_endings(text)
    text = _collapse_spaces_per_line(text)
    text = _strip_lines(text)
    text = _collapse_blank_lines(text)
    return text.strip()


# ---------------------------------------------------------------------------
# Private helpers — each does exactly one thing
# ---------------------------------------------------------------------------


def _normalize_unicode(text: str) -> str:
    """NFC-normalize unicode and replace common non-ASCII characters.

    Converts the text to NFC form so that composed characters (e.g. é
    stored as e + combining accent) are unified.  PDF extraction often
    produces ligatures and special punctuation that should map to their
    plain ASCII equivalents.
    """
    text = unicodedata.normalize("NFC", text)

    # Typographic quotes → straight quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    # Em-dash / en-dash → hyphen
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    # Bullet variants → plain hyphen-bullet
    text = text.replace("\u2022", "-").replace("\u2023", "-")

    # Non-breaking space → regular space
    text = text.replace("\u00a0", " ")

    return text


def _remove_control_characters(text: str) -> str:
    """Strip ASCII control characters (0x00–0x1F, 0x7F) except ``\\n`` and ``\\t``.

    PDF extraction can embed form-feed (0x0C), carriage-return (0x0D),
    null bytes, and other control chars that should not appear in output.
    """
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def _normalize_line_endings(text: str) -> str:
    """Convert ``\\r\\n`` and bare ``\\r`` to ``\\n``."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _collapse_spaces_per_line(text: str) -> str:
    """Replace runs of horizontal whitespace on each line with a single space.

    Tabs are treated as spaces. This does not touch blank lines.
    """
    lines = text.split("\n")
    return "\n".join(re.sub(r"[ \t]+", " ", line) for line in lines)


def _strip_lines(text: str) -> str:
    """Strip leading and trailing whitespace from every individual line."""
    return "\n".join(line.strip() for line in text.split("\n"))


def _collapse_blank_lines(text: str) -> str:
    """Reduce any run of three or more consecutive blank lines to exactly two.

    Two consecutive blank lines (one empty line between paragraphs) are
    kept as-is because they represent genuine paragraph separation in many
    resumes.  Three or more are almost always extraction noise.
    """
    return re.sub(r"\n{3,}", "\n\n", text)
