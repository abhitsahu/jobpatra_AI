"""Parser factory — the single public entry point for document parsing.

No other module should import pdf_parser or docx_parser directly.
All callers go through parse() in this module.

Supported inputs:
    - PDF  (.pdf)
    - DOCX (.docx)
    - Raw text (no filename / no bytes)
"""

from app.analysis.parsers import docx_parser, pdf_parser
from app.core.errors import ValidationError


# Map of lowercase file extensions to parser modules.
_EXTENSION_MAP = {
    ".pdf": pdf_parser,
    ".docx": docx_parser,
}


def parse(filename: str, file_bytes: bytes) -> str:
    """Convert an uploaded document to plain text.

    Selects the correct parser based on the file extension, delegates
    the actual extraction, and returns the resulting plain text.

    Args:
        filename: Original filename including extension (e.g. "resume.pdf").
            Used solely to determine which parser to dispatch to.
        file_bytes: Raw bytes of the uploaded file.

    Returns:
        Plain text extracted from the document.

    Raises:
        ValidationError: If the file extension is not supported.
        UnparsableDocumentError: Propagated from the underlying parser
            when the document contains no extractable text.
    """
    extension = _get_extension(filename)
    parser = _EXTENSION_MAP.get(extension)

    if parser is None:
        supported = ", ".join(_EXTENSION_MAP.keys())
        raise ValidationError(
            f"Unsupported file type '{extension}'. Supported types: {supported}."
        )

    return parser.parse(file_bytes)


def parse_text(text: str) -> str:
    """Pass raw text through unchanged.

    Allows callers to treat plain-text input uniformly with file input —
    both paths return a plain-text string.

    Args:
        text: Raw resume or job-description text pasted by the user.

    Returns:
        The same text, stripped of leading/trailing whitespace.

    Raises:
        ValidationError: If the supplied text is empty.
    """
    stripped = text.strip()
    if not stripped:
        raise ValidationError("Text input must not be empty.")
    return stripped


def _get_extension(filename: str) -> str:
    """Return the lowercase extension including the leading dot.

    Args:
        filename: A filename string such as "my_resume.PDF".

    Returns:
        Lowercase extension, e.g. ".pdf".
    """
    dot_index = filename.rfind(".")
    if dot_index == -1:
        return ""
    return filename[dot_index:].lower()
