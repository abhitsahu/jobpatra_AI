"""DOCX document parser.

Converts DOCX bytes into a single plain-text string using python-docx.
This module has ONE responsibility: extract text from DOCX bytes.

It does NOT normalize, score, analyse, or classify the extracted text.
"""

import io

from docx import Document
from docx.opc.exceptions import PackageNotFoundError

from app.core.errors import UnparsableDocumentError


def parse(docx_bytes: bytes) -> str:
    """Extract plain text from DOCX bytes.

    Reads every paragraph in the document in order and joins non-empty
    paragraphs with newlines, preserving the visual paragraph structure
    without introducing spurious blank lines for empty paragraphs.

    Args:
        docx_bytes: Raw bytes of a DOCX file.

    Returns:
        A single plain-text string with the full content of the DOCX.

    Raises:
        UnparsableDocumentError: If the bytes cannot be opened as a valid
            DOCX document, or if the document contains no text.
    """
    try:
        doc = Document(io.BytesIO(docx_bytes))
    except (PackageNotFoundError, Exception) as exc:
        raise UnparsableDocumentError(
            f"Could not open DOCX: {exc}"
        ) from exc

    lines: list[str] = []
    for para in doc.paragraphs:
        stripped = para.text.strip()
        if stripped:
            lines.append(stripped)

    if not lines:
        raise UnparsableDocumentError(
            "DOCX contains no extractable text."
        )

    return "\n".join(lines)
