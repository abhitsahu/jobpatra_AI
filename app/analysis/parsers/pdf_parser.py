"""PDF document parser.

Converts PDF bytes into a single plain-text string using PyMuPDF.
This module has ONE responsibility: extract text from PDF bytes.

It does NOT normalize, score, analyse, or classify the extracted text.
"""

import fitz  # PyMuPDF

from app.core.errors import UnparsableDocumentError


def parse(pdf_bytes: bytes) -> str:
    """Extract plain text from PDF bytes.

    Iterates over every page in the document, extracts the text block
    from each page, and joins them with a single blank line separator so
    that page boundaries are visible without being noisy.

    Args:
        pdf_bytes: Raw bytes of a PDF file.

    Returns:
        A single plain-text string with the full content of the PDF.

    Raises:
        UnparsableDocumentError: If the PDF contains no selectable text
            (e.g. a scanned image-only document).
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise UnparsableDocumentError(
            f"Could not open PDF: {exc}"
        ) from exc

    pages: list[str] = []
    for page in doc:
        text = page.get_text("text")
        stripped = text.strip()
        if stripped:
            pages.append(stripped)

    doc.close()

    if not pages:
        raise UnparsableDocumentError(
            "PDF contains no extractable text. "
            "The document may be a scanned image without a text layer."
        )

    return "\n\n".join(pages)
