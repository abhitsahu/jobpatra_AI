"""TXT document parser.

Converts TXT bytes into a single plain-text string using LangChain's TextLoader.
This module has ONE responsibility: extract text from TXT bytes.

It does NOT normalize, score, analyse, or classify the extracted text.
"""

import os
import tempfile
from langchain_community.document_loaders import TextLoader

from app.core.errors import UnparsableDocumentError


def parse(txt_bytes: bytes) -> str:
    """Extract plain text from TXT bytes using LangChain's TextLoader.

    Args:
        txt_bytes: Raw bytes of a TXT file.

    Returns:
        A single plain-text string with the full content of the TXT.

    Raises:
        UnparsableDocumentError: If the document cannot be parsed or contains no text.
    """
    temp_dir = os.path.join(os.getcwd(), ".temp_txt_parser")
    os.makedirs(temp_dir, exist_ok=True)

    temp_file = tempfile.NamedTemporaryFile(dir=temp_dir, suffix=".txt", delete=False)
    temp_path = temp_file.name

    try:
        temp_file.write(txt_bytes)
        temp_file.close()

        try:
            # TextLoader accepts an optional encoding. If it fails, try other common encodings.
            loader = TextLoader(temp_path, encoding="utf-8")
            documents = loader.load()
        except Exception:
            try:
                loader = TextLoader(temp_path, encoding="latin-1")
                documents = loader.load()
            except Exception as exc:
                raise UnparsableDocumentError(
                    f"Could not open/parse TXT: {exc}"
                ) from exc

        pages: list[str] = []
        for doc in documents:
            stripped = doc.page_content.strip()
            if stripped:
                pages.append(stripped)

        if not pages:
            raise UnparsableDocumentError("TXT contains no extractable text.")

        return "\n\n".join(pages)

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        try:
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except OSError:
            pass
