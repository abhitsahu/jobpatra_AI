"""JSON document parser.

Converts JSON bytes into a single plain-text string using LangChain's JSONLoader.
This module has ONE responsibility: extract text from JSON bytes.
"""

import os
import tempfile
from langchain_community.document_loaders import JSONLoader

from app.core.errors import UnparsableDocumentError


def parse(json_bytes: bytes) -> str:
    """Extract plain text from JSON bytes using LangChain's JSONLoader.

    Args:
        json_bytes: Raw bytes of a JSON file.

    Returns:
        A single plain-text string with the full content of the JSON.

    Raises:
        UnparsableDocumentError: If the document cannot be parsed or contains no text.
    """
    temp_dir = os.path.join(os.getcwd(), ".temp_json_parser")
    os.makedirs(temp_dir, exist_ok=True)

    temp_file = tempfile.NamedTemporaryFile(dir=temp_dir, suffix=".json", delete=False)
    temp_path = temp_file.name

    try:
        temp_file.write(json_bytes)
        temp_file.close()

        try:
            loader = JSONLoader(temp_path, jq_schema=".", text_content=False)
            documents = loader.load()
        except Exception as exc:
            raise UnparsableDocumentError(
                f"Could not open/parse JSON: {exc}"
            ) from exc

        pages: list[str] = []
        for doc in documents:
            stripped = doc.page_content.strip()
            if stripped:
                pages.append(stripped)

        if not pages:
            raise UnparsableDocumentError("JSON contains no extractable text.")

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
