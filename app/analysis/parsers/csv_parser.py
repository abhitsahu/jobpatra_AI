"""CSV document parser.

Converts CSV bytes into a single plain-text string using LangChain's CSVLoader.
This module has ONE responsibility: extract text from CSV bytes.
"""

import os
import tempfile
from langchain_community.document_loaders import CSVLoader

from app.core.errors import UnparsableDocumentError


def parse(csv_bytes: bytes) -> str:
    """Extract plain text from CSV bytes using LangChain's CSVLoader.

    Args:
        csv_bytes: Raw bytes of a CSV file.

    Returns:
        A single plain-text string with the full content of the CSV.

    Raises:
        UnparsableDocumentError: If the document cannot be parsed or contains no text.
    """
    temp_dir = os.path.join(os.getcwd(), ".temp_csv_parser")
    os.makedirs(temp_dir, exist_ok=True)

    temp_file = tempfile.NamedTemporaryFile(dir=temp_dir, suffix=".csv", delete=False)
    temp_path = temp_file.name

    try:
        temp_file.write(csv_bytes)
        temp_file.close()

        try:
            loader = CSVLoader(temp_path)
            documents = loader.load()
        except Exception as exc:
            raise UnparsableDocumentError(
                f"Could not open/parse CSV: {exc}"
            ) from exc

        pages: list[str] = []
        for doc in documents:
            stripped = doc.page_content.strip()
            if stripped:
                pages.append(stripped)

        if not pages:
            raise UnparsableDocumentError("CSV contains no extractable text.")

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
