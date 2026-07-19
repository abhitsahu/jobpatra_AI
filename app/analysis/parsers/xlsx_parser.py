"""Excel document parser.

Converts Excel (.xlsx) bytes into a single plain-text string using LangChain's UnstructuredExcelLoader.
This module has ONE responsibility: extract text from Excel bytes.
"""

import os
import tempfile
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

from app.core.errors import UnparsableDocumentError


def parse(excel_bytes: bytes) -> str:
    """Extract plain text from Excel bytes using LangChain's UnstructuredExcelLoader.

    Args:
        excel_bytes: Raw bytes of an Excel file.

    Returns:
        A single plain-text string with the full content of the Excel sheet.

    Raises:
        UnparsableDocumentError: If the document cannot be parsed or contains no text.
    """
    temp_dir = os.path.join(os.getcwd(), ".temp_excel_parser")
    os.makedirs(temp_dir, exist_ok=True)

    temp_file = tempfile.NamedTemporaryFile(dir=temp_dir, suffix=".xlsx", delete=False)
    temp_path = temp_file.name

    try:
        temp_file.write(excel_bytes)
        temp_file.close()

        try:
            loader = UnstructuredExcelLoader(temp_path)
            documents = loader.load()
        except Exception as exc:
            raise UnparsableDocumentError(
                f"Could not open/parse Excel: {exc}"
            ) from exc

        pages: list[str] = []
        for doc in documents:
            stripped = doc.page_content.strip()
            if stripped:
                pages.append(stripped)

        if not pages:
            raise UnparsableDocumentError("Excel contains no extractable text.")

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
