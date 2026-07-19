"""PDF document parser.

Converts PDF bytes into a single plain-text string using LangChain's DirectoryLoader and PyMuPDFLoader.
This module has ONE responsibility: extract text from PDF bytes.

It does NOT normalize, score, analyse, or classify the extracted text.
"""

import os
import uuid
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

from app.core.errors import UnparsableDocumentError


def parse(pdf_bytes: bytes) -> str:
    """Extract plain text from PDF bytes using LangChain's DirectoryLoader with PyMuPDFLoader.

    Args:
        pdf_bytes: Raw bytes of a PDF file.

    Returns:
        A single plain-text string with the full content of the PDF.

    Raises:
        UnparsableDocumentError: If the PDF contains no selectable text
            (e.g. a scanned image-only document) or is corrupt.
    """
    # Create a local temp directory inside the workspace root.
    # Cwd is guaranteed to be in the workspace root (/home/abhit/Abhit/jobpatra/Ai_backend)
    temp_dir = os.path.join(os.getcwd(), ".temp_pdf_parser")
    os.makedirs(temp_dir, exist_ok=True)

    # Generate a unique filename to prevent concurrency collisions
    filename = f"resume_{uuid.uuid4().hex}.pdf"
    temp_path = os.path.join(temp_dir, filename)

    try:
        with open(temp_path, "wb") as f:
            f.write(pdf_bytes)

        # Load using LangChain's DirectoryLoader with PyMuPDFLoader
        try:
            from app.core.logging import logger
            logger.info("========== PARSER ==========")
            logger.info("Selected Parser:")
            logger.info("PDFParser")
            logger.info("Loader:")
            logger.info("PyMuPDFLoader")
            logger.info("Creating PyMuPDFLoader...")
            logger.info("Calling loader.load()")

            # ────────────────────────────────────────────────────────────
            # Verification of written temporary file
            # ────────────────────────────────────────────────────────────
            import hashlib
            import binascii
            import fitz

            file_size = os.path.getsize(temp_path)
            
            # Compute SHA-256 of the written file
            sha256 = hashlib.sha256()
            with open(temp_path, "rb") as f:
                content = f.read()
                sha256.update(content)
            sha256_hex = sha256.hexdigest()
            
            first_32_hex = binascii.hexlify(content[:32]).decode('utf-8')
            first_32_formatted = " ".join(first_32_hex[i:i+2] for i in range(0, len(first_32_hex), 2))
            
            last_32_hex = binascii.hexlify(content[-32:]).decode('utf-8') if len(content) >= 32 else binascii.hexlify(content).decode('utf-8')
            last_32_formatted = " ".join(last_32_hex[i:i+2] for i in range(0, len(last_32_hex), 2))
            
            logger.info("Stage: Temporary PDF File written to disk")
            logger.info(f"absolute path: {os.path.abspath(temp_path)}")
            logger.info(f"file size: {file_size} bytes")
            logger.info(f"SHA-256: {sha256_hex}")
            logger.info(f"first 32 bytes: {first_32_formatted}")
            logger.info(f"last 32 bytes: {last_32_formatted}")

            # Verify fitz.open
            try:
                doc = fitz.open(temp_path)
                logger.info("fitz.open(temp_path) succeeded!")
                doc.close()
            except Exception as fitz_exc:
                logger.error(f"fitz.open(temp_path) failed: {fitz_exc}")

            loader = DirectoryLoader(
                temp_dir,
                glob=filename,
                loader_cls=PyMuPDFLoader,
                show_progress=False
            )
            documents = loader.load()
            logger.info("Documents returned:")
            logger.info(str(len(documents)))
        except Exception as exc:
            raise UnparsableDocumentError(
                f"Could not open/parse PDF: {exc}"
            ) from exc

        pages: list[str] = []
        for doc in documents:
            stripped = doc.page_content.strip()
            if stripped:
                pages.append(stripped)

        if not pages:
            raise UnparsableDocumentError(
                "PDF contains no extractable text. "
                "The document may be a scanned image without a text layer."
            )

        parsed_text = "\n\n".join(pages)
        logger.info("Preview")
        logger.info(parsed_text)
        return parsed_text

    finally:
        # Cleanup temporary file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        # Clean up temp directory if empty
        try:
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except OSError:
            pass
