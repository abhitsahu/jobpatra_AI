"""Unit tests for the PDF parser.

Tests use only pre-generated fixture files — no network, no AI,
no external services. All outcomes are deterministic.
"""

import pathlib

import pytest

from app.analysis.parsers import pdf_parser
from app.core.errors import UnparsableDocumentError

FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"


def _read(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


class TestPdfParserValidDocument:
    """PDF with a selectable text layer."""

    def test_returns_non_empty_string(self) -> None:
        text = pdf_parser.parse(_read("sample_resume.pdf"))
        assert isinstance(text, str)
        assert len(text.strip()) > 0

    def test_contains_expected_name(self) -> None:
        text = pdf_parser.parse(_read("sample_resume.pdf"))
        assert "John Doe" in text

    def test_contains_expected_skills(self) -> None:
        text = pdf_parser.parse(_read("sample_resume.pdf"))
        assert "Python" in text

    def test_returns_str_not_bytes(self) -> None:
        result = pdf_parser.parse(_read("sample_resume.pdf"))
        assert type(result) is str


class TestPdfParserScannedDocument:
    """Image-only PDF with no text layer must raise, never return empty string."""

    def test_raises_unparsable_document_error(self) -> None:
        with pytest.raises(UnparsableDocumentError):
            pdf_parser.parse(_read("scanned_resume.pdf"))

    def test_error_message_is_descriptive(self) -> None:
        with pytest.raises(UnparsableDocumentError) as exc_info:
            pdf_parser.parse(_read("scanned_resume.pdf"))
        assert exc_info.value.message  # non-empty message

    def test_never_returns_empty_string(self) -> None:
        """Confirm the parser does not silently return '' for image PDFs."""
        try:
            result = pdf_parser.parse(_read("scanned_resume.pdf"))
            # If no exception was raised the result must not be empty
            assert result.strip(), "Parser returned empty string instead of raising"
        except UnparsableDocumentError:
            pass  # correct behaviour


class TestPdfParserInvalidInput:
    """Feed the parser corrupt bytes — it must raise, not crash."""

    def test_corrupt_bytes_raise_unparsable(self) -> None:
        with pytest.raises(UnparsableDocumentError):
            pdf_parser.parse(b"this is not a pdf")
