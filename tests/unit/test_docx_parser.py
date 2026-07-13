"""Unit tests for the DOCX parser.

Tests use only pre-generated fixture files — no network, no AI,
no external services. All outcomes are deterministic.
"""

import pathlib

import pytest

from app.analysis.parsers import docx_parser
from app.core.errors import UnparsableDocumentError

FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"


def _read(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


class TestDocxParserValidDocument:
    """DOCX with readable paragraph content."""

    def test_returns_non_empty_string(self) -> None:
        text = docx_parser.parse(_read("sample_resume.docx"))
        assert isinstance(text, str)
        assert len(text.strip()) > 0

    def test_contains_expected_name(self) -> None:
        text = docx_parser.parse(_read("sample_resume.docx"))
        assert "Jane Smith" in text

    def test_contains_expected_skills(self) -> None:
        text = docx_parser.parse(_read("sample_resume.docx"))
        assert "SQL" in text

    def test_preserves_paragraph_order(self) -> None:
        """Name should appear before skills in the extracted text."""
        text = docx_parser.parse(_read("sample_resume.docx"))
        assert text.index("Jane Smith") < text.index("SQL")

    def test_returns_str_not_bytes(self) -> None:
        result = docx_parser.parse(_read("sample_resume.docx"))
        assert type(result) is str


class TestDocxParserInvalidInput:
    """Feed the parser corrupt bytes — it must raise, not crash."""

    def test_corrupt_bytes_raise_unparsable(self) -> None:
        with pytest.raises(UnparsableDocumentError):
            docx_parser.parse(b"this is definitely not a docx file")

    def test_pdf_bytes_raise_unparsable(self) -> None:
        """A PDF fed to the DOCX parser must raise, not silently return garbage."""
        pdf_bytes = _read("sample_resume.pdf")
        with pytest.raises(UnparsableDocumentError):
            docx_parser.parse(pdf_bytes)
