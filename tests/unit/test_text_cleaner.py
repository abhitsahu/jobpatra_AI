"""Unit tests for text_cleaner.

All tests are deterministic. No network. No AI. No file I/O beyond reading
the shared fixture once per class.
"""

import pathlib

from app.analysis.normalization import text_cleaner

FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"
SAMPLE = (FIXTURES / "sample_resume.txt").read_text(encoding="utf-8")


class TestCleanExtraSpaces:
    """Multiple consecutive horizontal spaces should be collapsed to one."""

    def test_multiple_spaces_collapsed(self) -> None:
        result = text_cleaner.clean("John    Doe")
        assert "  " not in result

    def test_tabs_collapsed(self) -> None:
        result = text_cleaner.clean("Name:\tJohn\tDoe")
        assert "\t" not in result

    def test_mixed_spaces_and_tabs(self) -> None:
        result = text_cleaner.clean("Skills  \t  React")
        assert "  " not in result
        assert "\t" not in result

    def test_leading_trailing_space_on_lines_removed(self) -> None:
        result = text_cleaner.clean("  John Doe  \n  Engineer  ")
        for line in result.splitlines():
            assert line == line.strip()

    def test_fixture_has_no_double_spaces(self) -> None:
        result = text_cleaner.clean(SAMPLE)
        assert "  " not in result


class TestBlankLineCollapse:
    """Three or more consecutive blank lines should be reduced to two."""

    def test_three_blank_lines_collapsed(self) -> None:
        result = text_cleaner.clean("A\n\n\n\nB")
        assert "\n\n\n" not in result

    def test_ten_blank_lines_collapsed(self) -> None:
        result = text_cleaner.clean("A" + "\n" * 10 + "B")
        assert "\n\n\n" not in result

    def test_single_blank_line_preserved(self) -> None:
        result = text_cleaner.clean("A\n\nB")
        assert "\n\n" in result

    def test_fixture_has_no_triple_blank_lines(self) -> None:
        result = text_cleaner.clean(SAMPLE)
        assert "\n\n\n" not in result


class TestControlCharacterRemoval:
    """Non-printable control characters except newline should be removed."""

    def test_null_byte_removed(self) -> None:
        result = text_cleaner.clean("John\x00Doe")
        assert "\x00" not in result

    def test_form_feed_removed(self) -> None:
        result = text_cleaner.clean("Page 1\x0cPage 2")
        assert "\x0c" not in result

    def test_carriage_return_normalized(self) -> None:
        result = text_cleaner.clean("Line 1\r\nLine 2\rLine 3")
        assert "\r" not in result
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result


class TestOutputConsistency:
    """The overall shape of the cleaned output must be consistently formatted."""

    def test_result_is_str(self) -> None:
        assert isinstance(text_cleaner.clean(SAMPLE), str)

    def test_result_not_empty(self) -> None:
        assert text_cleaner.clean(SAMPLE).strip()

    def test_leading_trailing_whitespace_removed(self) -> None:
        result = text_cleaner.clean("\n\n  hello  \n\n")
        assert result == result.strip()

    def test_idempotent(self) -> None:
        """Cleaning already-clean text must not change it further."""
        once = text_cleaner.clean(SAMPLE)
        twice = text_cleaner.clean(once)
        assert once == twice

    def test_unicode_normalization(self) -> None:
        result = text_cleaner.clean("\u201cquoted\u201d and \u2018single\u2019")
        assert "\u201c" not in result
        assert "\u201d" not in result
        assert '"quoted"' in result

    def test_em_dash_replaced(self) -> None:
        result = text_cleaner.clean("2021\u20142024")
        assert "\u2014" not in result
        assert "2021-2024" in result

    def test_fixture_name_present(self) -> None:
        result = text_cleaner.clean(SAMPLE)
        assert "JOHN DOE" in result or "John Doe" in result or "JOHN" in result
