"""Unit tests for section_splitter.

All tests are deterministic. No network. No AI. No file I/O beyond reading
the shared fixture once at module level.
"""

import pathlib

from app.analysis.normalization import section_splitter, text_cleaner
from app.analysis.normalization.section_splitter import ResumeSection

FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"
_RAW = (FIXTURES / "sample_resume.txt").read_text(encoding="utf-8")
# Always clean before splitting — mirrors production pipeline order
CLEANED = text_cleaner.clean(_RAW)


class TestSectionDetection:
    """Verify that standard resume sections are correctly detected."""

    def test_experience_detected(self) -> None:
        result = section_splitter.split(CLEANED)
        assert result.experience is not None

    def test_education_detected(self) -> None:
        result = section_splitter.split(CLEANED)
        assert result.education is not None

    def test_skills_detected(self) -> None:
        result = section_splitter.split(CLEANED)
        assert result.skills is not None

    def test_projects_detected(self) -> None:
        result = section_splitter.split(CLEANED)
        assert result.projects is not None

    def test_certifications_detected(self) -> None:
        result = section_splitter.split(CLEANED)
        assert result.certifications is not None

    def test_languages_detected(self) -> None:
        result = section_splitter.split(CLEANED)
        assert result.languages is not None


class TestMissingSections:
    """Missing sections must return None — never raise."""

    def test_summary_absent_returns_none(self) -> None:
        """The sample fixture has no Summary heading."""
        result = section_splitter.split(CLEANED)
        assert result.summary is None

    def test_missing_section_does_not_raise(self) -> None:
        minimal = text_cleaner.clean("Experience\n\nWorked at Acme.")
        result = section_splitter.split(minimal)
        assert result.summary is None
        assert result.education is None
        assert result.skills is None

    def test_no_headings_at_all_does_not_raise(self) -> None:
        result = section_splitter.split("Just some plain text with no headings.")
        assert isinstance(result, ResumeSection)
        assert result.summary is None
        assert result.experience is None

    def test_empty_string_does_not_raise(self) -> None:
        result = section_splitter.split("")
        assert isinstance(result, ResumeSection)

    def test_missing_section_is_none_not_empty_string(self) -> None:
        result = section_splitter.split(CLEANED)
        # summary is absent; must be None, not ""
        assert result.summary is None


class TestSectionContent:
    """Section bodies should contain expected content from the fixture."""

    def test_skills_content(self) -> None:
        result = section_splitter.split(CLEANED)
        assert result.skills is not None
        assert "React" in result.skills or "Node" in result.skills

    def test_experience_content(self) -> None:
        result = section_splitter.split(CLEANED)
        assert result.experience is not None
        assert "Acme" in result.experience or "Developer" in result.experience

    def test_education_content(self) -> None:
        result = section_splitter.split(CLEANED)
        assert result.education is not None
        assert "University" in result.education or "Computer" in result.education


class TestSectionSplitterIsolation:
    """Section splitter must not perform extraction or analysis."""

    def test_returns_resume_section_dataclass(self) -> None:
        result = section_splitter.split(CLEANED)
        assert isinstance(result, ResumeSection)

    def test_section_content_is_plain_text(self) -> None:
        """Section content should be str, not a list or parsed structure."""
        result = section_splitter.split(CLEANED)
        if result.experience is not None:
            assert isinstance(result.experience, str)
        if result.skills is not None:
            assert isinstance(result.skills, str)

    def test_alias_headings_recognized(self) -> None:
        """Alternative heading spellings should map to canonical sections."""
        text = text_cleaner.clean(
            "Professional Experience\n\nWorked at Acme.\n\nTechnical Skills\n\nPython Docker"
        )
        result = section_splitter.split(text)
        assert result.experience is not None
        assert result.skills is not None
