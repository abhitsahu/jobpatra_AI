"""Unit tests for formatting_score.

Deterministic. No AI. No network.
"""

from app.analysis.normalization.section_splitter import ResumeSection
from app.analysis.scoring import formatting_score


def _full_sections() -> ResumeSection:
    """All sections present and non-empty."""
    return ResumeSection(
        summary="I am a developer.",
        experience="Software Engineer at XYZ 2020-2023",
        education="B.Sc Computer Science 2020",
        skills="Python Docker React",
        projects="Built a web app.",
        certifications="AWS Certified",
        languages="English Spanish",
    )


def _empty_sections() -> ResumeSection:
    """All sections absent (None)."""
    return ResumeSection()


class TestFullSections:
    def test_all_sections_present_gives_100(self) -> None:
        assert formatting_score.calculate(_full_sections()) == 100.0


class TestMissingSections:
    def test_no_sections_gives_zero(self) -> None:
        assert formatting_score.calculate(_empty_sections()) == 0.0

    def test_only_experience_gives_30(self) -> None:
        sections = ResumeSection(experience="Software Engineer at XYZ 2020-2023")
        assert formatting_score.calculate(sections) == 30.0

    def test_missing_experience_reduces_score(self) -> None:
        sections = _full_sections()
        sections.experience = None
        full = formatting_score.calculate(_full_sections())
        reduced = formatting_score.calculate(sections)
        assert reduced < full
        assert full - reduced == 30.0  # experience worth 30 pts

    def test_missing_education_reduces_score(self) -> None:
        sections = _full_sections()
        sections.education = None
        reduced = formatting_score.calculate(sections)
        assert reduced == 80.0  # 100 - 20

    def test_whitespace_only_section_not_counted(self) -> None:
        sections = _full_sections()
        sections.summary = "   "
        assert formatting_score.calculate(sections) == 85.0  # 100 - 15


class TestEdgeCases:
    def test_returns_float(self) -> None:
        assert isinstance(formatting_score.calculate(_full_sections()), float)

    def test_score_in_valid_range(self) -> None:
        for s in [_full_sections(), _empty_sections()]:
            sc = formatting_score.calculate(s)
            assert 0.0 <= sc <= 100.0

    def test_deterministic(self) -> None:
        s = _full_sections()
        assert formatting_score.calculate(s) == formatting_score.calculate(s)
