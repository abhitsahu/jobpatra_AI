"""Unit tests for education_extractor.

Deterministic. No AI. No network.
"""

from app.analysis.extraction import education_extractor
from app.analysis.extraction.education_extractor import EducationExtractionResult


SAMPLE_EDUCATION = """\
B.Sc Computer Science
State University 2017-2021
"""

MASTERS_EDUCATION = """\
Master of Science in Data Science
MIT 2021-2023
"""

CERTIFICATION_ONLY = """\
AWS Certified Developer
"""

MIXED = """\
B.Tech Computer Engineering
IIT Delhi 2016-2020

AWS Certified Solutions Architect
Google Cloud Professional
"""


class TestDegreeExtraction:
    def test_degree_extracted(self) -> None:
        result = education_extractor.extract(SAMPLE_EDUCATION)
        assert len(result.entries) >= 1
        assert result.entries[0].degree is not None
        assert "B.Sc" in result.entries[0].degree or "B" in result.entries[0].degree

    def test_masters_degree_extracted(self) -> None:
        result = education_extractor.extract(MASTERS_EDUCATION)
        assert len(result.entries) >= 1
        assert result.entries[0].degree is not None

    def test_field_of_study_extracted(self) -> None:
        result = education_extractor.extract(SAMPLE_EDUCATION)
        assert result.entries[0].field is not None
        assert "Computer" in result.entries[0].field or "Science" in result.entries[0].field


class TestInstitutionExtraction:
    def test_institution_extracted(self) -> None:
        result = education_extractor.extract(SAMPLE_EDUCATION)
        assert result.entries[0].institution is not None
        assert "University" in result.entries[0].institution or "State" in result.entries[0].institution

    def test_mit_institution_extracted(self) -> None:
        result = education_extractor.extract(MASTERS_EDUCATION)
        assert result.entries[0].institution is not None
        assert "MIT" in result.entries[0].institution


class TestGraduationYearExtraction:
    def test_graduation_year_extracted(self) -> None:
        result = education_extractor.extract(SAMPLE_EDUCATION)
        assert result.entries[0].graduation_year is not None
        assert result.entries[0].graduation_year in ("2017", "2021")

    def test_year_is_string(self) -> None:
        result = education_extractor.extract(SAMPLE_EDUCATION)
        assert isinstance(result.entries[0].graduation_year, str)


class TestCertificationExtraction:
    def test_certification_extracted(self) -> None:
        result = education_extractor.extract(CERTIFICATION_ONLY)
        assert len(result.certifications) >= 1
        assert any("AWS" in c for c in result.certifications)

    def test_certifications_and_degree_separated(self) -> None:
        result = education_extractor.extract(MIXED)
        assert len(result.entries) >= 1           # has degree
        assert len(result.certifications) >= 1    # has cert
        assert any("AWS" in c for c in result.certifications)


class TestEdgeCases:
    def test_empty_string_returns_empty_result(self) -> None:
        result = education_extractor.extract("")
        assert isinstance(result, EducationExtractionResult)
        assert result.entries == []
        assert result.certifications == []

    def test_returns_education_extraction_result(self) -> None:
        result = education_extractor.extract(SAMPLE_EDUCATION)
        assert isinstance(result, EducationExtractionResult)

    def test_no_exception_on_freeform_text(self) -> None:
        result = education_extractor.extract("Studied things and learned stuff at a place.")
        assert isinstance(result, EducationExtractionResult)
