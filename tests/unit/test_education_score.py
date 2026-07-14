"""Unit tests for education_score.

Deterministic. No AI. No network.
"""

from app.analysis.extraction.education_extractor import EducationEntry, EducationExtractionResult
from app.analysis.scoring import education_score


def _result(degrees: list[str | None] | None = None, certs: int = 0) -> EducationExtractionResult:
    entries = [EducationEntry(degree=d) for d in (degrees or [])]
    return EducationExtractionResult(
        entries=entries,
        certifications=[f"Cert{i}" for i in range(certs)],
    )


class TestDegreeLevel:
    def test_bachelor_gives_80(self) -> None:
        assert education_score.calculate(_result(["B.Sc"])) == 80.0

    def test_bachelor_keyword_gives_80(self) -> None:
        assert education_score.calculate(_result(["Bachelor of Science"])) == 80.0

    def test_masters_gives_90(self) -> None:
        assert education_score.calculate(_result(["Master of Science"])) == 90.0

    def test_mba_gives_90(self) -> None:
        assert education_score.calculate(_result(["MBA"])) == 90.0

    def test_phd_gives_100(self) -> None:
        assert education_score.calculate(_result(["PhD"])) == 100.0

    def test_doctorate_gives_100(self) -> None:
        assert education_score.calculate(_result(["Doctorate"])) == 100.0

    def test_no_degree_gives_zero(self) -> None:
        assert education_score.calculate(_result()) == 0.0

    def test_associate_gives_60(self) -> None:
        assert education_score.calculate(_result(["Associate of Science"])) == 60.0


class TestCertificationBonus:
    def test_one_cert_adds_bonus(self) -> None:
        # Bachelor (80) + 1 cert × 5 = 85
        score = education_score.calculate(_result(["B.Sc"], certs=1))
        assert score == 85.0

    def test_two_certs_adds_10_bonus(self) -> None:
        # Bachelor (80) + 2 × 5 = 90
        score = education_score.calculate(_result(["B.Sc"], certs=2))
        assert score == 90.0

    def test_cert_bonus_capped_at_10(self) -> None:
        # Bachelor (80) + cap(10) = 90 regardless of 5 certs
        score = education_score.calculate(_result(["B.Sc"], certs=5))
        assert score == 90.0

    def test_certs_alone_no_degree(self) -> None:
        score = education_score.calculate(_result(certs=2))
        # No degree → base 0 + 10 bonus = 10
        assert score == 10.0


class TestEdgeCases:
    def test_highest_degree_used(self) -> None:
        # Has both associate and masters → should use masters (90)
        score = education_score.calculate(_result(["Associate of Science", "Master of Science"]))
        assert score == 90.0

    def test_score_clamped_to_100(self) -> None:
        # PhD (100) + certs → must not exceed 100
        score = education_score.calculate(_result(["PhD"], certs=5))
        assert score == 100.0

    def test_deterministic(self) -> None:
        r = _result(["B.Sc"], certs=1)
        assert education_score.calculate(r) == education_score.calculate(r)
