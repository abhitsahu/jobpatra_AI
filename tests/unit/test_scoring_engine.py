"""Unit tests for scoring_engine (the top-level orchestrator).

Tests hand-calculate the expected weighted average to verify the engine
combines sub-scores correctly.

Weights:  keyword=0.30  experience=0.25  skills=0.25
          formatting=0.10  education=0.05  summary=0.05

Deterministic. No AI. No network.
"""

import pytest

from app.analysis.extraction.education_extractor import (
    EducationEntry,
    EducationExtractionResult,
)
from app.analysis.extraction.experience_extractor import ExperienceEntry
from app.analysis.matching.keyword_matcher import MatchedKeyword, MatchResult
from app.analysis.normalization.section_splitter import ResumeSection
from app.analysis.scoring import scoring_engine
from app.analysis.scoring.scoring_engine import ATSReport
from app.analysis.scoring.weights_config import DEFAULT_WEIGHTS, ScoringWeights


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _match_result(matched: int, missing: int) -> MatchResult:
    r = MatchResult()
    for i in range(matched):
        r.matched.append(MatchedKeyword(keyword=f"kw{i}", matchType="EXACT"))
    for i in range(missing):
        r.missing.append(f"m{i}")
    return r


def _experience(duration: float, bullets: int, metrics: int) -> list[ExperienceEntry]:
    return [
        ExperienceEntry(
            duration_years=duration,
            bullets=[f"b{i}" for i in range(bullets)],
            metrics=[f"{i}%" for i in range(metrics)],
        )
    ]


def _education(degree: str | None = "B.Sc", certs: int = 0) -> EducationExtractionResult:
    entries = [EducationEntry(degree=degree)] if degree else []
    return EducationExtractionResult(
        entries=entries,
        certifications=[f"Cert{i}" for i in range(certs)],
    )


def _full_sections() -> ResumeSection:
    return ResumeSection(
        summary=(
            "Led teams of engineers and reduced latency by 40% through "
            "architectural improvements. Built scalable distributed systems."
        ),
        experience="Software Engineer at XYZ 2020-2023",
        education="B.Sc Computer Science 2020",
        skills="Python Docker React",
        projects="Built a web app.",
        certifications="AWS Certified",
        languages="English",
    )


# ---------------------------------------------------------------------------
# Tests — return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_ats_report(self) -> None:
        report = scoring_engine.score(
            match_result=_match_result(3, 1),
            experience_entries=_experience(5.0, 3, 2),
            resume_skills=["Python"],
            required_skills=["Python"],
            education_result=_education(),
            sections=_full_sections(),
        )
        assert isinstance(report, ATSReport)

    def test_all_fields_present(self) -> None:
        report = scoring_engine.score(
            match_result=_match_result(3, 1),
            experience_entries=[],
            resume_skills=[],
            required_skills=[],
            education_result=_education(degree=None),
            sections=ResumeSection(),
        )
        assert hasattr(report, "keyword_score")
        assert hasattr(report, "experience_score")
        assert hasattr(report, "skills_score")
        assert hasattr(report, "formatting_score")
        assert hasattr(report, "education_score")
        assert hasattr(report, "summary_score")
        assert hasattr(report, "overall_score")


# ---------------------------------------------------------------------------
# Tests — weighted average hand-verification
# ---------------------------------------------------------------------------


class TestWeightedAverage:
    def test_known_sub_scores_produce_correct_overall(self) -> None:
        """Hand-calculated verification of the weighted formula.

        Sub-scores (hand-crafted inputs):
          keyword_score    = 75.0   (3 matched, 1 missing)
          experience_score = 0.0    (empty entries)
          skills_score     = 100.0  (1/1 required skill)
          formatting_score = 30.0   (experience section only)
          education_score  = 80.0   (B.Sc)
          summary_score    = 0.0    (no summary)

        weighted = 75×0.30 + 0×0.25 + 100×0.25 + 30×0.10 + 80×0.05 + 0×0.05
                 = 22.5 + 0 + 25.0 + 3 + 4 + 0
                 = 54.5
        """
        report = scoring_engine.score(
            match_result=_match_result(3, 1),
            experience_entries=[],
            resume_skills=["Python"],
            required_skills=["Python"],
            education_result=_education("B.Sc"),
            sections=ResumeSection(experience="Software Engineer at XYZ 2020-2023"),
        )
        assert report.keyword_score == 75.0
        assert report.experience_score == 0.0
        assert report.skills_score == 100.0
        assert report.formatting_score == 30.0
        assert report.education_score == 80.0
        assert report.summary_score == 0.0
        assert report.overall_score == pytest.approx(54.5, abs=0.01)

    def test_all_perfect_scores_give_overall_100(self) -> None:
        """When all sub-scores are at or near maximum, overall is near 100.

        Note: summary from _full_sections() scores 80 (32 words, not 50+),
        so the theoretical ceiling with these inputs is ~98.  We assert ≥95.
        """
        entries = [ExperienceEntry(
            duration_years=10.0,
            bullets=[f"b{i}" for i in range(4)],
            metrics=[f"{i}%" for i in range(6)],
        )] * 4
        report = scoring_engine.score(
            match_result=_match_result(10, 0),
            experience_entries=entries,
            resume_skills=["Python"],
            required_skills=["Python"],
            education_result=_education("PhD"),
            sections=_full_sections(),
        )
        assert report.overall_score >= 95.0

    def test_all_zero_scores_give_overall_zero(self) -> None:
        report = scoring_engine.score(
            match_result=_match_result(0, 5),
            experience_entries=[],
            resume_skills=[],
            required_skills=["Python"],
            education_result=_education(degree=None),
            sections=ResumeSection(),
        )
        assert report.overall_score == 0.0


# ---------------------------------------------------------------------------
# Tests — custom weights
# ---------------------------------------------------------------------------


class TestCustomWeights:
    def test_keyword_only_weights(self) -> None:
        """All weight on keyword_score → overall equals keyword_score."""
        weights = ScoringWeights(
            keyword_score=1.0,
            experience_score=0.0,
            skills_score=0.0,
            formatting_score=0.0,
            education_score=0.0,
            summary_score=0.0,
        )
        report = scoring_engine.score(
            match_result=_match_result(3, 1),      # keyword_score = 75
            experience_entries=[],
            resume_skills=[],
            required_skills=[],
            education_result=_education(degree=None),
            sections=ResumeSection(),
            weights=weights,
        )
        assert report.overall_score == pytest.approx(75.0, abs=0.01)


# ---------------------------------------------------------------------------
# Tests — determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_same_output(self) -> None:
        kwargs = dict(
            match_result=_match_result(5, 2),
            experience_entries=_experience(4.0, 3, 2),
            resume_skills=["Python", "Docker"],
            required_skills=["Python", "Docker", "AWS"],
            education_result=_education("B.Sc"),
            sections=_full_sections(),
        )
        r1 = scoring_engine.score(**kwargs)
        r2 = scoring_engine.score(**kwargs)
        assert r1.overall_score == r2.overall_score
        assert r1.keyword_score == r2.keyword_score

    def test_scores_in_valid_range(self) -> None:
        report = scoring_engine.score(
            match_result=_match_result(3, 1),
            experience_entries=_experience(5.0, 3, 2),
            resume_skills=["Python"],
            required_skills=["Python"],
            education_result=_education("B.Sc"),
            sections=_full_sections(),
        )
        for field in ("keyword_score", "experience_score", "skills_score",
                      "formatting_score", "education_score", "summary_score",
                      "overall_score"):
            val = getattr(report, field)
            assert 0.0 <= val <= 100.0, f"{field} = {val} out of range"
