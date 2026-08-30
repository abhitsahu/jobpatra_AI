"""Unit tests for skills_score.

Deterministic. No AI. No network.
"""

from app.analysis.scoring import skills_score
import pytest


class TestBasicScoring:
    def test_four_of_five_required_skills(self) -> None:
        """4 resume / 5 required → 80.0."""
        resume = ["Python", "Docker", "React", "AWS"]
        required = ["Python", "Docker", "React", "Redis", "AWS"]
        assert skills_score.calculate(resume, required) == pytest.approx((3.4 / 4.4) * 100.0)

    def test_all_required_skills_present(self) -> None:
        resume = ["Python", "Docker"]
        required = ["Python", "Docker"]
        assert skills_score.calculate(resume, required) == 100.0

    def test_no_required_skills_matched(self) -> None:
        resume = ["Java", "Spring"]
        required = ["Python", "Docker"]
        assert skills_score.calculate(resume, required) == 0.0

    def test_empty_required_returns_zero(self) -> None:
        assert skills_score.calculate(["Python"], []) == 0.0

    def test_empty_resume_returns_zero(self) -> None:
        assert skills_score.calculate([], ["Python", "Docker"]) == 0.0

    def test_evaluation_exposes_the_exact_score_evidence(self) -> None:
        result = skills_score.evaluate(
            ["React.js", "Microservices", "Python"],
            ["React", "Distributed Systems", "Functional Programming"],
            embedding_provider=None,
        )

        assert result.required_skill_count == 2
        assert result.score == 100.0
        assert [match.keyword for match in result.match_result.matched] == [
            "React.js",
            "Microservices",
        ]
        assert result.match_result.missing == []


class TestCaseInsensitivity:
    def test_case_insensitive_match(self) -> None:
        resume = ["python", "docker"]
        required = ["Python", "Docker"]
        assert skills_score.calculate(resume, required) == 100.0

    def test_mixed_case_both_sides(self) -> None:
        resume = ["PYTHON"]
        required = ["python"]
        assert skills_score.calculate(resume, required) == 100.0


class TestEdgeCases:
    def test_score_in_valid_range(self) -> None:
        for r, req in [
            ([], []),
            (["Python"], ["Python"]),
            (["Java"], ["Python"]),
            (["a", "b", "c"], ["a", "b"]),
        ]:
            s = skills_score.calculate(r, req)
            assert 0.0 <= s <= 100.0

    def test_deterministic(self) -> None:
        r = ["React", "Node"]
        req = ["React", "Node", "Docker"]
        assert skills_score.calculate(r, req) == skills_score.calculate(r, req)
