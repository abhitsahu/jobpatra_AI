"""Unit tests for skills_score.

Deterministic. No AI. No network.
"""

from app.analysis.scoring import skills_score


class TestBasicScoring:
    def test_four_of_five_required_skills(self) -> None:
        """4 resume / 5 required → 80.0."""
        resume = ["Python", "Docker", "React", "AWS"]
        required = ["Python", "Docker", "React", "Redis", "AWS"]
        assert skills_score.calculate(resume, required) == 80.0

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
