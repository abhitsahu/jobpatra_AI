"""Unit tests for exact_matcher.

Deterministic. No AI. No network.
"""

from app.analysis.matching import exact_matcher


class TestExactMatch:
    def test_identical_strings_match(self) -> None:
        assert exact_matcher.match("React", ["React", "Docker"]) == "React"

    def test_returns_none_when_no_match(self) -> None:
        assert exact_matcher.match("AWS", ["React", "Docker"]) is None

    def test_returns_matching_jd_keyword(self) -> None:
        result = exact_matcher.match("Docker", ["React", "Docker", "Python"])
        assert result == "Docker"

    def test_empty_jd_returns_none(self) -> None:
        assert exact_matcher.match("React", []) is None


class TestCaseInsensitiveComparison:
    def test_uppercase_resume_lowercase_jd(self) -> None:
        assert exact_matcher.match("REACT", ["react"]) == "react"

    def test_lowercase_resume_uppercase_jd(self) -> None:
        assert exact_matcher.match("docker", ["DOCKER"]) == "DOCKER"

    def test_mixed_case_both_sides(self) -> None:
        assert exact_matcher.match("PyThOn", ["python"]) == "python"

    def test_case_insensitive_no_false_positive(self) -> None:
        # "React" should not match "Reacts"
        assert exact_matcher.match("React", ["Reacts"]) is None


class TestWhitespaceNormalisation:
    def test_leading_trailing_spaces_ignored(self) -> None:
        assert exact_matcher.match("  React  ", ["React"]) == "React"

    def test_multiple_inner_spaces_collapsed(self) -> None:
        assert exact_matcher.match("Node  js", ["Node js"]) == "Node js"

    def test_tab_treated_as_space(self) -> None:
        assert exact_matcher.match("Node\tjs", ["Node js"]) == "Node js"


class TestNormaliseHelper:
    def test_lowercases(self) -> None:
        assert exact_matcher.normalise("REACT") == "react"

    def test_strips_whitespace(self) -> None:
        assert exact_matcher.normalise("  hello  ") == "hello"

    def test_collapses_inner_spaces(self) -> None:
        assert exact_matcher.normalise("a  b  c") == "a b c"


class TestMatchAll:
    def test_all_matched(self) -> None:
        matched, unmatched_r, unmatched_jd = exact_matcher.match_all(
            ["React", "Docker"], ["React", "Docker"]
        )
        assert len(matched) == 2
        assert unmatched_r == []
        assert unmatched_jd == []

    def test_partial_match(self) -> None:
        matched, unmatched_r, unmatched_jd = exact_matcher.match_all(
            ["React", "Terraform"], ["React", "Docker"]
        )
        assert len(matched) == 1
        assert matched[0] == ("React", "React")
        assert "Terraform" in unmatched_r
        assert "Docker" in unmatched_jd

    def test_no_match(self) -> None:
        matched, unmatched_r, unmatched_jd = exact_matcher.match_all(
            ["Python"], ["Java"]
        )
        assert matched == []
        assert "Python" in unmatched_r
        assert "Java" in unmatched_jd
