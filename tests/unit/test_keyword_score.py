"""Unit tests for keyword_score.

Deterministic. No AI. No network.
"""

from app.analysis.matching.keyword_matcher import MatchedKeyword, MatchResult
from app.analysis.scoring import keyword_score


def _result(matched: int, missing: int) -> MatchResult:
    """Build a minimal MatchResult with given counts."""
    r = MatchResult()
    for i in range(matched):
        r.matched.append(MatchedKeyword(keyword=f"kw{i}", matchType="EXACT"))
    for i in range(missing):
        r.missing.append(f"missing{i}")
    return r


class TestBasicScoring:
    def test_three_matched_one_missing(self) -> None:
        """3 matched + 1 missing → 75.0."""
        assert keyword_score.calculate(_result(3, 1)) == 75.0

    def test_all_matched(self) -> None:
        """4 matched + 0 missing → 100.0."""
        assert keyword_score.calculate(_result(4, 0)) == 100.0

    def test_none_matched(self) -> None:
        """0 matched + 4 missing → 0.0."""
        assert keyword_score.calculate(_result(0, 4)) == 0.0

    def test_half_matched(self) -> None:
        """2 matched + 2 missing → 50.0."""
        assert keyword_score.calculate(_result(2, 2)) == 50.0


class TestEdgeCases:
    def test_empty_result(self) -> None:
        """No keywords at all → 0.0."""
        assert keyword_score.calculate(MatchResult()) == 0.0

    def test_returns_float(self) -> None:
        result = keyword_score.calculate(_result(3, 1))
        assert isinstance(result, float)

    def test_score_in_valid_range(self) -> None:
        for matched, missing in [(0, 0), (1, 0), (0, 1), (5, 5)]:
            s = keyword_score.calculate(_result(matched, missing))
            assert 0.0 <= s <= 100.0

    def test_one_matched_nine_missing(self) -> None:
        assert keyword_score.calculate(_result(1, 9)) == 10.0

    def test_deterministic_same_input_same_output(self) -> None:
        r = _result(7, 3)
        assert keyword_score.calculate(r) == keyword_score.calculate(r)
