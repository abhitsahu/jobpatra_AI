"""Unit tests for summary_score.

Deterministic. No AI. No network.
"""

from app.analysis.scoring import summary_score


_GOOD_SUMMARY = (
    "Experienced software engineer with 5 years of expertise in building scalable "
    "distributed systems. Led teams of 10+ engineers and reduced latency by 40% "
    "through architectural improvements. Passionate about clean code and mentoring."
)

_SHORT_SUMMARY = "I am a developer."

_MINIMAL_SUMMARY = "Developer."


class TestPresence:
    def test_none_returns_zero(self) -> None:
        assert summary_score.calculate(None) == 0.0

    def test_empty_string_returns_zero(self) -> None:
        assert summary_score.calculate("") == 0.0

    def test_whitespace_only_returns_zero(self) -> None:
        assert summary_score.calculate("   ") == 0.0

    def test_present_summary_scores_above_zero(self) -> None:
        assert summary_score.calculate(_SHORT_SUMMARY) > 0.0


class TestLengthBonuses:
    def test_short_summary_no_word_count_bonus(self) -> None:
        # "I am a developer." = 4 words → only presence (40 pts)
        score = summary_score.calculate(_MINIMAL_SUMMARY)
        assert score == 40.0

    def test_summary_over_20_words_gets_first_bonus(self) -> None:
        text = " ".join(["word"] * 22)
        score = summary_score.calculate(text)
        assert score >= 60.0  # presence(40) + 20wds(20)

    def test_summary_over_50_words_gets_both_bonuses(self) -> None:
        text = " ".join(["word"] * 55)
        score = summary_score.calculate(text)
        assert score >= 80.0  # presence(40) + 20wds(20) + 50wds(20)


class TestActionWord:
    def test_action_word_adds_10_points(self) -> None:
        text = " ".join(["word"] * 22) + " led"
        score_with = summary_score.calculate(text)
        text_without = " ".join(["word"] * 22)
        score_without = summary_score.calculate(text_without)
        assert score_with == score_without + 10

    def test_good_summary_gets_action_word_bonus(self) -> None:
        # _GOOD_SUMMARY has 32 words (≥20 but <50)
        # presence(40) + word≥20(20) + action_word(10) + metric_40%(10) = 80
        score = summary_score.calculate(_GOOD_SUMMARY)
        assert score == 80.0


class TestMetricBonus:
    def test_percentage_adds_10_points(self) -> None:
        # Use neutral words (not in action list) to isolate the metric bonus.
        # 22 'item' tokens (≥20 words) + metric. No action word.
        # presence(40) + word≥20(20) + metric(10) = 70
        text_with = " ".join(["item"] * 22) + " 40%"
        text_without = " ".join(["item"] * 22)
        score_with = summary_score.calculate(text_with)
        score_without = summary_score.calculate(text_without)
        assert score_with == score_without + 10
        assert score_with == 70.0
        assert score_without == 60.0

    def test_dollar_metric_adds_bonus(self) -> None:
        text = " ".join(["word"] * 22) + " saved $200K"
        score = summary_score.calculate(text)
        assert score >= 70.0


class TestEdgeCases:
    def test_score_in_valid_range(self) -> None:
        for text in [None, "", _MINIMAL_SUMMARY, _GOOD_SUMMARY]:
            s = summary_score.calculate(text)
            assert 0.0 <= s <= 100.0

    def test_deterministic(self) -> None:
        assert summary_score.calculate(_GOOD_SUMMARY) == summary_score.calculate(_GOOD_SUMMARY)
