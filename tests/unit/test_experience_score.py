"""Unit tests for experience_score.

Deterministic. No AI. No network.
"""

from app.analysis.extraction.experience_extractor import ExperienceEntry
from app.analysis.scoring import experience_score


def _entry(
    duration: float | None = None,
    bullets: list[str] | None = None,
    metrics: list[str] | None = None,
) -> ExperienceEntry:
    return ExperienceEntry(
        duration_years=duration,
        bullets=bullets or [],
        metrics=metrics or [],
    )


class TestEmptyInput:
    def test_empty_entries_returns_zero(self) -> None:
        assert experience_score.calculate([]) == 0.0


class TestDurationScore:
    def test_ten_years_gives_full_duration_points(self) -> None:
        # 10 years → full 40 pts; no bullets/metrics → 0; 1 job → 5 pts
        # 40 + 5 = 45 (continuity 1/4 × 20 = 5)
        entries = [_entry(duration=10.0)]
        score = experience_score.calculate(entries)
        assert score == 45.0

    def test_zero_duration_gives_zero_duration_points(self) -> None:
        entries = [_entry(duration=0.0)]
        score = experience_score.calculate(entries)
        # continuity 1 job → 1/4 × 20 = 5 pts only
        assert score == 5.0

    def test_five_years_gives_half_duration_points(self) -> None:
        entries = [_entry(duration=5.0)]
        # duration 5/10 × 40 = 20, continuity 1/4 × 20 = 5
        score = experience_score.calculate(entries)
        assert score == 25.0


class TestContinuityScore:
    def test_four_jobs_gives_full_continuity_points(self) -> None:
        entries = [_entry()] * 4
        # duration 0, continuity 4/4 × 20 = 20, bullets 0, metrics 0
        score = experience_score.calculate(entries)
        assert score == 20.0

    def test_two_jobs(self) -> None:
        entries = [_entry()] * 2
        # continuity 2/4 × 20 = 10
        assert experience_score.calculate(entries) == 10.0


class TestBulletDensityScore:
    def test_four_bullets_per_entry_gives_full_bullet_points(self) -> None:
        entries = [_entry(bullets=["a", "b", "c", "d"])]
        # duration 0, continuity 1/4×20=5, bullets 4/4×20=20, metrics 0
        score = experience_score.calculate(entries)
        assert score == 25.0


class TestMetricsScore:
    def test_six_metrics_gives_full_metric_points(self) -> None:
        entries = [_entry(metrics=["40%", "$100K", "5x", "200+", "3M", "$50K"])]
        # duration 0, continuity 5, bullets 0, metrics 6/6×20=20
        score = experience_score.calculate(entries)
        assert score == 25.0


class TestCombined:
    def test_strong_resume_near_hundred(self) -> None:
        # 2 entries: duration=5yr each → total 10yr (full 40pts)
        # 2 jobs → 2/4 × 20 = 10pts
        # avg 4 bullets/entry → full 20pts
        # 3 metrics × 2 entries = 6 total → full 20pts
        # Total = 40 + 10 + 20 + 20 = 90
        entries = [
            _entry(duration=5.0, bullets=["b1", "b2", "b3", "b4"], metrics=["40%", "$100K", "5x"]),
            _entry(duration=5.0, bullets=["b1", "b2", "b3", "b4"], metrics=["200+", "3M", "$50K"]),
        ]
        score = experience_score.calculate(entries)
        assert score == 90.0

    def test_score_clamped_to_100(self) -> None:
        entries = [_entry(duration=20.0, bullets=["x"] * 10, metrics=["1%"] * 10)] * 10
        assert experience_score.calculate(entries) == 100.0

    def test_score_in_valid_range(self) -> None:
        for dur in [0, 3, 7, 10]:
            entries = [_entry(duration=float(dur))]
            s = experience_score.calculate(entries)
            assert 0.0 <= s <= 100.0

    def test_deterministic(self) -> None:
        entries = [_entry(duration=3.0, bullets=["x", "y"], metrics=["5%"])]
        assert experience_score.calculate(entries) == experience_score.calculate(entries)
