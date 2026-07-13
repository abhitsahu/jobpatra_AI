"""Unit tests for experience_extractor.

Deterministic. No AI. No network.
"""

from app.analysis.extraction import experience_extractor
from app.analysis.extraction.experience_extractor import ExperienceEntry


SAMPLE_EXPERIENCE = """\
Senior Developer at Acme Corp 2021-2024

Built microservices using Node.js and Docker.
Led a team of 5 engineers.
Reduced latency by 40%.

Junior Developer at Beta Inc 2019-2021

Developed REST APIs for 100+ clients.
Increased revenue by $200K.
"""

SINGLE_ENTRY = "Software Engineer at XYZ Corp 2020-2023"


class TestJobTitleExtraction:
    def test_title_extracted(self) -> None:
        entries = experience_extractor.extract(SINGLE_ENTRY)
        assert len(entries) >= 1
        assert entries[0].title is not None
        assert "Software Engineer" in entries[0].title

    def test_multiple_titles_extracted(self) -> None:
        entries = experience_extractor.extract(SAMPLE_EXPERIENCE)
        titles = [e.title for e in entries if e.title]
        assert any("Developer" in t for t in titles)


class TestCompanyExtraction:
    def test_company_extracted(self) -> None:
        entries = experience_extractor.extract(SINGLE_ENTRY)
        assert entries[0].company is not None
        assert "XYZ Corp" in entries[0].company

    def test_multiple_companies_extracted(self) -> None:
        entries = experience_extractor.extract(SAMPLE_EXPERIENCE)
        companies = [e.company for e in entries if e.company]
        assert any("Acme" in c for c in companies)
        assert any("Beta" in c for c in companies)


class TestDateExtraction:
    def test_start_date_extracted(self) -> None:
        entries = experience_extractor.extract(SINGLE_ENTRY)
        assert entries[0].start_date == "2020"

    def test_end_date_extracted(self) -> None:
        entries = experience_extractor.extract(SINGLE_ENTRY)
        assert entries[0].end_date == "2023"

    def test_duration_computed(self) -> None:
        entries = experience_extractor.extract(SINGLE_ENTRY)
        assert entries[0].duration_years == 3.0

    def test_present_end_date(self) -> None:
        text = "Engineer at OpenAI 2022-Present"
        entries = experience_extractor.extract(text)
        assert entries[0].end_date == "Present"
        assert entries[0].duration_years is not None


class TestMetricsExtraction:
    def test_percentage_metric_extracted(self) -> None:
        entries = experience_extractor.extract(SAMPLE_EXPERIENCE)
        all_metrics = [m for e in entries for m in e.metrics]
        assert any("40%" in m for m in all_metrics)

    def test_dollar_metric_extracted(self) -> None:
        entries = experience_extractor.extract(SAMPLE_EXPERIENCE)
        all_metrics = [m for e in entries for m in e.metrics]
        assert any("$200K" in m or "200K" in m for m in all_metrics)

    def test_plus_metric_extracted(self) -> None:
        entries = experience_extractor.extract(SAMPLE_EXPERIENCE)
        all_metrics = [m for e in entries for m in e.metrics]
        assert any("100+" in m for m in all_metrics)


class TestEdgeCases:
    def test_empty_string_returns_empty_list(self) -> None:
        assert experience_extractor.extract("") == []

    def test_no_dates_still_parses(self) -> None:
        text = "Software Engineer at Acme\n\nBuilt many things."
        entries = experience_extractor.extract(text)
        assert len(entries) >= 1
        assert entries[0].start_date is None

    def test_returns_list_of_experience_entries(self) -> None:
        entries = experience_extractor.extract(SAMPLE_EXPERIENCE)
        assert all(isinstance(e, ExperienceEntry) for e in entries)
