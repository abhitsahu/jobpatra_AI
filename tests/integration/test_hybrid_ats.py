"""Integration tests for the Hybrid AI Extraction ATS flow.

Tests:
  1. Happy path: AI entity extraction succeeds and feeds clean lists to the
     deterministic matcher & scoring engine.
  2. Fallback path: AI extraction fails (raises AIGenerationError) and the
     service gracefully falls back to naive extractors without crashing.
  3. Non-taxonomy business and culture terms remain feedback-only rather than
     silently diluting the technical ATS denominator.
"""

from unittest.mock import patch
import pytest

from app.core.errors import AIGenerationError
from app.schemas.ats import ATSAnalyzeRequest, JobDescriptionInput, ResumeInput
from app.schemas.extraction import JDExtraction, ResumeExtraction
from app.services import ats_service


@pytest.fixture
def mock_explain_score():
    """Mock the explain-score chain so tests run fast without calling real LLMs."""
    with patch("app.services.ats_service.run_explain_score") as mock_fn:
        mock_fn.return_value = None
        yield mock_fn


def test_hybrid_ai_extraction_success_path(mock_explain_score):
    """Verify that when AI entity extraction succeeds, the output score and matches are computed correctly."""
    mock_resume_ext = ResumeExtraction(
        hard_skills=["Python", "FastAPI", "PostgreSQL"],
        soft_skills=["Leadership", "Communication"],
        domain_terms=["Microservices", "REST APIs"],
        experience_years=5.0,
        job_titles=["Senior Backend Engineer"],
        education="B.Tech Computer Science",
    )

    mock_jd_ext = JDExtraction(
        required_hard_skills=["Python", "FastAPI", "PostgreSQL"],
        preferred_hard_skills=["Docker", "Kubernetes"],
        required_soft_skills=["Leadership"],
        domain_terms=["REST APIs"],
        min_experience=3.0,
        key_responsibilities=["Develop microservices"],
    )

    with (
        patch("app.services.ats_service.extract_resume_entities", return_value=mock_resume_ext),
        patch("app.services.ats_service.extract_jd_entities", return_value=mock_jd_ext),
    ):
        request = ATSAnalyzeRequest(
            resume=ResumeInput(
                text="Senior Software Engineer with 5 years experience in Python, FastAPI, PostgreSQL, and REST APIs."
            ),
            job_description=JobDescriptionInput(
                text="Looking for a Python developer proficient in FastAPI and PostgreSQL with REST APIs experience."
            ),
        )

        response = ats_service.analyze(request)

        assert response.overall_score > 0.0
        assert response.keyword_score == 75.0  # 3 matched of 4 taxonomy-required JD skills
        matched_words = [m.keyword for m in response.matched_keywords]
        assert "Python" in matched_words
        assert "FastAPI" in matched_words


def test_hybrid_merge_keeps_skills_section_terms_omitted_by_ai(mock_explain_score):
    """Explicit Skills-section terms survive even when Gemini does not return them."""
    mock_resume_ext = ResumeExtraction(hard_skills=["Python"])
    mock_jd_ext = JDExtraction(required_hard_skills=["HTML", "CSS", "React"])

    with (
        patch("app.services.ats_service.extract_resume_entities", return_value=mock_resume_ext),
        patch("app.services.ats_service.extract_jd_entities", return_value=mock_jd_ext),
    ):
        entities = ats_service._extract_entities_hybrid(
            resume_clean="Resume text",
            jd_clean="Frontend Developer requires HTML, CSS, and React.",
            skills_section="Frontend: HTML, CSS, React",
        )

    assert {"HTML", "CSS", "React", "Python"}.issubset(entities.resume_skills)


def test_hybrid_merge_supports_another_comma_separated_skills_section(mock_explain_score):
    """The merge is data-driven for skills omitted from any AI response."""
    with (
        patch("app.services.ats_service.extract_resume_entities", return_value=ResumeExtraction()),
        patch("app.services.ats_service.extract_jd_entities", return_value=JDExtraction()),
    ):
        entities = ats_service._extract_entities_hybrid(
            resume_clean="Another candidate resume",
            jd_clean="Platform Engineer",
            skills_section="Kubernetes, Docker",
        )

    assert entities.resume_skills == ["Kubernetes", "Docker"]


def test_hybrid_ai_extraction_fallback_path(mock_explain_score):
    """Verify that when AI entity extraction fails, the pipeline falls back to naive extractors seamlessly."""
    with (
        patch(
            "app.services.ats_service.extract_resume_entities",
            side_effect=AIGenerationError("LLM rate limit reached"),
        ),
    ):
        request = ATSAnalyzeRequest(
            resume=ResumeInput(
                text="John Doe\nExperience:\nSoftware Engineer at Acme (2020-2023).\nBuilt Python web apps."
            ),
            job_description=JobDescriptionInput(
                text="We are hiring a Python Software Engineer with web development experience."
            ),
        )

        response = ats_service.analyze(request)

        # Output should be valid and derived from naive extractors
        assert response.overall_score >= 0.0
        assert response.processing_time_ms > 0.0


def test_unknown_hr_domain_terms_are_excluded_from_technical_scoring(mock_explain_score):
    """Business terms outside the taxonomy must not become technical requirements."""
    mock_resume_ext = ResumeExtraction(
        hard_skills=["Workday", "Excel"],
        soft_skills=["Interpersonal Communication"],
        domain_terms=["Talent Acquisition", "Employee Relations", "HRIS"],
        experience_years=4.0,
        job_titles=["HR Specialist"],
        education="Bachelor of Business Administration",
    )

    mock_jd_ext = JDExtraction(
        required_hard_skills=["Workday"],
        preferred_hard_skills=["Greenhouse"],
        required_soft_skills=["Interpersonal Communication"],
        domain_terms=["Talent Acquisition", "Employee Relations"],
        min_experience=3.0,
    )

    with (
        patch("app.services.ats_service.extract_resume_entities", return_value=mock_resume_ext),
        patch("app.services.ats_service.extract_jd_entities", return_value=mock_jd_ext),
    ):
        request = ATSAnalyzeRequest(
            resume=ResumeInput(
                text="HR Generalist experienced in Talent Acquisition, Employee Relations, and Workday HRIS."
            ),
            job_description=JobDescriptionInput(
                text="HR Manager needed to lead Talent Acquisition and Employee Relations using Workday."
            ),
        )

        response = ats_service.analyze(request)

        assert response.matched_keywords == []
        assert response.missing_keywords == []
        assert response.keyword_score == 0.0
