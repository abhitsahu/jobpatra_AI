"""Regression coverage for taxonomy-backed ATS requirement scoring."""

from app.analysis.extraction.requirement_taxonomy import (
    classify_jd_requirements,
    fallback_jd_extraction,
)
from app.analysis.matching import keyword_matcher
from app.analysis.scoring.skills_score import evaluate
from app.schemas.extraction import JDExtraction
from app.services.taxonomy_service import get_taxonomy_service


def test_taxonomy_normalizes_aliases_and_resolves_related_technical_skills() -> None:
    """Aliases and graph relations replace the former manual synonym map."""
    taxonomy = get_taxonomy_service()
    assert taxonomy.normalize("ReactJS") == "React"
    assert taxonomy.are_related("Git", "GitHub")
    assert taxonomy.are_related("Pydantic", "FastAPI")
    assert taxonomy.is_parent_of("EC2", "AWS")

    result = keyword_matcher.match(
        resume_keywords=["Git", "Pydantic", "EC2"],
        jd_keywords=["GitHub", "FastAPI", "AWS"],
    )
    assert len(result.matched) == 3
    assert result.missing == []


def test_tools_and_fluff_do_not_lower_juspay_skill_coverage() -> None:
    """Only score-bearing taxonomy requirements may enter ATS denominators."""
    job_description = JDExtraction(
        required_hard_skills=["Python", "Black", "enthusiastic", "global"],
        domain_terms=["Payment Orchestration", "Microservices"],
        required_soft_skills=["First Principles Thinking"],
    )

    requirements = classify_jd_requirements(job_description)
    result = evaluate(["Python", "Microservices"], requirements.required_technical_skills)

    assert requirements.required_technical_skills == ["Python", "Microservices Architecture"]
    assert requirements.preferred_technical_skills == ["Black"]
    assert "enthusiastic" in requirements.feedback_only
    assert "global" in requirements.feedback_only
    assert "Payment Orchestration" in requirements.feedback_only
    assert result.required_skill_count == 2
    assert result.score == 100.0


def test_fallback_extractor_drops_generic_jd_prose() -> None:
    """The AI-outage path admits recognized taxonomy skills and nothing else."""
    fallback = fallback_jd_extraction(
        "Leading global payment team seeks an enthusiastic Python engineer with EC2 experience."
    )
    requirements = classify_jd_requirements(fallback)

    assert requirements.required_technical_skills == ["Python", "Amazon EC2"]
    assert requirements.feedback_only == []
