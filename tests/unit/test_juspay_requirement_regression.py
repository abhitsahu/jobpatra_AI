"""Regression coverage for the Juspay backend-role requirement taxonomy.

This test uses the stable entities observed in the Abhit Sahu/Juspay run. It
does not call an LLM or read the candidate's source document.
"""

from app.analysis.extraction.requirement_taxonomy import (
    classify_jd_requirements,
    resume_technical_evidence,
)
from app.analysis.matching.semantic_matcher import EmbeddingProvider
from app.analysis.scoring.skills_score import evaluate
from app.schemas.extraction import JDExtraction, ResumeExtraction


class ZeroEmbeddingProvider(EmbeddingProvider):
    """Deterministic provider that prevents semantic matches in this test."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]


def test_juspay_technical_coverage_excludes_culture_signals() -> None:
    """Only five explicit technical requirements form the skills denominator."""
    resume = ResumeExtraction(
        hard_skills=["Python", "React.js", "FastAPI", "AWS"],
        domain_terms=["Microservices", "API Integration", "Infrastructure as Code"],
    )
    job_description = JDExtraction(
        required_hard_skills=[
            "Functional Programming",
            "API Integration",
            "Distributed Systems",
            "React",
            "System Architecture",
        ],
        required_soft_skills=["Problem Solving"],
        domain_terms=[
            "Payment Orchestration",
            "First Principles Thinking",
            "Passion for Reliability",
        ],
    )

    taxonomy = classify_jd_requirements(job_description)
    result = evaluate(
        resume_technical_evidence(resume),
        taxonomy.required_technical_skills,
        embedding_provider=ZeroEmbeddingProvider(),
    )

    assert result.required_skill_count == 5
    assert result.score == 60.0
    assert {match.keyword for match in result.match_result.matched} == {
        "React.js",
        "Microservices",
        "API Integration",
    }
    assert result.match_result.missing == [
        "Functional Programming",
        "System Architecture",
    ]
    assert taxonomy.culture_signals == [
        "Problem Solving",
        "First Principles Thinking",
        "Passion for Reliability",
    ]
