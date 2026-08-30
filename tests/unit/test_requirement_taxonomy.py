"""Unit tests for the score-bearing ATS requirement taxonomy."""

from app.analysis.extraction.requirement_taxonomy import (
    classify_jd_requirements,
    resume_technical_evidence,
)
from app.schemas.extraction import JDExtraction, ResumeExtraction


class TestJDRequirementTaxonomy:
    def test_culture_signals_do_not_enter_technical_score_inputs(self) -> None:
        taxonomy = classify_jd_requirements(
            JDExtraction(
                required_hard_skills=["React", "API Integration"],
                preferred_hard_skills=["Kafka"],
                required_soft_skills=["Problem Solving"],
                domain_terms=[
                    "Payment Orchestration",
                    "First Principles Thinking",
                    "Passion for Reliability",
                ],
            )
        )

        assert taxonomy.required_technical_skills == ["React", "API Integration"]
        assert taxonomy.preferred_technical_skills == ["Kafka"]
        assert taxonomy.domain_terms == ["Payment Orchestration"]
        assert taxonomy.culture_signals == [
            "Problem Solving",
            "First Principles Thinking",
            "Passion for Reliability",
        ]
        assert taxonomy.keyword_requirements == [
            "React",
            "API Integration",
            "Kafka",
            "Payment Orchestration",
        ]

    def test_resume_technical_evidence_excludes_soft_skills(self) -> None:
        evidence = resume_technical_evidence(
            ResumeExtraction(
                hard_skills=["Python", "React"],
                soft_skills=["Communication"],
                domain_terms=["Microservices", "Python"],
            )
        )

        assert evidence == ["Python", "React", "Microservices"]

    def test_misclassified_culture_signal_cannot_enter_skill_denominator(self) -> None:
        taxonomy = classify_jd_requirements(
            JDExtraction(required_hard_skills=["Python", "First Principles Thinking"])
        )

        assert taxonomy.required_technical_skills == ["Python"]
        assert taxonomy.culture_signals == ["First Principles Thinking"]
