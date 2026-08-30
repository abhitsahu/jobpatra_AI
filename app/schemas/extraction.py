"""Extraction Pydantic schemas — structured output contracts for AI entity extraction.

Defines the data models that the ``extract_entities_chain`` validates LLM
output against.  All list fields default to empty lists so a partially valid
LLM response never causes hard failures — the chain repairs what it can and
falls through to the Python fallback when the output is unrecoverable.

Does NOT contain:
  - Prompt text
  - Chain logic
  - LLM provider configuration
  - Scoring logic
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ResumeExtraction(BaseModel):
    """Structured entities extracted from a candidate's resume by the AI agent.

    All list fields use default_factory=list so Pydantic never raises on a
    partially returned JSON object.  Callers must handle empty lists gracefully.
    """

    hard_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Technical tools, programming languages, frameworks, and platforms "
            "explicitly mentioned in the resume (e.g., 'Python', 'React', 'AWS')."
        ),
    )
    soft_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Behavioural and interpersonal traits mentioned "
            "(e.g., 'leadership', 'communication', 'cross-functional collaboration')."
        ),
    )
    domain_terms: list[str] = Field(
        default_factory=list,
        description=(
            "Role-specific domain concepts for both technical and non-technical roles. "
            "Technical examples: 'CI/CD', 'Microservices', 'REST APIs'. "
            "Non-technical examples: 'Talent Acquisition', 'P&L Management', 'Pipeline Management'."
        ),
    )
    experience_years: float = Field(
        default=0.0,
        description="Total years of professional experience parsed from dates (e.g., 4.5).",
    )
    job_titles: list[str] = Field(
        default_factory=list,
        description="All job titles held by the candidate (e.g., 'Senior Software Engineer').",
    )
    achievements: list[str] = Field(
        default_factory=list,
        description=(
            "Quantified achievement bullets containing %, $, or multipliers. "
            "Empty list if none found."
        ),
    )
    education: str = Field(
        default="",
        description="Highest degree detected (e.g., 'B.Tech Computer Science').",
    )


class JDExtraction(BaseModel):
    """Structured hiring requirements extracted from a job description by the AI agent.

    All list fields use default_factory=list so Pydantic never raises on a
    partially returned JSON object.
    """

    required_hard_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Technical skills the JD explicitly requires "
            "(e.g., 'Python', 'PostgreSQL', 'Docker')."
        ),
    )
    preferred_hard_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Technical skills listed as 'nice to have' or 'preferred' in the JD "
            "(e.g., 'Kubernetes', 'GraphQL')."
        ),
    )
    required_soft_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Behavioural traits the JD explicitly requires "
            "(e.g., 'team player', 'strong communication')."
        ),
    )
    domain_terms: list[str] = Field(
        default_factory=list,
        description=(
            "Role-specific domain concepts for both technical and non-technical roles. "
            "HR examples: 'Talent Acquisition', 'HRIS', 'Employee Relations'. "
            "Sales examples: 'CRM', 'Pipeline Management', 'Revenue Forecasting'. "
            "Finance examples: 'GAAP', 'Financial Modeling', 'P&L Ownership'."
        ),
    )
    min_experience: float = Field(
        default=0.0,
        description="Minimum years of experience stated in the JD (numeric). 0 if not stated.",
    )
    required_education_level: Literal[
        "none", "associate", "bachelors", "masters", "phd"
    ] = Field(
        default="none",
        description=(
            "Minimum education level explicitly required by the JD. Use 'none' "
            "when the JD does not state an education requirement."
        ),
    )
    key_responsibilities: list[str] = Field(
        default_factory=list,
        description="3–6 concise phrases summarising the core responsibilities of the role.",
    )
