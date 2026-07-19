"""AI Pydantic schemas — structured output contract for LLM responses.

This file defines the shapes that the AI layer must conform to.
The LLM is instructed to produce output matching these models exactly.

JobPatra AI role
----------------
JobPatra is an **ATS Analyzer and AI Resume Advisor**.
The AI NEVER modifies a user's resume.
The AI ONLY:
  - Explains ATS scores
  - Explains missing keywords
  - Explains weaknesses
  - Suggests improvements (user acts on these manually)

Does NOT contain:
  - Prompt text
  - Chain logic
  - LLM provider configuration
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class SectionExplanation(BaseModel):
    """AI explanation for a single ATS sub-score dimension."""

    section: str = Field(description="Name of the dimension, e.g. 'Keywords', 'Experience'.")
    score: float = Field(description="The deterministic sub-score for this section [0–100].")
    explanation: str = Field(
        description=(
            "Concise explanation of why the resume received this score for this section. "
            "Reference specific matched or missing items where possible."
        )
    )


class RecommendationSchema(BaseModel):
    """A detailed, actionable recommendation with copy-paste ready content."""

    priority: Literal["High", "Medium", "Low"] = Field(
        description="The priority of the change: High, Medium, or Low, sorted by estimated ATS impact."
    )
    issue: str = Field(
        description="A specific, clear description of the problem identified (e.g., 'Missing summary' or a specific bullet to rewrite)."
    )
    why: str = Field(
        description="Why this recommendation matters for the ATS system scoring or human recruiter's decision."
    )
    copy_paste_content: str = Field(
        description="Ready-to-use, copy-paste content to solve the issue. Must be complete and fully tailored to the user's resume and job description."
    )
    placement: str = Field(
        description="Specific instructions detailing where to place the generated copy-paste content in the resume."
    )
    ats_impact: str = Field(
        description="Estimated ATS score improvement (e.g., '+15 points' or 'Significantly improves skills matching score')."
    )


class ATSExplanation(BaseModel):
    """Structured AI explanation and actionable suggestions for a deterministic ATS report.

    Produced by the explain-score chain.

    INVARIANTS:
      - The AI NEVER modifies scores — it only explains the scores the deterministic engine computed.
      - Suggestions are advisory only — the user decides whether and how to act on them.
    """

    strengths: list[str] = Field(
        description=(
            "2–4 concrete strengths of this resume relative to the job description. "
            "Each item is a single, actionable sentence."
        )
    )
    weaknesses: list[str] = Field(
        description=(
            "2–4 concrete weaknesses or gaps. "
            "Each item is a single, actionable sentence."
        )
    )
    section_explanations: list[SectionExplanation] = Field(
        description="Per-dimension breakdown explaining each sub-score."
    )
    suggestions: list[str] = Field(
        description=(
            "3–6 specific, actionable improvement suggestions the user can apply manually. "
            "Each suggestion is one clear sentence."
        )
    )
    summary: str = Field(
        description=(
            "One-paragraph executive summary of the resume's fit for the role, "
            "referencing the overall ATS score."
        )
    )
    recommendations: list[RecommendationSchema] = Field(
        default=[],
        description="List of detailed recommendations containing copy-paste ready content, prioritized by ATS impact."
    )
