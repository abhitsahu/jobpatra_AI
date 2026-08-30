"""ATS Pydantic schemas — request and response shapes for the ATS API.

This file owns the public contract between callers (Next.js) and the
ATS pipeline.  Every field here is intentional and documented.

Does NOT contain:
  - ATS scoring logic
  - Parsing logic
  - Matching logic
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from typing import Literal, Any

from app.schemas.ai import ATSExplanation


# ---------------------------------------------------------------------------
# Existing input schemas (Phase 3 — preserved as-is)
# ---------------------------------------------------------------------------


class ResumeInput(BaseModel):
    """Input for a resume, supporting two mutually exclusive modes.

    Mode 1 — Uploaded file:
        Provide ``filename`` and ``file_bytes``. The parser factory will
        select the correct parser (PDF or DOCX) based on the extension.

    Mode 2 — Raw text:
        Provide ``text`` directly. Useful when the user pastes their
        resume content rather than uploading a file.

    Exactly one of (filename + file_bytes) or text must be provided.
    Validation enforces this constraint.
    """

    filename: str | None = None
    file_bytes: bytes | None = None
    text: str | None = None

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str | None) -> str | None:
        """Reject explicitly empty strings; None is acceptable (means file mode)."""
        if v is not None and not v.strip():
            raise ValueError("text must not be blank.")
        return v

    @field_validator("file_bytes", mode="before")
    @classmethod
    def decode_base64_file_bytes(cls, v: Any) -> Any:
        """Decode base64 string value to raw bytes."""
        if isinstance(v, str):
            import base64
            try:
                return base64.b64decode(v)
            except Exception as e:
                raise ValueError(f"Invalid base64 encoding: {e}")
        return v

    def is_file_mode(self) -> bool:
        """Return True if the caller supplied file bytes rather than raw text."""
        return self.filename is not None and self.file_bytes is not None

    def is_text_mode(self) -> bool:
        """Return True if the caller supplied raw text."""
        return self.text is not None


class JobDescriptionInput(BaseModel):
    """Input for a job description.

    Only plain text is accepted — job descriptions are always pasted or
    typed, never uploaded as files.
    """

    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        """Reject empty or whitespace-only job descriptions."""
        if not v.strip():
            raise ValueError("Job description text must not be blank.")
        return v


# ---------------------------------------------------------------------------
# Phase 9 — ATS analyze request / response
# ---------------------------------------------------------------------------


class ATSAnalyzeRequest(BaseModel):
    """Request body for POST /v1/ats/analyze.

    The resume can be supplied as raw text or as a file (PDF/DOCX via
    filename + file_bytes).  The job description is always raw text.
    """

    resume: ResumeInput
    """Resume — raw text or file bytes."""
    job_description: JobDescriptionInput
    """Job description — always raw text."""
    stream: bool = False
    """Whether to stream progress events via Server-Sent Events (SSE)."""

    @field_validator("resume")
    @classmethod
    def resume_must_have_input(cls, v: ResumeInput) -> ResumeInput:
        """Ensure the resume has either text or file bytes, not neither."""
        if not v.is_text_mode() and not v.is_file_mode():
            raise ValueError(
                "Resume must supply either 'text' or both 'filename' and 'file_bytes'."
            )
        return v


class MatchedKeywordSchema(BaseModel):
    """A single keyword that was matched between resume and JD."""

    keyword: str
    """The resume keyword as originally provided."""
    matchType: str
    """How it was matched: EXACT | SYNONYM | FUZZY | SEMANTIC."""
    similarity: float | None = None
    """Cosine similarity (only for SEMANTIC matches)."""
    matched_jd_keyword: str | None = None
    """The corresponding job-description term."""
    is_related_concept: bool = False
    """True when this is advisory related-concept evidence, not a direct match."""


class ExperienceSummarySchema(BaseModel):
    """High-level summary of extracted work experience."""

    total_entries: int
    """Number of distinct experience entries found."""
    total_years: float
    """Sum of all duration_years fields (where parseable)."""
    has_metrics: bool
    """True if any entry contained quantified achievements."""


class EducationSummarySchema(BaseModel):
    """High-level summary of extracted education."""

    highest_degree: str | None
    """The highest degree level detected (e.g. 'B.Sc', 'Master of Science')."""
    certifications: list[str]
    """List of certification names found in the resume."""


class ATSAnalyzeResponse(BaseModel):
    """Response body for POST /v1/ats/analyze.

    Contains the full deterministic ATS report — scores, matched/missing
    keywords, skill coverage, and extracted resume metadata — plus an
    optional AI-generated explanation produced by the LangChain chain.

    The deterministic scores are ALWAYS present.
    ``ai_explanation`` is populated only when the AI layer succeeds.
    ``ai_status`` is always present — "ok" or "unavailable".
    """

    # ── Scores ────────────────────────────────────────────────────────────
    overall_score: float
    """Weighted ATS score [0–100]."""
    keyword_score: float
    """Keyword coverage score [0–100]."""
    experience_score: float
    """Work experience quality score [0–100]."""
    skills_score: float
    """Skill coverage score [0–100]."""
    education_score: float
    """Education credential score [0–100]."""
    summary_score: float
    """Resume summary quality score [0–100]."""
    formatting_score: float
    """Resume structure completeness score [0–100]."""

    # ── Keyword matching ───────────────────────────────────────────────────
    matched_keywords: list[MatchedKeywordSchema]
    """Keywords from the JD that were found in the resume."""
    missing_keywords: list[str]
    """Keywords from the JD that were NOT found in the resume."""
    related_keywords: list[MatchedKeywordSchema] = Field(default_factory=list)
    """Embedding-derived related concepts excluded from coverage scoring."""

    # ── Skill coverage ─────────────────────────────────────────────────────
    matched_skills: list[str]
    """Canonical skill names present in both resume and JD."""
    missing_skills: list[str]
    """Canonical skill names required by the JD but absent from the resume."""
    required_skill_count: int = 0
    """Number of score-bearing required technical skills in the JD."""
    culture_signals: list[str] = Field(default_factory=list)
    """Feedback-only culture signals; these do not affect technical coverage."""
    extraction_mode: Literal["hybrid_ai", "deterministic_fallback"] = "deterministic_fallback"
    """Whether score inputs came from Hybrid AI extraction or technical fallback."""
    required_experience_years: float = 0.0
    candidate_experience_years: float = 0.0
    required_education_level: str = "none"
    candidate_education_level: str = "unknown"

    # ── Extracted metadata ─────────────────────────────────────────────────
    experience_summary: ExperienceSummarySchema
    """High-level work experience summary."""
    education_summary: EducationSummarySchema
    """High-level education summary."""

    # ── Meta ───────────────────────────────────────────────────────────────
    processing_time_ms: float
    """Wall-clock time for the full pipeline in milliseconds."""
    version: str = "1.2"
    """API response schema version."""

    # ── AI explanation (Phase 10.1 — optional) ─────────────────────────────
    ai_status: Literal["ok", "unavailable"] = Field(
        default="unavailable",
        description=(
            "'ok' when the AI layer produced a valid explanation. "
            "'unavailable' when the AI layer was skipped, failed, or degraded."
        ),
    )
    """AI availability status for this response."""

    ai_explanation: ATSExplanation | None = Field(
        default=None,
        description=(
            "AI-generated explanation of the ATS scores. "
            "Null when the AI service is unavailable or not configured."
        ),
    )
    """Structured AI explanation — strengths, weaknesses, per-section breakdown."""
