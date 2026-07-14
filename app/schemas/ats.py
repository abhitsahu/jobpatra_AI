"""ATS Pydantic schemas — request and response shapes for the ATS API.

This file owns the public contract between callers (Next.js) and the
ATS pipeline.  Every field here is intentional and documented.

Does NOT contain:
  - ATS scoring logic
  - Parsing logic
  - Matching logic
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator


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
    keywords, skill coverage, and extracted resume metadata.

    Does NOT contain:
      - AI suggestions
      - Resume rewrites
      - Cover letter content
      - Interview questions
      - Explanations (those belong to a future AI phase)
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

    # ── Skill coverage ─────────────────────────────────────────────────────
    matched_skills: list[str]
    """Canonical skill names present in both resume and JD."""
    missing_skills: list[str]
    """Canonical skill names required by the JD but absent from the resume."""

    # ── Extracted metadata ─────────────────────────────────────────────────
    experience_summary: ExperienceSummarySchema
    """High-level work experience summary."""
    education_summary: EducationSummarySchema
    """High-level education summary."""

    # ── Meta ───────────────────────────────────────────────────────────────
    processing_time_ms: float
    """Wall-clock time for the full pipeline in milliseconds."""
    version: str = "1.0"
    """API response schema version."""
