"""ATS input schemas.

These schemas define what data enters the parsing layer.
They do NOT carry ATS scores, normalized fields, or analysis results —
those belong to future phases.
"""

from pydantic import BaseModel, field_validator


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
