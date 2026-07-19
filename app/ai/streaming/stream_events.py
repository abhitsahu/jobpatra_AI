"""Stream events.

Typed dataclasses representing each stage of the AI explanation pipeline.

These events are emitted during a streaming SSE response so the frontend
can display live progress to the user:

  User clicks Analyze
       ↓
  ATS Engine Running
       ↓
  Generating AI Explanation...
       ↓
  Analyzing Strengths...
       ↓
  Analyzing Weaknesses...
       ↓
  Generating Suggestions...
       ↓
  Completed

Design
------
* Each event has a fixed ``event`` field (the SSE event name).
* Each event has a ``data`` dict with structured payload.
* The ``ErrorEvent`` is always the last event on failure.
* The ``CompleteEvent`` carries the full ``ATSAnalyzeResponse`` on success.

This file does NOT:
  - Implement SSE encoding (see sse_encoder.py).
  - Implement the streaming generator (see streaming_service.py).
  - Import from FastAPI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EventType(str, Enum):
    """SSE event type names — matched by the frontend EventSource listener."""

    PIPELINE_STARTED = "pipeline_started"
    ATS_RUNNING = "ats_running"
    ATS_COMPLETE = "ats_complete"
    AI_STARTED = "ai_started"
    AI_ANALYZING_STRENGTHS = "ai_analyzing_strengths"
    AI_ANALYZING_WEAKNESSES = "ai_analyzing_weaknesses"
    AI_GENERATING_SUGGESTIONS = "ai_generating_suggestions"
    AI_COMPLETE = "ai_complete"
    AI_UNAVAILABLE = "ai_unavailable"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass(frozen=True)
class PipelineStartedEvent:
    """Emitted immediately when the pipeline begins processing."""

    event: str = field(default=EventType.PIPELINE_STARTED, init=False)
    data: dict = field(
        default_factory=lambda: {"message": "ATS analysis pipeline started."}
    )


@dataclass(frozen=True)
class ATSRunningEvent:
    """Emitted when the deterministic ATS engine is executing."""

    event: str = field(default=EventType.ATS_RUNNING, init=False)
    data: dict = field(
        default_factory=lambda: {"message": "ATS Engine running…"}
    )


@dataclass(frozen=True)
class ATSCompleteEvent:
    """Emitted when the deterministic ATS engine has finished scoring.

    Args:
        overall_score: The final ATS overall score [0–100].
    """

    overall_score: float

    event: str = field(default=EventType.ATS_COMPLETE, init=False)

    @property
    def data(self) -> dict:
        return {
            "message": f"ATS scoring complete — overall score: {self.overall_score:.1f}",
            "overall_score": self.overall_score,
        }


@dataclass(frozen=True)
class AIStartedEvent:
    """Emitted when the AI explanation chain begins."""

    event: str = field(default=EventType.AI_STARTED, init=False)
    data: dict = field(
        default_factory=lambda: {"message": "Generating AI explanation…"}
    )


@dataclass(frozen=True)
class AIAnalyzingStrengthsEvent:
    """Emitted at the conceptual start of strength analysis (pre-LLM call)."""

    event: str = field(default=EventType.AI_ANALYZING_STRENGTHS, init=False)
    data: dict = field(
        default_factory=lambda: {"message": "Analyzing strengths…"}
    )


@dataclass(frozen=True)
class AIAnalyzingWeaknessesEvent:
    """Emitted to indicate weakness gap analysis is in progress."""

    event: str = field(default=EventType.AI_ANALYZING_WEAKNESSES, init=False)
    data: dict = field(
        default_factory=lambda: {"message": "Analyzing weaknesses and gaps…"}
    )


@dataclass(frozen=True)
class AIGeneratingSuggestionsEvent:
    """Emitted to indicate suggestions are being generated."""

    event: str = field(default=EventType.AI_GENERATING_SUGGESTIONS, init=False)
    data: dict = field(
        default_factory=lambda: {"message": "Generating improvement suggestions…"}
    )


@dataclass(frozen=True)
class AICompleteEvent:
    """Emitted when the AI chain produced a valid explanation."""

    event: str = field(default=EventType.AI_COMPLETE, init=False)
    data: dict = field(
        default_factory=lambda: {"message": "AI explanation ready."}
    )


@dataclass(frozen=True)
class AIUnavailableEvent:
    """Emitted when the AI chain failed — deterministic result is still returned."""

    reason: str = "AI service unavailable."

    event: str = field(default=EventType.AI_UNAVAILABLE, init=False)

    @property
    def data(self) -> dict:
        return {"message": self.reason}


@dataclass(frozen=True)
class CompleteEvent:
    """Final event — carries the complete analysis response payload.

    Args:
        payload: The serialised ``ATSAnalyzeResponse`` dict.
    """

    payload: dict

    event: str = field(default=EventType.COMPLETE, init=False)

    @property
    def data(self) -> dict:
        return self.payload


@dataclass(frozen=True)
class ErrorEvent:
    """Terminal error event — emitted when the pipeline itself fails.

    Note: AI failures are NOT errors — they emit ``AIUnavailableEvent`` instead
    and the deterministic result is still returned via ``CompleteEvent``.
    """

    message: str = "An unexpected error occurred."
    code: str = "INTERNAL_ERROR"

    event: str = field(default=EventType.ERROR, init=False)

    @property
    def data(self) -> dict:
        return {"code": self.code, "message": self.message}


# Union type alias for typing convenience

StreamEvent = (
    PipelineStartedEvent
    | ATSRunningEvent
    | ATSCompleteEvent
    | AIStartedEvent
    | AIAnalyzingStrengthsEvent
    | AIAnalyzingWeaknessesEvent
    | AIGeneratingSuggestionsEvent
    | AICompleteEvent
    | AIUnavailableEvent
    | CompleteEvent
    | ErrorEvent
)
