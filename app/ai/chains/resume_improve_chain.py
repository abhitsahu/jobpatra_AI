"""Resume improvement chain.

Thin LCEL chain that wraps the section-specific prompt → LLM → string output.
Designed to be streamed token-by-token for a real-time UX.

Responsibilities
----------------
* Select the correct ``ChatPromptTemplate`` for the requested section type.
* Build the optional resume context block to inject as additional context.
* Return an async generator of string tokens via ``astream()``.

This module does NOT:
* Handle SSE encoding — that belongs in the service layer.
* Call FastAPI or know about HTTP.
* Parse or validate section type — the caller (service layer) is responsible.
"""

from __future__ import annotations

from typing import AsyncGenerator

from langchain_core.output_parsers import StrOutputParser

from app.ai.chains.base_chain import invoke_with_tracing  # noqa: F401 (kept for sync usage)
from app.ai.prompts.resume_improve import SECTION_PROMPTS
from app.ai.providers.litellm_client import get_chat_model
from observability.langsmith_config import get_langsmith_callback


def _build_context_block(resume_context: str | None) -> str:
    """Produce an optional context section injected into the human message."""
    if not resume_context or not resume_context.strip():
        return ""
    return (
        "FULL RESUME CONTEXT (for reference only — do NOT copy wholesale):\n"
        f"{resume_context.strip()}\n"
    )


async def stream_improve_section(
    section_type: str,
    current_text: str,
    resume_context: str | None = None,
    *,
    request_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream improved text tokens for a resume section.

    Args:
        section_type:    One of ``SECTION_PROMPTS`` keys (e.g. ``"summary"``).
                         Falls back to ``"_default"`` if unknown.
        current_text:    The original section text to rewrite.
        resume_context:  Optional full-resume string for extra context.
        request_id:      Optional request ID for LangSmith tracing.

    Yields:
        String tokens from the LLM, one at a time.
    """
    prompt = SECTION_PROMPTS.get(section_type) or SECTION_PROMPTS["_default"]
    context_block = _build_context_block(resume_context)

    chain = prompt | get_chat_model() | StrOutputParser()

    # Attach LangSmith callback when tracing is enabled
    callback = get_langsmith_callback(
        tags=[f"resume_improve_{section_type}"],
        metadata={"request_id": request_id or "unknown", "section_type": section_type},
    )
    callbacks = [callback] if callback is not None else []

    async for token in chain.astream(
        {
            "current_text": current_text,
            "context_block": context_block,
        },
        config={"callbacks": callbacks},
    ):
        yield token
