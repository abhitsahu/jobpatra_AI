"""LangSmith observability configuration — Phase 10.2.

This is the **only** file responsible for configuring LangSmith.

No chain, service, or route should import ``langsmith`` or create a
``LangChainTracer`` directly.  Everything goes through ``get_langsmith_callback()``.

Design decisions
----------------
* Tracing is **opt-in via config**.  ``LANGSMITH_TRACING_ENABLED=false`` (the
  default) means zero network calls, zero LangSmith imports at runtime.
* ``get_langsmith_callback()`` returns ``None`` when tracing is disabled or
  when configuration is incomplete.  Callers always check for ``None``.
* The LangSmith client is created fresh per-call (no singleton) so that API
  key rotation works without a server restart.
* Errors during client construction are caught and logged — tracing must never
  become a single point of failure for the AI chain.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.tracers.langchain import LangChainTracer

from app.core.config import settings
from app.core.logging import logger


def _configure_langchain_env() -> None:
    """Set the LangChain environment variables LangSmith reads at chain runtime.

    LangChain's automatic tracing is controlled by ``LANGCHAIN_TRACING_V2``
    and related env vars.  We set them programmatically from ``settings`` so
    that they always match the application configuration — not whatever
    happens to be in the shell environment.

    This function is idempotent: calling it multiple times is safe.
    """
    if not settings.LANGSMITH_TRACING_ENABLED:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return

    if settings.LANGSMITH_API_KEY is None:
        logger.warning(
            "[LangSmith] LANGSMITH_TRACING_ENABLED=true but LANGSMITH_API_KEY is not set. "
            "Tracing will be disabled."
        )
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY.get_secret_value()
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"


def get_langsmith_callback(
    *,
    tags: list[str] | None = None,
    metadata: dict[str, str] | None = None,
) -> "LangChainTracer | None":
    """Build a ``LangChainTracer`` callback for a single chain execution.

    Returns ``None`` when:
    - ``LANGSMITH_TRACING_ENABLED`` is ``False``.
    - ``LANGSMITH_API_KEY`` is absent.
    - Any error occurs during client construction.

    Callers pass the returned object (when non-None) to ``chain.invoke``
    via the ``config`` argument:

    .. code-block:: python

        cb = get_langsmith_callback(tags=["explain_score"])
        callbacks = [cb] if cb else []
        chain.invoke(inputs, config={"callbacks": callbacks})

    Args:
        tags:     Optional list of string tags attached to every run in this trace.
        metadata: Optional key-value metadata attached to every run.

    Returns:
        A configured ``LangChainTracer`` or ``None``.
    """
    _configure_langchain_env()

    if not settings.LANGSMITH_TRACING_ENABLED:
        return None

    if settings.LANGSMITH_API_KEY is None:
        return None

    try:
        import langsmith  # noqa: PLC0415 — lazy import, tracing is optional
        from langchain_core.tracers.langchain import LangChainTracer  # noqa: PLC0415

        client = langsmith.Client(
            api_key=settings.LANGSMITH_API_KEY.get_secret_value(),
            api_url="https://api.smith.langchain.com",
        )
        tracer = LangChainTracer(
            project_name=settings.LANGSMITH_PROJECT,
            client=client,
            tags=tags,
            metadata=metadata,
        )
        return tracer

    except Exception as exc:  # noqa: BLE001
        # Tracing failure must NEVER propagate to the caller.
        logger.warning("[LangSmith] Failed to create tracer (tracing skipped): %s", exc)
        return None


def is_tracing_enabled() -> bool:
    """Return True if LangSmith tracing is enabled and properly configured.

    Useful for logging or diagnostic endpoints.
    """
    return (
        settings.LANGSMITH_TRACING_ENABLED
        and settings.LANGSMITH_API_KEY is not None
    )
