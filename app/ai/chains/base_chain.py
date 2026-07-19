"""Base chain utilities

This module provides the ``with_tracing`` helper that every LCEL chain should
use to attach LangSmith observability.

Responsibilities
----------------
* Attach the LangSmith callback to a chain's invocation config.
* Forward any additional callbacks supplied by the caller.
* Return the same chain — no wrapping, no mutation.

This file does NOT:
  - Build prompts
  - Contain business logic
  - Interact with LLMs directly
  - Know about specific chains

Usage
-----
.. code-block:: python

    from app.ai.chains.base_chain import invoke_with_tracing

    result = invoke_with_tracing(
        chain=my_chain,
        inputs={"key": "value"},
        tags=["my_chain_v1"],
        metadata={"request_id": rid},
    )

``invoke_with_tracing`` is a thin wrapper around ``chain.invoke``.  It passes
the LangSmith callback through ``config["callbacks"]``.  If tracing is
disabled, the chain executes with an empty callback list — identical to calling
``chain.invoke(inputs)`` directly.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import Runnable

from observability.langsmith_config import get_langsmith_callback


def invoke_with_tracing(
    chain: Runnable,
    inputs: dict[str, Any],
    *,
    tags: list[str] | None = None,
    metadata: dict[str, str] | None = None,
) -> Any:
    """Invoke a LangChain ``Runnable`` with optional LangSmith tracing.

    The chain's behavior is completely unchanged — ``invoke_with_tracing``
    simply passes a ``LangChainTracer`` through the chain's ``config``
    argument when tracing is enabled.

    If ``get_langsmith_callback()`` returns ``None`` (tracing disabled or
    misconfigured), the chain runs without any callback — identical to a
    plain ``chain.invoke(inputs)`` call.

    Args:
        chain:    Any LangChain ``Runnable`` (LCEL chain, LLM, etc.).
        inputs:   Input dict passed to ``chain.invoke``.
        tags:     Optional tags forwarded to the LangSmith trace.
        metadata: Optional key-value metadata forwarded to the LangSmith trace.

    Returns:
        Whatever ``chain.invoke`` returns.
    """
    callback = get_langsmith_callback(tags=tags, metadata=metadata)
    callbacks: list[Any] = [callback] if callback is not None else []
    return chain.invoke(inputs, config={"callbacks": callbacks})
