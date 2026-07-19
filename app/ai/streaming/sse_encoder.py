"""SSE encoder.

Serialises stream event dataclasses into the SSE wire format.

SSE format (RFC 8895)
---------------------
Each event is a block of one or more ``field: value`` lines terminated
by a blank line.  The two mandatory fields are:

  event: <event-name>
  data:  <JSON string>

Example output for a single event:

  event: ats_running
  data: {"message": "ATS Engine running…"}

  (blank line terminates the event)

Design
------
* ``encode(event)`` → str   — serialise a single stream event.
* ``encode_error(message)`` → str — shortcut for terminal errors.
* All JSON is compact (``separators=(","":"")``).
* The encoder is a pure, stateless module — no side effects.

This file does NOT:
  - Import from FastAPI.
  - Import from LangChain.
  - Maintain state.
"""

from __future__ import annotations

import json
from typing import Any


def encode(event: Any) -> str:
    """Serialise a stream event dataclass into SSE wire format.

    The event dataclass must expose:
      - ``.event`` (str) — the SSE event type name.
      - ``.data``  (dict) — the payload to JSON-encode.

    Args:
        event: Any stream event dataclass from ``stream_events.py``.

    Returns:
        A string containing the SSE-formatted event, terminated by a
        double newline (``\\n\\n``) as required by the SSE spec.
    """
    event_name = event.event.value if hasattr(event.event, "value") else str(event.event)
    data_json = json.dumps(event.data, separators=(",", ":"), ensure_ascii=False)
    return f"event: {event_name}\ndata: {data_json}\n\n"


def encode_error(message: str, code: str = "INTERNAL_ERROR") -> str:
    """Produce an SSE error event from a plain message string.

    Use this when you have an error string rather than an ``ErrorEvent``
    dataclass — avoids importing ``stream_events`` in tight error paths.

    Args:
        message: Human-readable error description.
        code:    Machine-readable error code (default ``INTERNAL_ERROR``).

    Returns:
        A string containing the SSE-formatted error event.
    """
    payload = json.dumps({"code": code, "message": message}, separators=(",", ":"))
    return f"event: error\ndata: {payload}\n\n"


def encode_heartbeat() -> str:
    """Produce an SSE comment line to keep the connection alive.

    SSE comments (lines starting with ``:`` ) are ignored by the browser
    EventSource but prevent proxies and load balancers from timing out
    long-running connections.

    Returns:
        A single SSE comment line.
    """
    return ": heartbeat\n\n"
