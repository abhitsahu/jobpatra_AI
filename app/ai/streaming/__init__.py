"""Streaming package — Phase 10.6.

Provides SSE (Server-Sent Events) streaming for the AI explanation pipeline.

Modules
-------
stream_events.py  — Typed event dataclasses that describe pipeline progress.
sse_encoder.py    — Serialises events into the SSE wire format.

This package does NOT:
  - Implement any ATS logic.
  - Implement any LLM logic.
  - Define FastAPI routes.
"""
