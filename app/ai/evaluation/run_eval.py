"""Evaluation harness.

Evaluates the quality of AI explanations and suggestions produced by
the explain-score chain.

Evaluation scope (JobPatra MVP — AI Advisor only)
-------------------------------------------------
ATS Explanation quality:
     - Clarity: Is the explanation easy to understand?
     - Correctness: Does it accurately describe why the score was given?
     - ATS relevance: Does it reference actual keywords / skills from the data?

Suggestion quality:
     - Usefulness: Does each suggestion target a real gap?
     - Specificity: Does each suggestion reference a specific keyword, skill, or section?
     - Actionability: Can the user act on the suggestion without AI assistance?

NOT evaluated (intentionally removed from MVP):
     - Resume rewrite quality (Phase 10.4 skipped — no rewrite feature)
     - HITL approval quality (Phase 10.5 skipped — no HITL workflow)

Dataset format (eval_datasets/*.jsonl)
--------------------------------------
Each line is a JSON object:

  {
    "id": "sample_001",
    "resume": "...",
    "job_description": "...",
    "expected": {
      "strengths": ["..."],
      "weaknesses": ["..."],
      "suggestions": ["..."],
      "summary": "..."
    },
    "metadata": {
      "role": "Software Engineer",
      "industry": "Tech"
    }
  }

Usage
-----
Run the evaluator against a live server:

  uv run python -m app.ai.evaluation.run_eval \\
    --dataset app/ai/evaluation/eval_datasets/sample_dataset.jsonl \\
    --base-url http://localhost:8000 \\
    --api-key <INTERNAL_API_KEY>

Or evaluate offline against cached LLM responses:

  uv run python -m app.ai.evaluation.run_eval \\
    --dataset app/ai/evaluation/eval_datasets/sample_dataset.jsonl \\
    --offline

This module does NOT:
  - Evaluate resume rewrite quality.
  - Call the LLM directly — it calls the FastAPI endpoint.
  - Require LangSmith (it produces a local report).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Evaluation metrics


@dataclass
class ExplanationEvalResult:
    """Evaluation result for a single dataset entry."""

    sample_id: str
    has_strengths: bool = False
    has_weaknesses: bool = False
    has_suggestions: bool = False
    has_summary: bool = False
    suggestions_reference_jd_keywords: bool = False
    all_sections_present: bool = False
    ai_status: str = "unavailable"
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if the response meets minimum quality bar."""
        return (
            self.has_strengths
            and self.has_weaknesses
            and self.has_suggestions
            and self.has_summary
            and self.all_sections_present
        )


# Evaluators (heuristic — no LLM needed)


def evaluate_response(
    sample: dict[str, Any],
    response: dict[str, Any],
) -> ExplanationEvalResult:
    """Evaluate a single API response against the expected dataset entry.

    This evaluator is heuristic (no LLM). It checks structural completeness
    and keyword overlap with the job description.

    Args:
        sample:   Dataset entry with ``id``, ``resume``, ``job_description``, ``expected``.
        response: Raw API response dict from ``POST /v1/ats/analyze``.

    Returns:
        ``ExplanationEvalResult`` with pass/fail flags and notes.
    """
    result = ExplanationEvalResult(sample_id=sample.get("id", "unknown"))
    result.ai_status = response.get("ai_status", "unavailable")

    if result.ai_status != "ok":
        result.notes.append("AI status is not 'ok' — explanation not generated.")
        return result

    ai = response.get("ai_explanation") or {}
    missing_kws = [k.lower() for k in response.get("missing_keywords", [])]

    result.has_strengths = bool(ai.get("strengths"))
    result.has_weaknesses = bool(ai.get("weaknesses"))
    result.has_suggestions = bool(ai.get("suggestions"))
    result.has_summary = bool(ai.get("summary", "").strip())

    sections = {s["section"] for s in (ai.get("section_explanations") or [])}
    expected_sections = {"Keywords", "Experience", "Skills", "Education", "Summary", "Formatting"}
    result.all_sections_present = expected_sections.issubset(sections)
    if not result.all_sections_present:
        missing = expected_sections - sections
        result.notes.append(f"Missing sections in ai_explanation: {missing}")

    # Check if at least one suggestion references a JD-missing keyword
    suggestion_text = " ".join(ai.get("suggestions") or []).lower()
    if missing_kws:
        result.suggestions_reference_jd_keywords = any(
            kw in suggestion_text for kw in missing_kws
        )
        if not result.suggestions_reference_jd_keywords:
            result.notes.append(
                "No suggestion references a missing JD keyword. "
                f"Missing JD keywords: {missing_kws[:5]}"
            )

    return result


# Dataset loader


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load evaluation dataset from a JSONL file.

    Args:
        path: Path to a ``.jsonl`` file where each line is a JSON object.

    Returns:
        List of dataset entry dicts.
    """
    samples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping malformed line {lineno}: {exc}", file=sys.stderr)
    return samples


# Report printer


def print_report(results: list[ExplanationEvalResult]) -> None:
    """Print a human-readable evaluation report to stdout."""
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    print("\n" + "=" * 60)
    print("JobPatra AI Explanation Evaluation Report")
    print("=" * 60)
    print(f"Total samples:  {len(results)}")
    print(f"Passed:         {len(passed)} ({100 * len(passed) / max(len(results), 1):.1f}%)")
    print(f"Failed:         {len(failed)}")
    print("=" * 60)

    if failed:
        print("\nFailed samples:")
        for r in failed:
            print(f"  [{r.sample_id}] ai_status={r.ai_status}")
            for note in r.notes:
                print(f"    ↳ {note}")

    print()


# CLI


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_eval",
        description="Evaluate JobPatra AI explanation and suggestion quality.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to evaluation dataset (.jsonl)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Offline mode: evaluate from cached response files (not implemented yet)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the FastAPI service (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Internal API key (INTERNAL_API_KEY env var used if not provided)",
    )
    return parser


def main() -> None:
    """CLI entry point for the evaluation harness."""
    parser = _build_parser()
    args = parser.parse_args()

    dataset_path: Path = args.dataset
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    samples = load_dataset(dataset_path)
    if not samples:
        print("[ERROR] Dataset is empty.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(samples)} samples from {dataset_path}")

    if args.offline:
        print("[INFO] Offline mode: no API calls. Evaluate cached responses manually.")
        print("[INFO] Full offline evaluation not yet implemented.")
        sys.exit(0)

    try:
        import httpx  # optional; only needed for live evaluation
    except ImportError:
        print("[ERROR] httpx is required for live evaluation. Run: uv add httpx", file=sys.stderr)
        sys.exit(1)

    import os

    api_key = args.api_key or os.environ.get("INTERNAL_API_KEY", "")
    if not api_key:
        print("[ERROR] --api-key or INTERNAL_API_KEY env var is required.", file=sys.stderr)
        sys.exit(1)

    results: list[ExplanationEvalResult] = []

    with httpx.Client(base_url=args.base_url, timeout=60) as client:
        for sample in samples:
            sample_id = sample.get("id", "?")
            print(f"  Evaluating [{sample_id}]...", end=" ")

            try:
                resp = client.post(
                    "/v1/ats/analyze",
                    json={
                        "resume": {"text": sample["resume"]},
                        "job_description": {"text": sample["job_description"]},
                    },
                    headers={"X-Internal-API-Key": api_key},
                )
                resp.raise_for_status()
                response_body = resp.json()
                eval_result = evaluate_response(sample, response_body)
                results.append(eval_result)
                status = "✅" if eval_result.passed else "❌"
                print(f"{status} (ai_status={eval_result.ai_status})")
            except Exception as exc:
                print(f"❌ ERROR: {exc}")
                results.append(ExplanationEvalResult(sample_id=sample_id, notes=[str(exc)]))

    print_report(results)

    # Exit 1 if any sample failed — useful for CI integration
    if any(not r.passed for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
