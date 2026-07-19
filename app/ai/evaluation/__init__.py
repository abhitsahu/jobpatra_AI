"""Evaluation package — Phase 10.7.

Provides infrastructure for evaluating AI explanation quality.

JobPatra AI is an ATS Analyzer and AI Resume Advisor.
Evaluation measures:
  - Explanation quality (clarity, correctness, ATS relevance)
  - Suggestion quality (usefulness, specificity, actionability)

Evaluation does NOT measure:
  - Resume rewrite quality (no rewrite feature)
  - HITL approval accuracy (no HITL workflow)

Modules
-------
run_eval.py        — Evaluation harness: loads datasets, runs evaluators, prints report.
eval_datasets/     — JSONL dataset files with resume+JD+expected explanation+suggestions.
"""
