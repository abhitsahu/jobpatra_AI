"""Explain-score chain — LCEL pipeline: Prompt → LiteLLM → Structured Output.

Pipeline
---------------------
1. Input Guardrail — validate resume + JD text before any LLM call.
2. ``EXPLAIN_SCORE_PROMPT_V1`` formats the ATS report into a chat prompt.
3. ``get_chat_model()`` calls LiteLLM (routed to Claude by default).
4. ``with_structured_output(ATSExplanation)`` parses the response.
5. ``run_with_retry`` invokes the chain (with LangSmith tracing) and runs
   output validation.  One retry is attempted on validation failure.

This chain does NOT:
  - Retry on network / provider / timeout errors.
  - Stream output.

Inputs (passed as keyword arguments to ``run_explain_score``)
-------------------------------------------------------------
See ``build_chain_inputs()`` for the full list.  All values come from
the deterministic ``ATSAnalyzeResponse`` produced by the ATS engine.
"""

from __future__ import annotations

from app.ai.guardrails.input_guardrails import validate_all as validate_input
from app.ai.guardrails.retry_policy import run_with_retry
from app.ai.prompts.explain_score_v2 import EXPLAIN_SCORE_PROMPT_V2
from app.ai.providers.litellm_client import get_chat_model
from app.analysis.normalization.jd_preprocessor import preprocess_jd
from app.middleware.request_id_middleware import get_request_id
from app.schemas.ai import ATSExplanation
from app.schemas.ats import ATSAnalyzeResponse


def build_chain_inputs(
    response: ATSAnalyzeResponse,
    jd_text: str,
    resume_text: str
) -> dict[str, object]:
    """Convert an ``ATSAnalyzeResponse`` into the prompt template variables.

    Args:
        response: The completed deterministic ATS report.
        jd_text:  The original job description text (for context injection).
        resume_text: The actual parsed resume text.

    Returns:
        A dict whose keys match every ``{variable}`` in the prompt template.
    """
    matched_kw = ", ".join(k.keyword for k in response.matched_keywords) or "None"
    missing_kw = ", ".join(response.missing_keywords) or "None"
    matched_sk = ", ".join(response.matched_skills) or "None"
    missing_sk = ", ".join(response.missing_skills) or "None"
    certs = ", ".join(response.education_summary.certifications) or "None"

    return {
        "overall_score": round(response.overall_score, 1),
        "keyword_score": round(response.keyword_score, 1),
        "experience_score": round(response.experience_score, 1),
        "skills_score": round(response.skills_score, 1),
        "education_score": round(response.education_score, 1),
        "summary_score": round(response.summary_score, 1),
        "formatting_score": round(response.formatting_score, 1),
        "matched_keywords": matched_kw,
        "missing_keywords": missing_kw,
        "matched_skills": matched_sk,
        "missing_skills": missing_sk,
        "exp_total_entries": response.experience_summary.total_entries,
        "exp_total_years": round(response.experience_summary.total_years, 1),
        "exp_has_metrics": response.experience_summary.has_metrics,
        "edu_highest_degree": response.education_summary.highest_degree or "Not detected",
        "edu_certifications": certs,
        "jd_context": preprocess_jd(jd_text),
        "resume_text": resume_text,
    }


def run_explain_score(
    response: ATSAnalyzeResponse,
    jd_text: str,
    resume_text: str | None = None
) -> ATSExplanation:
    """Execute the explain-score LCEL chain with guardrails and LangSmith tracing.

    Flow:
        1. Input guardrail — raises ``InvalidInputError`` if text is invalid.
        2. Build chain: Prompt | LiteLLM | with_structured_output.
        3. ``run_with_retry`` invokes the chain, validates output, retries once
           on ``OutputValidationError``, raises ``AIGenerationError`` on failure.

    Args:
        response: Completed deterministic ATS report (scores are final).
        jd_text:  Job description text used as context in the prompt.

    Returns:
        A validated ``ATSExplanation``.

    Raises:
        InvalidInputError: If resume or JD text fails input validation.
            The LLM is never called in this case.
        AIGenerationError: If output validation fails after one retry.
            The caller (``ats_service.py``) catches this and returns
            ``ai_status='unavailable'``.
        Exception: Any network / provider / timeout error propagates.
            The caller catches these with a broad except.
    """
    # ── Input guardrail ──────────────────────────────────────────────────────
    # Raises InvalidInputError if text is empty, too large, or injected.
    # The LLM is never reached in that case.
    if resume_text is None:
        resume_text = _extract_resume_text(response)
    validate_input(resume_text, jd_text)

    # ── Build chain ──────────────────────────────────────────────────────────
    llm = get_chat_model()
    inputs = build_chain_inputs(response, jd_text, resume_text)

    from datetime import datetime, timezone
    rid = get_request_id()
    metadata: dict[str, str] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if rid:
        metadata["request_id"] = rid

    # ── Invoke with tracing + output validation + retries ─────────────────────
    result = run_with_retry(
        prompt=EXPLAIN_SCORE_PROMPT_V2,
        llm=llm,
        inputs=inputs,
        resume_text=resume_text,
        jd_text=jd_text,
        tags=["explain_score_v2"],
        metadata=metadata,
    )
    return result


def _extract_resume_text(response: ATSAnalyzeResponse) -> str:
    """Extract a best-effort resume text proxy from the ATS response.

    The input guardrail needs text to check for size and injection.
    At this point the resume has already been parsed and scored; we
    reconstruct a representative string from the matched/missing keywords
    and scores.  The actual full text is no longer available here (it was
    consumed by the parser earlier in the pipeline).

    The guardrail is therefore applied to the JD text (most important for
    injection) and a synthetic resume summary string.

    Note: The definitive resume size check happens via ``validate_combined_length``
    which receives both lengths from the caller.
    """
    # Build a readable proxy string from the response data.
    # This is enough to catch injection patterns embedded in skill/keyword fields.
    parts = [
        f"overall_score:{response.overall_score}",
        f"keywords:{','.join(k.keyword for k in response.matched_keywords)}",
        f"skills:{','.join(response.matched_skills)}",
    ]
    return " ".join(parts)
