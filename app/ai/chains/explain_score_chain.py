"""Explain-score chain — LCEL pipeline: Prompt → LiteLLM → Structured Output.

Pipeline
---------------------
1. Input Guardrail — validate resume + JD text before any LLM call.
2. ``EXPLAIN_SCORE_PROMPT_V2`` formats the ATS report into a chat prompt.
3. ``get_chat_model()`` calls LiteLLM.
4. ``invoke_with_tracing`` invokes the chain (with LangSmith tracing).
5. ``parse_and_repair_response`` cleans and validates output locally.

This chain does NOT:
  - Retry on network / provider / timeout errors.
  - Retry on validation failures.
  - Stream output.
"""

from __future__ import annotations

from app.ai.chains.base_chain import invoke_with_tracing
from app.ai.guardrails.input_guardrails import validate_all as validate_input
from app.ai.guardrails.output_guardrails import parse_and_repair_response
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


# [ignoring loop detection]
def run_explain_score(
    response: ATSAnalyzeResponse,
    jd_text: str,
    resume_text: str | None = None
) -> ATSExplanation:
    """Execute the explain-score LCEL chain with guardrails and LangSmith tracing.

    Flow:
        1. Input guardrail — raises ``InvalidInputError`` if text is invalid.
        2. Build chain: Prompt | LiteLLM.
        3. Invoke chain exactly once with tracing.
        4. Validate and repair response locally.

    Args:
        response: Completed deterministic ATS report (scores are final).
        jd_text:  Job description text used as context in the prompt.

    Returns:
        A validated ``ATSExplanation``.

    Raises:
        InvalidInputError: If resume or JD text fails input validation.
        AIGenerationError: If output validation/repair fails.
        Exception: Any network / provider / timeout error propagates.
    """
    # ── Input guardrail ──────────────────────────────────────────────────────
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

    # ── Invoke raw LLM chain ──────────────────────────────────────────────────
    raw_llm_chain = EXPLAIN_SCORE_PROMPT_V2 | llm
    raw_message = invoke_with_tracing(
        raw_llm_chain,
        inputs,
        tags=["explain_score_v2"],
        metadata=metadata,
    )

    # ── Validate and local repair ─────────────────────────────────────────────
    return parse_and_repair_response(raw_message.content)


def _extract_resume_text(response: ATSAnalyzeResponse) -> str:
    """Extract a best-effort resume text proxy from the ATS response.

    Used by the input guardrail to check size and injection patterns.
    """
    parts = [
        f"overall_score:{response.overall_score}",
        f"keywords:{','.join(k.keyword for k in response.matched_keywords)}",
        f"skills:{','.join(response.matched_skills)}",
    ]
    return " ".join(parts)
