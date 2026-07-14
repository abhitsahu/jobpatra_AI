"""ATS Service — orchestrator for the deterministic ATS pipeline.

This module is the ONLY file that understands how to wire the analysis
modules together.  It contains no ATS logic of its own.

Pipeline
--------
1. Parse resume (file or raw text)
2. Clean + normalize resume text and JD text
3. Split resume into sections
4. Extract from resume: keywords, skills, experience, education
5. Extract from JD: keywords, skills
6. Keyword matching (Exact → Synonym → Fuzzy)
7. Semantic matching skipped — no provider configured at this phase
8. Calculate ATS scores via scoring_engine
9. Build and return ATSAnalyzeResponse

Every step delegates to an existing analysis module.
No ATS logic lives here.

This module does NOT:
  - implement any matching logic
  - implement any scoring logic
  - call AI, LangChain, or external services
  - import from FastAPI routes
"""

from __future__ import annotations

import time

from app.analysis.extraction import (
    education_extractor,
    experience_extractor,
    keyword_extractor,
    skill_extractor,
)
from app.analysis.extraction.education_extractor import EducationExtractionResult
from app.analysis.extraction.experience_extractor import ExperienceEntry
from app.analysis.matching import keyword_matcher
from app.analysis.normalization import section_splitter, text_cleaner
from app.analysis.normalization.jd_normalizer import normalize as normalize_jd
from app.analysis.parsers import parser_factory
from app.analysis.scoring import scoring_engine
from app.core.errors import ValidationError
from app.core.logging import logger
from app.middleware.request_id_middleware import get_request_id
from app.schemas.ats import (
    ATSAnalyzeRequest,
    ATSAnalyzeResponse,
    EducationSummarySchema,
    ExperienceSummarySchema,
    MatchedKeywordSchema,
)


def analyze(request: ATSAnalyzeRequest) -> ATSAnalyzeResponse:
    """Execute the full deterministic ATS pipeline.

    Orchestrates every analysis module in sequence and returns a complete
    ``ATSAnalyzeResponse``.  Each step calls exactly one existing module.

    Args:
        request: Validated ``ATSAnalyzeRequest`` containing the resume and JD.

    Returns:
        ``ATSAnalyzeResponse`` with all scores, matched/missing keywords,
        skill coverage, and extracted metadata.

    Raises:
        ValidationError: If the resume input is neither text nor file.
        UnparsableDocumentError: If a file resume cannot be parsed to text.
    """
    rid = get_request_id()
    _log = lambda msg: logger.info("[%s] %s", rid[:8] if rid else "-", msg)  # noqa: E731

    start = time.perf_counter()
    _log("ATS pipeline started")

    # ── Step 1: Parse resume → plain text ───────────────────────────────────
    _log("Parsing resume")
    resume_raw = _parse_resume(request)

    # ── Step 2: Normalize texts ─────────────────────────────────────────────
    _log("Normalizing text")
    resume_clean = text_cleaner.clean(resume_raw)
    jd_clean = normalize_jd(request.job_description.text)

    # ── Step 3: Split resume into sections ──────────────────────────────────
    _log("Splitting sections")
    sections = section_splitter.split(resume_clean)

    # ── Step 4: Extract from resume ─────────────────────────────────────────
    _log("Extracting resume data")
    resume_keywords = keyword_extractor.extract(resume_clean)
    resume_skills_result = skill_extractor.extract(resume_clean)
    resume_skills = [m.canonical for m in resume_skills_result.skills]

    exp_text = sections.experience or ""
    experience_entries: list[ExperienceEntry] = experience_extractor.extract(exp_text)

    edu_text = sections.education or ""
    education_result: EducationExtractionResult = education_extractor.extract(edu_text)

    # ── Step 5: Extract from JD ─────────────────────────────────────────────
    _log("Extracting JD data")
    jd_keywords = keyword_extractor.extract(jd_clean)
    jd_skills_result = skill_extractor.extract(jd_clean)
    required_skills = [m.canonical for m in jd_skills_result.skills]

    # ── Step 6: Keyword matching (deterministic — no semantic provider) ──────
    _log("Matching keywords")
    match_result = keyword_matcher.match(
        resume_keywords=resume_keywords,
        jd_keywords=jd_keywords,
    )

    # ── Step 7: Score ────────────────────────────────────────────────────────
    _log("Calculating scores")
    report = scoring_engine.score(
        match_result=match_result,
        experience_entries=experience_entries,
        resume_skills=resume_skills,
        required_skills=required_skills,
        education_result=education_result,
        sections=sections,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    _log(f"ATS pipeline completed in {elapsed_ms:.1f}ms — overall score: {report.overall_score}")

    # ── Step 8: Build response ───────────────────────────────────────────────
    return _build_response(
        report=report,
        match_result=match_result,
        resume_skills=resume_skills,
        required_skills=required_skills,
        experience_entries=experience_entries,
        education_result=education_result,
        processing_time_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_resume(request: ATSAnalyzeRequest) -> str:
    """Convert the resume input to plain text.

    Args:
        request: Validated analyze request.

    Returns:
        Plain text of the resume.

    Raises:
        ValidationError: If neither text nor file bytes are provided.
    """
    if request.resume.is_text_mode():
        return parser_factory.parse_text(request.resume.text)  # type: ignore[arg-type]

    if request.resume.is_file_mode():
        return parser_factory.parse(
            filename=request.resume.filename,  # type: ignore[arg-type]
            file_bytes=request.resume.file_bytes,  # type: ignore[arg-type]
        )

    raise ValidationError("Resume must provide either 'text' or 'filename'+'file_bytes'.")


def _build_response(
    *,
    report: scoring_engine.ATSReport,
    match_result: keyword_matcher.MatchResult,
    resume_skills: list[str],
    required_skills: list[str],
    experience_entries: list[ExperienceEntry],
    education_result: EducationExtractionResult,
    processing_time_ms: float,
) -> ATSAnalyzeResponse:
    """Assemble the final API response from all pipeline outputs.

    Args:
        report: Scored ATS report from scoring_engine.
        match_result: Keyword match result.
        resume_skills: Canonical skill list from the resume.
        required_skills: Canonical skill list from the JD.
        experience_entries: Parsed experience entries.
        education_result: Parsed education result.
        processing_time_ms: Total pipeline wall-clock time.

    Returns:
        Fully populated ``ATSAnalyzeResponse``.
    """
    matched_kw = [
        MatchedKeywordSchema(
            keyword=m.keyword,
            matchType=m.matchType,
            similarity=m.similarity,
        )
        for m in match_result.matched
    ]

    # Skill coverage: intersection of resume and JD skills (case-insensitive)
    required_lower = {s.lower() for s in required_skills}
    resume_lower_map = {s.lower(): s for s in resume_skills}
    matched_skills = [
        resume_lower_map[s]
        for s in resume_lower_map
        if s in required_lower
    ]
    missing_skills = [
        s for s in required_skills
        if s.lower() not in {ms.lower() for ms in matched_skills}
    ]

    # Experience summary
    exp_summary = ExperienceSummarySchema(
        total_entries=len(experience_entries),
        total_years=sum(
            e.duration_years for e in experience_entries
            if e.duration_years is not None
        ),
        has_metrics=any(e.metrics for e in experience_entries),
    )

    # Education summary
    highest = None
    if education_result.entries:
        # Pick the entry with the longest degree string as a rough proxy
        highest = max(
            (e.degree for e in education_result.entries if e.degree),
            key=len,
            default=None,
        )
    edu_summary = EducationSummarySchema(
        highest_degree=highest,
        certifications=education_result.certifications,
    )

    return ATSAnalyzeResponse(
        overall_score=report.overall_score,
        keyword_score=report.keyword_score,
        experience_score=report.experience_score,
        skills_score=report.skills_score,
        education_score=report.education_score,
        summary_score=report.summary_score,
        formatting_score=report.formatting_score,
        matched_keywords=matched_kw,
        missing_keywords=match_result.missing,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        experience_summary=exp_summary,
        education_summary=edu_summary,
        processing_time_ms=round(processing_time_ms, 2),
    )
