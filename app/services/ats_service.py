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
9. Build ATSAnalyzeResponse (deterministic)
10. AI explain-score chain with guardrails
    a. Input guardrail — raises InvalidInputError (propagated → HTTP 422)
    b. Run chain with output validation
    c. AIGenerationError → ai_status='unavailable' (HTTP 200, no crash)
11. Return merged response

Every step delegates to an existing analysis module.
No ATS logic lives here.
"""

from __future__ import annotations

import os
import json
import time
import typing
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from app.analysis.extraction import (
    education_extractor,
    experience_extractor,
    keyword_extractor,
    skill_extractor,
)
from app.analysis.extraction.requirement_taxonomy import (
    classify_jd_requirements,
    fallback_jd_extraction,
    fallback_resume_extraction,
    resume_technical_evidence,
)
from app.analysis.extraction.education_extractor import EducationExtractionResult
from app.analysis.extraction.experience_extractor import ExperienceEntry
from app.analysis.matching import keyword_matcher
from app.analysis.normalization import section_splitter, text_cleaner
from app.analysis.normalization.jd_normalizer import normalize as normalize_jd
from app.analysis.parsers import parser_factory
from app.analysis.scoring import scoring_engine
from app.ai.chains.explain_score_chain import run_explain_score
from app.ai.chains.extract_entities_chain import (
    extract_jd_entities,
    extract_resume_entities,
)
from app.analysis.matching.semantic_matcher import get_shared_provider
from app.core.errors import AIGenerationError, InvalidInputError, ValidationError
from app.core.logging import logger
from app.core.config import settings
from app.middleware.request_id_middleware import get_request_id
from app.schemas.ats import (
    ATSAnalyzeRequest,
    ATSAnalyzeResponse,
    EducationSummarySchema,
    ExperienceSummarySchema,
    MatchedKeywordSchema,
)

def _get_rejected_tokens(text: str) -> list[tuple[str, str]]:
    import re
    from app.analysis.extraction.keyword_extractor import _STOP_WORDS
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9.#+\-_]*", text)
    rejected = []
    seen = set()
    whitelist = {"C++", "C#", "F#", "AWS", "SQL"}
    for t in tokens:
        t_clean = t.strip()
        if not t_clean or t_clean in seen:
            continue
        seen.add(t_clean)
        if len(t_clean) <= 1:
            rejected.append((t_clean, "Length <= 1"))
        elif t_clean.lower() in _STOP_WORDS:
            rejected.append((t_clean, "Stop word"))
        else:
            alphas = sum(1 for c in t_clean if c.isalpha())
            ratio = alphas / len(t_clean) if t_clean else 0.0
            if ratio < 0.5 and t_clean.upper() not in whitelist:
                rejected.append((t_clean, "Failed validation (alpha density < 0.5)"))
    return rejected


class PipelineDebugger:
    def __init__(self, filename: str | None, file_bytes: bytes | None, text_input: str | None, jd_text: str):
        self.filename = filename
        self.file_bytes = file_bytes
        self.text_input = text_input
        self.jd_text = jd_text
        self.enabled = os.environ.get("DEBUG_ATS_PIPELINE", "false").lower() == "true" or settings.DEBUG_ATS_PIPELINE
        self.timings = {}
        self.statuses = {}
        self.errors = {}

    def log(self, msg: str) -> None:
        if self.enabled:
            logger.info(msg)

    def log_stage_header(self, stage_name: str) -> None:
        self.log(f"\n==============================\n{stage_name.upper()}\n==============================")

    def start_stage(self, stage: str, display_name: str) -> float:
        self.log_stage_header(display_name)
        return time.perf_counter()

    def end_stage(self, stage: str, start_time: float, success: bool = True, error: Exception | None = None) -> None:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        self.timings[stage] = elapsed
        self.statuses[stage] = "SUCCESS" if success else "FAILED"
        if error:
            import traceback
            tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            self.errors[stage] = {
                "reason": str(error),
                "stack_trace": tb
            }
            self.log(f"\nFAILED\n\nReason:\n{error}\n\nStack Trace:\n{tb}")
        else:
            self.log(f"\nExecution Time: {elapsed:.1f} ms")

    def log_resume_upload(self) -> None:
        if not self.enabled:
            return
        self.log("\n========== RESUME UPLOAD ==========\n")
        self.log("Resume uploaded successfully\n")
        
        if self.file_bytes is not None:
            size_kb = len(self.file_bytes) / 1024.0
            size_str = f"{size_kb:.1f} KB"
            ext = os.path.splitext(self.filename)[1].lower() if self.filename else "unknown"
            
            mime_map = {
                ".pdf": "application/pdf",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".txt": "text/plain",
                ".csv": "text/csv",
                ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ".xls": "application/vnd.ms-excel",
                ".json": "application/json"
            }
            mime = mime_map.get(ext, "application/octet-stream")
            
            temp_dir_map = {
                ".pdf": ".temp_pdf_parser",
                ".docx": ".temp_docx_parser",
                ".txt": ".temp_txt_parser",
                ".csv": ".temp_csv_parser",
                ".xlsx": ".temp_excel_parser",
                ".xls": ".temp_excel_parser",
                ".json": ".temp_json_parser"
            }
            temp_dir = temp_dir_map.get(ext, ".temp_parser")
            temp_path = os.path.join(os.getcwd(), temp_dir, f"resume_*_temp{ext}")
            
            self.log(f"Filename:\n{self.filename}\n")
            self.log(f"Size:\n{size_str}\n")
            self.log(f"Extension:\n{ext}\n")
            self.log(f"Mime:\n{mime}\n")
            self.log(f"Temporary Path:\n{temp_path}\n")
        else:
            size_str = f"{len(self.text_input or '')} chars"
            self.log(f"Source:\nText Area (Raw Text Mode)\n")
            self.log(f"Size:\n{size_str}\n")
            self.log("Extension:\nN/A\n")
            self.log("Mime:\ntext/plain\n")
            
        self.log(f"Upload Time:\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.statuses["Resume Upload"] = "SUCCESS"

    def log_job_description(self) -> None:
        if not self.enabled:
            return
        self.log("\n========== JOB DESCRIPTION ==========\n")
        self.log(f"Source:\nText Area\n")
        self.log(f"Characters:\n{len(self.jd_text)}\n")
        self.log(f"Words:\n{len(self.jd_text.split())}\n")
        self.log(f"Preview (first 500 chars):\n{self.jd_text[:500]}\n")
        self.statuses["Job Description"] = "SUCCESS"

    def log_parser(self, resume_raw: str, elapsed_ms: float) -> None:
        if not self.enabled:
            return
        ext = os.path.splitext(self.filename)[1].lower() if self.filename else "N/A"
        loader_map = {
            ".pdf": "DirectoryLoader + PyMuPDFLoader",
            ".docx": "Docx2txtLoader",
            ".txt": "TextLoader",
            ".csv": "CSVLoader",
            ".xlsx": "UnstructuredExcelLoader",
            ".xls": "UnstructuredExcelLoader",
            ".json": "JSONLoader"
        }
        loader = loader_map.get(ext, "Raw Text Input")
        
        pages = resume_raw.split("\n\n")
        page_count = len(pages)
        
        self.log(f"Loader:\n{loader}\n")
        self.log(f"Pages:\n{page_count}\n")
        self.log(f"Characters extracted:\n{len(resume_raw)}\n")
        self.log(f"Words extracted:\n{len(resume_raw.split())}\n")
        
        for i, page_text in enumerate(pages):
            self.log(f"Page {i+1}:\n{len(page_text)} chars\n")
            
        self.log(f"Metadata:\nFilename: {self.filename or 'N/A'}\n")
        self.log(f"Page count:\n{page_count}\n")
        self.log(f"Preview of first 1000 characters:\n{resume_raw[:1000]}\n")
        self.log(f"Preview of last 500 characters:\n{resume_raw[-500:] if len(resume_raw) >= 500 else resume_raw}\n")
        self.log(f"Parser execution time:\n{elapsed_ms:.1f} ms\n")

    def log_normalizer(self, raw_text: str, cleaned_text: str, elapsed_ms: float) -> None:
        if not self.enabled:
            return
        import re
        self.log(f"Characters before normalization:\n{len(raw_text)}\n")
        self.log(f"Words before normalization:\n{len(raw_text.split())}\n")
        self.log(f"Characters after normalization:\n{len(cleaned_text)}\n")
        self.log(f"Words after normalization:\n{len(cleaned_text.split())}\n")
        
        unicode_removed = sorted(list(set(c for c in raw_text if ord(c) > 127 and c not in cleaned_text)))
        self.log("Unicode artifacts removed:")
        if unicode_removed:
            for char in unicode_removed:
                self.log(f"- {char}")
        else:
            self.log("None detected")
        self.log("")
        
        lines_removed = len(raw_text.splitlines()) - len(cleaned_text.splitlines())
        empty_lines_input = sum(1 for line in raw_text.splitlines() if not line.strip())
        empty_lines_output = sum(1 for line in cleaned_text.splitlines() if not line.strip())
        empty_removed = max(0, empty_lines_input - empty_lines_output)
        control_chars = len(re.findall(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", raw_text))
        
        self.log(f"Number of lines removed:\n{lines_removed}\n")
        self.log(f"Number of empty lines removed:\n{empty_removed}\n")
        self.log(f"Number of control characters removed:\n{control_chars}\n")
        
        self.log(f"First 1000 chars:\n{cleaned_text[:1000]}\n")
        self.log(f"Last 500 chars:\n{cleaned_text[-500:] if len(cleaned_text) >= 500 else cleaned_text}\n")

    def log_section_splitter(self, sections: typing.Any, elapsed_ms: float) -> None:
        if not self.enabled:
            return
        import dataclasses
        fields_list = dataclasses.fields(sections)
        for f in fields_list:
            val = getattr(sections, f.name)
            if val is not None and val.strip():
                self.log(f"{f.name.capitalize()}\n\nCharacters:\n{len(val)}\n")
            else:
                self.log(f"WARNING\n\n{f.name.capitalize()} not detected\n")
                
        exp_text = sections.experience or ""
        self.log(f"========== EXPERIENCE ==========\n{exp_text}\n")

    def log_entity_extraction(
        self,
        sections: typing.Any,
        experience_entries: list[typing.Any],
        education_result: typing.Any,
        resume_skills: list[str],
        elapsed_ms: float
    ) -> None:
        if not self.enabled:
            return
        summary_detected = "YES" if (sections.summary and sections.summary.strip()) else "NO"
        self.log(f"Summary\n\nDetected:\n{summary_detected}\n")
        
        self.log(f"Experience\n\nEntries:\n{len(experience_entries)}\n")
        for entry in experience_entries:
            self.log(
                f"Company:\n{entry.company or 'N/A'}\n\n"
                f"Role:\n{entry.title or 'N/A'}\n\n"
                f"Start Date:\n{entry.start_date or 'N/A'}\n\n"
                f"End Date:\n{entry.end_date or 'N/A'}\n\n"
                f"Duration:\n{entry.duration_years or 'N/A'}\n"
            )
            
        edu_entries = getattr(education_result, "entries", [])
        highest_degree = edu_entries[0].degree if edu_entries else "None"
        uni = edu_entries[0].institution if edu_entries else "None"
        cgpa = edu_entries[0].cgpa if edu_entries else "None"
        
        self.log(
            f"Education\n\nHighest Degree:\n{highest_degree}\n\n"
            f"University:\n{uni}\n\n"
            f"CGPA:\n{cgpa}\n"
        )
        
        self.log("Skills\n\n" + "\n".join(resume_skills) + "\n")
        
        proj_count = 0
        if sections.projects:
            lines = [l.strip() for l in sections.projects.splitlines() if l.strip()]
            proj_count = len([l for l in lines if l.startswith("-") or l.startswith("•") or l.startswith("*")])
            if proj_count == 0:
                proj_count = len(lines)
        self.log(f"Projects\n\nProject count:\n{proj_count}\n")
        
        ach_count = sum(len(entry.metrics) for entry in experience_entries)
        self.log(f"Achievements\n\nAchievement count:\n{ach_count}\n")

    def log_keyword_matching(
        self,
        resume_clean: str,
        resume_keywords: list[str],
        jd_keywords: list[str],
        match_result: typing.Any,
        elapsed_ms: float
    ) -> None:
        if not self.enabled:
            return
        self.log("JD Skills\n\n" + "\n".join(jd_keywords) + "\n")
        self.log("Resume Skills\n\n" + "\n".join(resume_keywords) + "\n")
        
        matched_kws = [m.keyword for m in match_result.matched]
        self.log("Matched\n\n" + "\n".join(matched_kws) + "\n")
        self.log("Missing\n\n" + "\n".join(match_result.missing) + "\n")
        
        rejected_tokens = _get_rejected_tokens(resume_clean)
        self.log("Rejected tokens\n")
        if rejected_tokens:
            for tok, reason in rejected_tokens:
                self.log(f"{tok}\n\nReason:\n{reason}\n")
        else:
            self.log("None detected\n")
            
        exact = sum(1 for m in match_result.matched if m.matchType == "EXACT")
        synonym = sum(1 for m in match_result.matched if m.matchType == "SYNONYM")
        fuzzy = sum(1 for m in match_result.matched if m.matchType == "FUZZY")
        semantic = sum(1 for m in match_result.matched if m.matchType == "SEMANTIC")
        
        self.log(
            f"Exact matches:\n{exact}\n\n"
            f"Synonym matches:\n{synonym}\n\n"
            f"Fuzzy matches:\n{fuzzy}\n\n"
            f"Semantic matches:\n{semantic}\n"
        )

    def log_ats_score(
        self,
        report: typing.Any,
        match_result: typing.Any,
        experience_entries: list[typing.Any],
        education_result: typing.Any,
        sections: typing.Any,
        elapsed_ms: float
    ) -> None:
        if not self.enabled:
            return
        
        self.log(f"Keyword Score\n\n{report.keyword_score}\n\nReason:")
        self.log(f"{len(match_result.matched)} matched out of {len(match_result.matched) + len(match_result.missing)} total JD keywords\n")
        
        self.log(f"Experience Score\n\n{report.experience_score}\n\nReason:")
        if not experience_entries:
            self.log("No experience entries detected\n")
        else:
            total_years = sum(e.duration_years for e in experience_entries if e.duration_years is not None)
            self.log(f"Based on duration ({total_years:.1f} years), continuity ({len(experience_entries)} jobs), bullets, and metrics\n")
            
        self.log(f"Education Score\n\n{report.education_score}\n\nReason:")
        edu_entries = getattr(education_result, "entries", [])
        if not edu_entries:
            self.log("No education entries detected\n")
        else:
            deg = edu_entries[0].degree or "unknown"
            self.log(f"Based on degree level ({deg}) and certifications\n")
            
        self.log(f"Skills Score\n\n{report.skills_score}\n\nReason:")
        self.log(
            f"Matched {len(report.skill_match_result.matched)} of "
            f"{report.required_skill_count} required technical skills\n"
        )
        
        self.log(f"Formatting Score\n\n{report.formatting_score}\n\nReason:")
        import dataclasses
        missing_sections = []
        for f in dataclasses.fields(sections):
            val = getattr(sections, f.name)
            if not val or not val.strip():
                missing_sections.append(f.name.capitalize())
        if missing_sections:
            self.log(f"Some sections missing: {', '.join(missing_sections)}\n")
        else:
            self.log("All required sections detected\n")
            
        self.log(f"Summary Score\n\n{report.summary_score}\n\nReason:")
        if not sections.summary or not sections.summary.strip():
            self.log("No summary section detected\n")
        else:
            self.log(f"Summary length is {len(sections.summary)} chars\n")
            
        self.log(f"Overall Score\n\n{report.overall_score}\n")

    def log_ai(self, response: typing.Any, elapsed_ms: float, resume_text: str | None = None) -> None:
        if not self.enabled:
            return
        from app.ai.prompts.explain_score_v2 import EXPLAIN_SCORE_PROMPT_V2
        from app.ai.chains.explain_score_chain import build_chain_inputs
        from app.ai.providers.litellm_client import get_last_routing_info
        
        effective_resume_text = resume_text or self.text_input or ""
        inputs = build_chain_inputs(response, self.jd_text, effective_resume_text)
        prompt_text = EXPLAIN_SCORE_PROMPT_V2.format(**inputs)
        
        self.log(f"Prompt sent:\n{prompt_text[:1000]}...\n")
        
        routing_info = get_last_routing_info()
        
        self.log(
            f"LLM\n\n"
            f"Model:\n{routing_info.get('model', 'unknown')}\n\n"
            f"Tokens:\nPrompt: {routing_info.get('prompt_tokens', 0)} / Completion: {routing_info.get('completion_tokens', 0)} / Total: {routing_info.get('total_tokens', 0)}\n\n"
            f"Response:\n{str(response.ai_explanation)[:2000]}...\n\n"
            f"Time taken:\n{elapsed_ms:.1f} ms\n"
        )

    def log_pipeline_summary(self) -> None:
        if not self.enabled:
            return
        self.log("\n==============================\nPIPELINE SUMMARY\n==============================")
        
        stages = [
            "Resume Upload",
            "Job Description",
            "PDF Parser",
            "Normalizer",
            "Section Splitter",
            "Entity Extraction",
            "Keyword Matching",
            "ATS Calculation",
            "AI Explanation"
        ]
        
        for stage in stages:
            status = self.statuses.get(stage, "SUCCESS")
            self.log(f"{stage}\n{status}\n")
            if status == "FAILED":
                err = self.errors.get(stage, {})
                self.log(f"Reason:\n{err.get('reason', 'Unknown')}\n\nStack Trace:\n{err.get('stack_trace', '')}\n")


def _save_artifact(run_dir: Path | None, stage: str, data: typing.Any) -> None:
    """Helper to save debug artifacts for intermediate pipeline stages."""
    if not run_dir:
        return
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(data, str):
            file_path = run_dir / f"{stage}.txt"
            file_path.write_text(data, encoding="utf-8")
        else:
            file_path = run_dir / f"{stage}.json"
            if hasattr(data, "model_dump"):
                dict_data = data.model_dump()
            elif hasattr(data, "__dict__"):
                import dataclasses
                if dataclasses.is_dataclass(data):
                    dict_data = dataclasses.asdict(data)
                else:
                    dict_data = {k: v for k, v in data.__dict__.items() if not k.startswith("_")}
            elif isinstance(data, (list, tuple)):
                import dataclasses
                dict_data = []
                for item in data:
                    if hasattr(item, "model_dump"):
                        dict_data.append(item.model_dump())
                    elif dataclasses.is_dataclass(item):
                        dict_data.append(dataclasses.asdict(item))
                    elif hasattr(item, "__dict__"):
                        dict_data.append({k: v for k, v in item.__dict__.items() if not k.startswith("_")})
                    else:
                        dict_data.append(item)
            else:
                dict_data = data
            
            file_path.write_text(json.dumps(dict_data, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        logger.error(f"Failed to save debug artifact {stage}: {exc}")


def analyze(request: ATSAnalyzeRequest) -> ATSAnalyzeResponse:
    """Execute the full deterministic ATS pipeline."""
    rid = get_request_id()
    _log = lambda msg: logger.info("[%s] %s", rid[:8] if rid else "-", msg)  # noqa: E731

    start = time.perf_counter()
    _log("ATS pipeline started")

    save_enabled = os.environ.get("SAVE_ATS_ARTIFACTS", "false").lower() == "true"
    run_dir: Path | None = None
    if save_enabled:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = Path("ats_debug_artifacts") / f"run_{timestamp}"

    # Initialize debugger
    debugger = PipelineDebugger(
        filename=request.resume.filename,
        file_bytes=request.resume.file_bytes,
        text_input=request.resume.text,
        jd_text=request.job_description.text,
    )

    try:
        # ── Stage 1: Resume Upload ──────────────────────────────────────────
        debugger.log_resume_upload()

        # ── Stage 2: Job Description ────────────────────────────────────────
        debugger.log_job_description()

        # ── Stage 3: PDF Parser ─────────────────────────────────────────────
        t_start = debugger.start_stage("pdf_parser", "PDF Parser")
        try:
            _log("Parsing resume")
            resume_raw = _parse_resume(request)
            _save_artifact(run_dir, "step1_resume_raw", resume_raw)
            elapsed_stage = (time.perf_counter() - t_start) * 1000.0
            debugger.log_parser(resume_raw, elapsed_stage)
            debugger.end_stage("PDF Parser", t_start, success=True)
        except Exception as exc:
            debugger.end_stage("PDF Parser", t_start, success=False, error=exc)
            raise

        # ── Stage 4: Normalizer ─────────────────────────────────────────────
        t_start = debugger.start_stage("normalizer", "Normalizer")
        try:
            _log("Normalizing text")
            resume_clean = text_cleaner.clean(resume_raw)
            jd_clean = normalize_jd(request.job_description.text)
            _save_artifact(run_dir, "step2_resume_clean", resume_clean)
            _save_artifact(run_dir, "step2_jd_clean", jd_clean)
            elapsed_stage = (time.perf_counter() - t_start) * 1000.0
            debugger.log_normalizer(resume_raw, resume_clean, elapsed_stage)
            debugger.end_stage("Normalizer", t_start, success=True)
        except Exception as exc:
            debugger.end_stage("Normalizer", t_start, success=False, error=exc)
            raise

        # ── Stage 5: Section Splitter ───────────────────────────────────────
        t_start = debugger.start_stage("section_splitter", "Section Splitter")
        try:
            _log("Splitting sections")
            sections = section_splitter.split(resume_clean)
            _save_artifact(run_dir, "step3_sections", sections)
            elapsed_stage = (time.perf_counter() - t_start) * 1000.0
            debugger.log_section_splitter(sections, elapsed_stage)
            debugger.end_stage("Section Splitter", t_start, success=True)
        except Exception as exc:
            debugger.end_stage("Section Splitter", t_start, success=False, error=exc)
            raise

        # ── Stage 6: Entity Extraction ──────────────────────────────────────
        t_start = debugger.start_stage("entity_extraction", "Entity Extraction")
        try:
            _log("Extracting entity data using Hybrid AI (with fallback)")
            entities = _extract_entities_hybrid(resume_clean, jd_clean, logger_fn=_log)
            resume_keywords = entities.resume_keywords
            resume_skills = entities.resume_skills
            jd_keywords = entities.jd_keywords
            required_skills = entities.required_skills

            exp_text = sections.experience or ""
            experience_entries: list[ExperienceEntry] = experience_extractor.extract(exp_text)

            edu_text = sections.education or ""
            education_result: EducationExtractionResult = education_extractor.extract(edu_text)

            _save_artifact(run_dir, "step4_resume_keywords", resume_keywords)
            _save_artifact(run_dir, "step4_resume_skills", resume_skills)
            _save_artifact(run_dir, "step4_experience_entries", experience_entries)
            _save_artifact(run_dir, "step4_education_result", education_result)

            _save_artifact(run_dir, "step5_jd_keywords", jd_keywords)
            _save_artifact(run_dir, "step5_required_skills", required_skills)

            elapsed_stage = (time.perf_counter() - t_start) * 1000.0
            debugger.log_entity_extraction(sections, experience_entries, education_result, resume_skills, elapsed_stage)
            debugger.end_stage("Entity Extraction", t_start, success=True)
        except Exception as exc:
            debugger.end_stage("Entity Extraction", t_start, success=False, error=exc)
            raise

        # ── Stage 7: Keyword Matching ───────────────────────────────────────
        t_start = debugger.start_stage("keyword_matching", "Keyword Matching")
        try:
            _log("Matching keywords")
            match_result = keyword_matcher.match(
                resume_keywords=resume_keywords,
                jd_keywords=jd_keywords,
                embedding_provider=get_shared_provider(),
                semantic_threshold=0.60,
            )
            _save_artifact(run_dir, "step6_match_result", match_result)
            elapsed_stage = (time.perf_counter() - t_start) * 1000.0
            debugger.log_keyword_matching(resume_clean, resume_keywords, jd_keywords, match_result, elapsed_stage)
            debugger.end_stage("Keyword Matching", t_start, success=True)
        except Exception as exc:
            debugger.end_stage("Keyword Matching", t_start, success=False, error=exc)
            raise

        # ── Stage 8: ATS Calculation ────────────────────────────────────────
        t_start = debugger.start_stage("ats_calculation", "ATS Calculation")
        try:
            _log("Calculating scores")
            report = scoring_engine.score(
                match_result=match_result,
                experience_entries=experience_entries,
                resume_skills=resume_skills,
                required_skills=required_skills,
                education_result=education_result,
                sections=sections,
                required_years=entities.required_years,
                required_education_level=entities.required_education_level,
                embedding_provider=get_shared_provider(),
            )
            _save_artifact(run_dir, "step7_report", report)
            elapsed_stage = (time.perf_counter() - t_start) * 1000.0
            debugger.log_ats_score(report, match_result, experience_entries, education_result, sections, elapsed_stage)
            debugger.end_stage("ATS Calculation", t_start, success=True)
        except Exception as exc:
            debugger.end_stage("ATS Calculation", t_start, success=False, error=exc)
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _log(f"ATS pipeline completed in {elapsed_ms:.1f}ms — overall score: {report.overall_score}")

        # ── Step 8: Build deterministic response ────────────────────────────────
        response = _build_response(
            report=report,
            match_result=match_result,
            culture_signals=entities.culture_signals,
            extraction_mode=entities.extraction_mode,
            experience_entries=experience_entries,
            education_result=education_result,
            processing_time_ms=elapsed_ms,
        )
        _save_artifact(run_dir, "step8_response_deterministic", response)

        # ── Stage 9: AI Explanation ─────────────────────────────────────────
        t_start = debugger.start_stage("ai_explanation", "AI Explanation")
        try:
            _log("Running AI explain-score chain")
            ai_explanation = run_explain_score(
                response=response,
                jd_text=request.job_description.text,
                resume_text=resume_raw,
            )
            response = response.model_copy(
                update={"ai_explanation": ai_explanation, "ai_status": "ok"}
            )
            _log("AI explanation generated successfully")
            _save_artifact(run_dir, "step9_response_final", response)
            elapsed_stage = (time.perf_counter() - t_start) * 1000.0
            debugger.log_ai(response, elapsed_stage, resume_text=resume_raw)
            debugger.end_stage("AI Explanation", t_start, success=True)
        except InvalidInputError:
            _log("AI input guardrail rejected input — raising InvalidInputError")
            _save_artifact(run_dir, "step9_rejected_input", {"rejected": True})
            debugger.end_stage("AI Explanation", t_start, success=False, error=InvalidInputError("AI input guardrail rejected input"))
            raise
        except AIGenerationError as exc:
            _log(f"AI generation failed: {exc.message}")
            response = response.model_copy(update={"ai_status": "unavailable"})
            _save_artifact(run_dir, "step9_response_final_unavailable", response)
            debugger.end_stage("AI Explanation", t_start, success=False, error=exc)
        except Exception as exc:  # noqa: BLE001
            _log(f"AI explain-score chain failed (returning unavailable): {exc}")
            response = response.model_copy(update={"ai_status": "unavailable"})
            _save_artifact(run_dir, "step9_response_final_error", response)
            debugger.end_stage("AI Explanation", t_start, success=False, error=exc)

        # Pipeline completed
        debugger.log_pipeline_summary()
        return response

    except Exception as exc:
        debugger.log_pipeline_summary()
        raise

async def analyze_stream(
    request: ATSAnalyzeRequest,
) -> typing.AsyncGenerator[str, None]:
    """Execute the full ATS pipeline and yield progress events in SSE format."""
    import asyncio
    from app.ai.streaming.stream_events import (
        PipelineStartedEvent,
        ATSRunningEvent,
        ATSCompleteEvent,
        AIStartedEvent,
        AIAnalyzingStrengthsEvent,
        AIAnalyzingWeaknessesEvent,
        AIGeneratingSuggestionsEvent,
        AICompleteEvent,
        AIUnavailableEvent,
        CompleteEvent,
    )
    from app.ai.streaming.sse_encoder import encode as sse_encode, encode_error
    from app.ai.guardrails.input_guardrails import validate_all as validate_input

    rid = get_request_id()
    _log = lambda msg: logger.info("[%s] %s", rid[:8] if rid else "-", msg)  # noqa: E731

    start = time.perf_counter()
    _log("Streaming ATS pipeline started")

    save_enabled = os.environ.get("SAVE_ATS_ARTIFACTS", "false").lower() == "true"
    run_dir: Path | None = None
    if save_enabled:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = Path("ats_debug_artifacts") / f"run_{timestamp}"

    # Initialize debugger
    debugger = PipelineDebugger(
        filename=request.resume.filename,
        file_bytes=request.resume.file_bytes,
        text_input=request.resume.text,
        jd_text=request.job_description.text,
    )

    yield sse_encode(PipelineStartedEvent())
    await asyncio.sleep(0.01)

    yield sse_encode(ATSRunningEvent())
    await asyncio.sleep(0.01)

    try:
        # ── Stage 1: Resume Upload ──────────────────────────────────────────
        debugger.log_resume_upload()

        # ── Stage 2: Job Description ────────────────────────────────────────
        debugger.log_job_description()

        # ── Stage 3: PDF Parser ─────────────────────────────────────────────
        t_start = debugger.start_stage("pdf_parser", "PDF Parser")
        try:
            _log("Parsing resume")
            resume_raw = _parse_resume(request)
            _save_artifact(run_dir, "step1_resume_raw", resume_raw)
            elapsed_stage = (time.perf_counter() - t_start) * 1000.0
            debugger.log_parser(resume_raw, elapsed_stage)
            debugger.end_stage("PDF Parser", t_start, success=True)
        except Exception as exc:
            debugger.end_stage("PDF Parser", t_start, success=False, error=exc)
            yield encode_error(str(exc), code="VALIDATION_ERROR")
            return

        # ── Stage 4: Normalizer ─────────────────────────────────────────────
        t_start = debugger.start_stage("normalizer", "Normalizer")
        try:
            _log("Normalizing text")
            resume_clean = text_cleaner.clean(resume_raw)
            jd_clean = normalize_jd(request.job_description.text)
            _save_artifact(run_dir, "step2_resume_clean", resume_clean)
            _save_artifact(run_dir, "step2_jd_clean", jd_clean)
            elapsed_stage = (time.perf_counter() - t_start) * 1000.0
            debugger.log_normalizer(resume_raw, resume_clean, elapsed_stage)
            debugger.end_stage("Normalizer", t_start, success=True)
        except Exception as exc:
            debugger.end_stage("Normalizer", t_start, success=False, error=exc)
            yield encode_error(str(exc), code="VALIDATION_ERROR")
            return

        # ── Stage 4b: Input guardrails (fail-fast) ───────────────────────────
        try:
            validate_input(resume_clean, jd_clean)
        except InvalidInputError as exc:
            _log("AI input guardrail rejected input")
            _save_artifact(run_dir, "step2b_rejected_input", {"rejected": True, "message": exc.message})
            yield encode_error(exc.message, code="VALIDATION_ERROR")
            return

        # ── Stage 5: Section Splitter ───────────────────────────────────────
        t_start = debugger.start_stage("section_splitter", "Section Splitter")
        try:
            _log("Splitting sections")
            sections = section_splitter.split(resume_clean)
            _save_artifact(run_dir, "step3_sections", sections)
            elapsed_stage = (time.perf_counter() - t_start) * 1000.0
            debugger.log_section_splitter(sections, elapsed_stage)
            debugger.end_stage("Section Splitter", t_start, success=True)
        except Exception as exc:
            debugger.end_stage("Section Splitter", t_start, success=False, error=exc)
            yield encode_error(str(exc), code="VALIDATION_ERROR")
            return

        # ── Stage 6: Entity Extraction ──────────────────────────────────────
        t_start = debugger.start_stage("entity_extraction", "Entity Extraction")
        try:
            _log("Extracting entity data using Hybrid AI (with fallback)")
            entities = _extract_entities_hybrid(resume_clean, jd_clean, logger_fn=_log)
            resume_keywords = entities.resume_keywords
            resume_skills = entities.resume_skills
            jd_keywords = entities.jd_keywords
            required_skills = entities.required_skills

            exp_text = sections.experience or ""
            experience_entries: list[ExperienceEntry] = experience_extractor.extract(exp_text)

            edu_text = sections.education or ""
            education_result: EducationExtractionResult = education_extractor.extract(edu_text)

            _save_artifact(run_dir, "step4_resume_keywords", resume_keywords)
            _save_artifact(run_dir, "step4_resume_skills", resume_skills)
            _save_artifact(run_dir, "step4_experience_entries", experience_entries)
            _save_artifact(run_dir, "step4_education_result", education_result)

            _save_artifact(run_dir, "step5_jd_keywords", jd_keywords)
            _save_artifact(run_dir, "step5_required_skills", required_skills)

            elapsed_stage = (time.perf_counter() - t_start) * 1000.0
            debugger.log_entity_extraction(sections, experience_entries, education_result, resume_skills, elapsed_stage)
            debugger.end_stage("Entity Extraction", t_start, success=True)
        except Exception as exc:
            debugger.end_stage("Entity Extraction", t_start, success=False, error=exc)
            yield encode_error(str(exc), code="VALIDATION_ERROR")
            return

        # ── Stage 7: Keyword Matching ───────────────────────────────────────
        t_start = debugger.start_stage("keyword_matching", "Keyword Matching")
        try:
            _log("Matching keywords")
            match_result = keyword_matcher.match(
                resume_keywords=resume_keywords,
                jd_keywords=jd_keywords,
                embedding_provider=get_shared_provider(),
                semantic_threshold=0.60,
            )
            _save_artifact(run_dir, "step6_match_result", match_result)
            elapsed_stage = (time.perf_counter() - t_start) * 1000.0
            debugger.log_keyword_matching(resume_clean, resume_keywords, jd_keywords, match_result, elapsed_stage)
            debugger.end_stage("Keyword Matching", t_start, success=True)
        except Exception as exc:
            debugger.end_stage("Keyword Matching", t_start, success=False, error=exc)
            yield encode_error(str(exc), code="VALIDATION_ERROR")
            return

        # ── Stage 8: ATS Calculation ────────────────────────────────────────
        t_start = debugger.start_stage("ats_calculation", "ATS Calculation")
        try:
            _log("Calculating scores")
            report = scoring_engine.score(
                match_result=match_result,
                experience_entries=experience_entries,
                resume_skills=resume_skills,
                required_skills=required_skills,
                education_result=education_result,
                sections=sections,
                required_years=entities.required_years,
                required_education_level=entities.required_education_level,
                embedding_provider=get_shared_provider(),
            )
            _save_artifact(run_dir, "step7_report", report)
            elapsed_stage = (time.perf_counter() - t_start) * 1000.0
            debugger.log_ats_score(report, match_result, experience_entries, education_result, sections, elapsed_stage)
            debugger.end_stage("ATS Calculation", t_start, success=True)
        except Exception as exc:
            debugger.end_stage("ATS Calculation", t_start, success=False, error=exc)
            yield encode_error(str(exc), code="VALIDATION_ERROR")
            return

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _log(f"ATS pipeline completed in {elapsed_ms:.1f}ms — overall score: {report.overall_score}")

        # ── Step 8: Build deterministic response ────────────────────────────────
        response = _build_response(
            report=report,
            match_result=match_result,
            culture_signals=entities.culture_signals,
            extraction_mode=entities.extraction_mode,
            experience_entries=experience_entries,
            education_result=education_result,
            processing_time_ms=elapsed_ms,
        )
        _save_artifact(run_dir, "step8_response_deterministic", response)

        yield sse_encode(ATSCompleteEvent(overall_score=response.overall_score))
        await asyncio.sleep(0.01)

        yield sse_encode(AIStartedEvent())
        await asyncio.sleep(0.01)

        # ── Step 9: AI explain-score chain with progress reporting ──────────────
        yield sse_encode(AIAnalyzingStrengthsEvent())
        await asyncio.sleep(0.01)
        yield sse_encode(AIAnalyzingWeaknessesEvent())
        await asyncio.sleep(0.01)
        yield sse_encode(AIGeneratingSuggestionsEvent())
        await asyncio.sleep(0.01)

        # ── Stage 9: AI Explanation ─────────────────────────────────────────
        t_start = debugger.start_stage("ai_explanation", "AI Explanation")
        try:
            _log("Running AI explain-score chain")
            ai_explanation = await asyncio.to_thread(
                run_explain_score,
                response=response,
                jd_text=request.job_description.text,
                resume_text=resume_raw,
            )
            response = response.model_copy(
                update={"ai_explanation": ai_explanation, "ai_status": "ok"}
            )
            _save_artifact(run_dir, "step9_response_final", response)
            yield sse_encode(AICompleteEvent())
            await asyncio.sleep(0.01)
            elapsed_stage = (time.perf_counter() - t_start) * 1000.0
            debugger.log_ai(response, elapsed_stage, resume_text=resume_raw)
            debugger.end_stage("AI Explanation", t_start, success=True)
        except InvalidInputError as exc:
            _log("AI input guardrail rejected input inside chain")
            _save_artifact(run_dir, "step9_rejected_input", {"rejected": True, "message": exc.message})
            debugger.end_stage("AI Explanation", t_start, success=False, error=exc)
            yield encode_error(exc.message, code="VALIDATION_ERROR")
            return
        except AIGenerationError as exc:
            _log(f"AI generation failed: {exc.message}")
            response = response.model_copy(update={"ai_status": "unavailable"})
            _save_artifact(run_dir, "step9_response_final_unavailable", response)
            debugger.end_stage("AI Explanation", t_start, success=False, error=exc)
            yield sse_encode(AIUnavailableEvent(reason=exc.message))
            await asyncio.sleep(0.01)
        except Exception as exc:  # noqa: BLE001
            _log(f"AI explain-score chain failed: {exc}")
            response = response.model_copy(update={"ai_status": "unavailable"})
            _save_artifact(run_dir, "step9_response_final_error", response)
            debugger.end_stage("AI Explanation", t_start, success=False, error=exc)
            yield sse_encode(AIUnavailableEvent(reason=str(exc)))
            await asyncio.sleep(0.01)

        # Yield the final complete result carrying the full payload
        yield sse_encode(CompleteEvent(payload=response.model_dump(mode="json")))

        # Pipeline completed
        debugger.log_pipeline_summary()

    except Exception as exc:
        debugger.log_pipeline_summary()
        raise


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractedATSEntities:
    """Score-ready entities with culture signals kept outside technical coverage."""

    resume_keywords: list[str]
    resume_skills: list[str]
    jd_keywords: list[str]
    required_skills: list[str]
    culture_signals: list[str]
    required_years: float
    required_education_level: str
    extraction_mode: str


def _extract_entities_hybrid(
    resume_clean: str,
    jd_clean: str,
    logger_fn: typing.Callable[[str], None] | None = None,
) -> ExtractedATSEntities:
    """Extract keywords and skills using Hybrid AI Agent extraction with fallback.

    Returns:
        Score-ready entities, with culture signals excluded from score inputs.
    """
    log = logger_fn or (lambda msg: None)
    try:
        log("Attempting Hybrid AI entity extraction")
        resume_ext = extract_resume_entities(resume_clean)
        jd_ext = extract_jd_entities(jd_clean)

        taxonomy = classify_jd_requirements(jd_ext)
        r_skills = resume_technical_evidence(resume_ext)
        r_keywords = r_skills + [
            item for item in resume_ext.soft_skills if item.lower() not in {skill.lower() for skill in r_skills}
        ]
        j_keywords = taxonomy.keyword_requirements
        req_skills = taxonomy.required_technical_skills

        log(f"[Hybrid AI] Extracted {len(r_keywords)} resume keywords, {len(j_keywords)} JD keywords")
        log(
            f"[Hybrid AI] Scoring {len(req_skills)} required technical skills; "
            f"retaining {len(taxonomy.culture_signals)} culture signals for feedback"
        )
        return ExtractedATSEntities(
            resume_keywords=r_keywords,
            resume_skills=r_skills,
            jd_keywords=j_keywords,
            required_skills=req_skills,
            culture_signals=taxonomy.culture_signals,
            required_years=jd_ext.min_experience,
            required_education_level=jd_ext.required_education_level,
            extraction_mode="hybrid_ai",
        )

    except Exception as exc:
        log(f"[Hybrid AI] Extraction failed or unavailable ({exc}). Falling back to naive extractors.")
        logger.warning(
            "[HybridExtract] AI extraction failed (%s). Falling back to naive extractors.",
            exc,
        )

        resume_ext = fallback_resume_extraction(resume_clean)
        jd_ext = fallback_jd_extraction(jd_clean)
        taxonomy = classify_jd_requirements(jd_ext)
        r_skills = resume_technical_evidence(resume_ext)
        r_keywords = list(r_skills)
        j_keywords = taxonomy.keyword_requirements
        req_skills = taxonomy.required_technical_skills

        return ExtractedATSEntities(
            resume_keywords=r_keywords,
            resume_skills=r_skills,
            jd_keywords=j_keywords,
            required_skills=req_skills,
            culture_signals=taxonomy.culture_signals,
            required_years=jd_ext.min_experience,
            required_education_level=jd_ext.required_education_level,
            extraction_mode="deterministic_fallback",
        )


def _parse_resume(request: ATSAnalyzeRequest) -> str:
    """Convert the resume input to plain text."""
    if request.resume.file_bytes is not None and request.resume.filename is not None:
        return parser_factory.parse(
            filename=request.resume.filename,
            file_bytes=request.resume.file_bytes,
        )

    if request.resume.text is not None:
        return parser_factory.parse_text(request.resume.text)

    raise ValidationError("Resume must provide either 'text' or 'filename'+'file_bytes'.")


def _build_response(
    *,
    report: scoring_engine.ATSReport,
    match_result: keyword_matcher.MatchResult,
    culture_signals: list[str],
    extraction_mode: str,
    experience_entries: list[ExperienceEntry],
    education_result: EducationExtractionResult,
    processing_time_ms: float,
) -> ATSAnalyzeResponse:
    """Assemble the final API response from all pipeline outputs."""
    matched_kw = [
        MatchedKeywordSchema(
            keyword=m.keyword,
            matchType=m.matchType,
            similarity=m.similarity,
            matched_jd_keyword=m.matched_jd_keyword,
            is_related_concept=m.is_related_concept,
        )
        for m in match_result.matched
    ]
    related_kw = [
        MatchedKeywordSchema(
            keyword=m.keyword,
            matchType=m.matchType,
            similarity=m.similarity,
            matched_jd_keyword=m.matched_jd_keyword,
            is_related_concept=True,
        )
        for m in match_result.related
    ]

    matched_skills = [match.keyword for match in report.skill_match_result.matched]
    missing_skills = list(report.skill_match_result.missing)

    exp_summary = ExperienceSummarySchema(
        total_entries=len(experience_entries),
        total_years=sum(
            e.duration_years for e in experience_entries
            if e.duration_years is not None
        ),
        has_metrics=any(e.metrics for e in experience_entries),
    )

    highest = None
    if education_result.entries:
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
        related_keywords=related_kw,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        required_skill_count=report.required_skill_count,
        culture_signals=culture_signals,
        extraction_mode=extraction_mode,
        required_experience_years=report.required_experience_years,
        candidate_experience_years=report.candidate_experience_years,
        required_education_level=report.required_education_level,
        candidate_education_level=report.candidate_education_level,
        experience_summary=exp_summary,
        education_summary=edu_summary,
        processing_time_ms=round(processing_time_ms, 2),
    )
