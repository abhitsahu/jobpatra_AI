"""Entity extraction prompts — version 1.

Defines two ``ChatPromptTemplate`` objects used by the Hybrid AI Extraction
pipeline to extract structured, categorized entities from resume and job
description text before keyword matching and scoring.

Design goals
------------
* Produce a **small, precise** list of keywords (15–30 items max) rather than
  exhaustive token dumps.
* Handle BOTH technical roles (software, data, devops) and non-technical roles
  (HR, Sales, Marketing, Finance, Legal) correctly via ``domain_terms``.
* Return strict JSON so the output guardrail can parse/repair without retries.

This file does NOT:
  - Call any LLM or external service.
  - Contain chain orchestration logic.
  - Import FastAPI.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------------------------
# Resume Extraction Prompt
# ---------------------------------------------------------------------------

_RESUME_SYSTEM = """\
You are a precise ATS data extraction engine. Your ONLY job is to extract \
structured entities from a candidate's resume text.

STRICT RULES:
1. Extract ONLY what is explicitly present in the text. Do NOT infer or add.
2. Return a SINGLE valid JSON object — no markdown, no prose, no code fences.
3. Keep lists tight: 10–25 items max per field. Prefer canonical names \
   (e.g., "React" not "React.js / ReactJS").
4. For non-technical resumes (HR, Sales, Marketing, Finance, Legal, Operations), \
   populate domain_terms with role-specific concepts such as: \
   "Talent Acquisition", "Employee Relations", "HRIS", "OKRs", \
   "Pipeline Management", "P&L Management", "Go-to-Market", "Financial Modeling", \
   "Contract Negotiation", "Compliance", "Stakeholder Management".
5. experience_years: total NUMERIC years (e.g., 4.5). If unclear, return 0.
6. education: highest degree only (e.g., "B.Tech Computer Science").
7. achievements: only quantified bullet points (%, $, x multipliers). \
   Max 5 items. Return [] if none found.
8. domain_terms: Include only technical domain concepts explicitly stated in the resume. Do NOT infer associated paradigms, architectures, or engineering concepts from a named technology.
"""

_RESUME_HUMAN = """\
Extract entities from the following resume text.

## Resume Text
{resume_text}

Return ONLY this JSON structure (no extra keys, no markdown):
{{
  "hard_skills": ["<skill>", ...],
  "soft_skills": ["<skill>", ...],
  "domain_terms": ["<term>", ...],
  "experience_years": <number>,
  "job_titles": ["<title>", ...],
  "achievements": ["<quantified achievement>", ...],
  "education": "<highest degree>"
}}
"""

RESUME_EXTRACTION_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", _RESUME_SYSTEM),
        ("human", _RESUME_HUMAN),
    ]
)


# ---------------------------------------------------------------------------
# Job Description Extraction Prompt
# ---------------------------------------------------------------------------

_JD_SYSTEM = """\
You are a precise ATS data extraction engine. Your ONLY job is to extract \
structured hiring requirements from a job description.

STRICT RULES:
1. Extract ONLY requirements explicitly stated in the text. Do NOT infer.
2. Return a SINGLE valid JSON object — no markdown, no prose, no code fences.
3. Keep lists tight: 10–25 items max per field. Use canonical names.
4. Distinguish clearly:
   - required_hard_skills: Technical tools/technologies the role REQUIRES.
   - preferred_hard_skills: Technical tools/technologies listed as "nice to have" or "preferred".
   - required_soft_skills: Behavioural/interpersonal traits REQUIRED (e.g., "leadership", \
     "communication").
5. For non-technical JDs (HR, Sales, Marketing, Finance, Legal, Operations), \
   populate domain_terms with role-specific concepts: \
   "Talent Acquisition", "Employee Relations", "HRIS", "Payroll Processing", \
   "Pipeline Management", "CRM", "Revenue Forecasting", "P&L Ownership", \
   "Financial Modeling", "GAAP", "Contract Drafting", "SLA Management".
6. min_experience: numeric minimum years stated (e.g., 3). Return 0 if not stated.
7. required_education_level: Return exactly one of "none", "associate", "bachelors", "masters", or "phd". Only set a non-"none" value when the JD explicitly requires that degree level; do not infer it.
8. key_responsibilities: 3–6 concise phrases (not full sentences).
9. domain_terms: Include only explicitly stated payment, infrastructure, or industry concepts. Do NOT infer associated languages, technologies, or architectures from a high-level requirement.
"""

_JD_HUMAN = """\
Extract hiring requirements from the following job description.

## Job Description
{jd_text}

Return ONLY this JSON structure (no extra keys, no markdown):
{{
  "required_hard_skills": ["<skill>", ...],
  "preferred_hard_skills": ["<skill>", ...],
  "required_soft_skills": ["<skill>", ...],
  "domain_terms": ["<term>", ...],
  "min_experience": <number>,
  "required_education_level": "<none|associate|bachelors|masters|phd>",
  "key_responsibilities": ["<phrase>", ...]
}}
"""

JD_EXTRACTION_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", _JD_SYSTEM),
        ("human", _JD_HUMAN),
    ]
)
