"""Resume improvement prompts — section-specific LLM instructions.

Each prompt is a ``ChatPromptTemplate`` that produces a concise, ATS-friendly
rewrite of a single resume section.

Design constraints (enforced in every prompt):
* Return ONLY the improved text — no preamble, no labels, no quotation marks.
* Preserve approximate length (±20 words of the original).
* Use strong action verbs and quantifiable metrics where possible.
* Keep the candidate's authentic voice; never invent roles or achievements.
* Output must be copy-paste ready.

Usage
-----
.. code-block:: python

    from app.ai.prompts.resume_improve import SECTION_PROMPTS

    prompt = SECTION_PROMPTS.get(section_type, SECTION_PROMPTS["_default"])
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Shared system preamble
# ---------------------------------------------------------------------------

_SYSTEM_BASE = """\
You are a world-class Senior Resume Writer with experience crafting resumes for \
engineers and tech professionals hired at Google, Amazon, Microsoft, and top-tier \
startups. You specialise in making resumes ATS-friendly, impactful, and concise.

HARD RULES — violating ANY of these is unacceptable:
1. Return ONLY the rewritten text. No preamble, no labels, no explanations, no quotation marks.
2. Do NOT invent new roles, companies, achievements, or metrics that are not in the original.
3. Preserve approximate length: ±20 words of the original.
4. Use strong action verbs (Spearheaded, Architected, Delivered, Reduced, Increased, etc.).
5. Integrate quantifiable metrics wherever the original has them; improve phrasing around them.
6. Keep the candidate's authentic first-person or bullet-point style consistent with the original.
"""

# ---------------------------------------------------------------------------
# Summary / Professional Summary
# ---------------------------------------------------------------------------

_SUMMARY_HUMAN = """\
Rewrite the professional summary below to be more ATS-friendly, impactful, and \
engaging for a recruiter. Focus on strong action language, key competencies, and \
clear value proposition.

{context_block}

ORIGINAL SUMMARY:
{current_text}

Rewrite:"""

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_BASE),
        ("human", _SUMMARY_HUMAN),
    ]
)

# ---------------------------------------------------------------------------
# Work Experience description
# ---------------------------------------------------------------------------

_EXPERIENCE_HUMAN = """\
Rewrite the work experience description below to be more ATS-friendly and impactful. \
Improve action verbs, tighten language, and highlight measurable impact where present.

{context_block}

ORIGINAL EXPERIENCE DESCRIPTION:
{current_text}

Rewrite:"""

EXPERIENCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_BASE),
        ("human", _EXPERIENCE_HUMAN),
    ]
)

# ---------------------------------------------------------------------------
# Project description
# ---------------------------------------------------------------------------

_PROJECTS_HUMAN = """\
Rewrite the project description below to better showcase technical depth, business \
impact, and key technologies. Make it ATS-friendly and recruiter-ready.

{context_block}

ORIGINAL PROJECT DESCRIPTION:
{current_text}

Rewrite:"""

PROJECTS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_BASE),
        ("human", _PROJECTS_HUMAN),
    ]
)

# ---------------------------------------------------------------------------
# Objective statement
# ---------------------------------------------------------------------------

_OBJECTIVE_HUMAN = """\
Rewrite the career objective below to be concise, confident, and targeted. \
Make it ATS-friendly and compelling for recruiters.

{context_block}

ORIGINAL OBJECTIVE:
{current_text}

Rewrite:"""

OBJECTIVE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_BASE),
        ("human", _OBJECTIVE_HUMAN),
    ]
)

# ---------------------------------------------------------------------------
# Generic / default fallback
# ---------------------------------------------------------------------------

_DEFAULT_HUMAN = """\
Rewrite the resume section text below to be more professional, ATS-friendly, \
and impactful. Improve clarity, action language, and conciseness.

{context_block}

ORIGINAL TEXT:
{current_text}

Rewrite:"""

DEFAULT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_BASE),
        ("human", _DEFAULT_HUMAN),
    ]
)

# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

SECTION_PROMPTS: dict[str, ChatPromptTemplate] = {
    "summary": SUMMARY_PROMPT,
    "experience": EXPERIENCE_PROMPT,
    "projects": PROJECTS_PROMPT,
    "objective": OBJECTIVE_PROMPT,
    "_default": DEFAULT_PROMPT,
}

VALID_SECTION_TYPES = frozenset(SECTION_PROMPTS.keys()) - {"_default"}
