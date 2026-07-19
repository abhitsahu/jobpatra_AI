"""Job Description Preprocessor.

Extracts key sections deterministically from a job description to build a
compact, high-signal context for the LLM.
"""

from __future__ import annotations

import re

from app.analysis.normalization import jd_normalizer
from app.analysis.extraction import keyword_extractor, skill_extractor

_JD_HEADING_MAP: dict[str, tuple[str, ...]] = {
    "responsibilities": (
        "responsibilities",
        "what you will do",
        "key responsibilities",
        "essential duties",
        "duties",
        "responsibilities include",
        "role and responsibilities",
        "your role",
        "what you'll do",
        "day-to-day",
        "typical day",
        "tasks",
        "role",
        "about the role",
        "the role",
        "expectations",
    ),
    "required_skills": (
        "required skills",
        "requirements",
        "what you need",
        "what you'll need",
        "qualifications",
        "basic qualifications",
        "experience required",
        "minimum qualifications",
        "must have",
        "key requirements",
        "skills & experience",
        "about you",
        "what you bring",
        "what we're looking for",
    ),
    "preferred_skills": (
        "preferred skills",
        "nice to have",
        "plus",
        "bonus",
        "preferred qualifications",
        "highly desired",
        "desirable skills",
        "preferred experience",
        "good to have",
        "optional",
        "desirable",
    ),
    "qualifications": (
        "education",
        "degree",
        "academic requirements",
        "academic background",
        "certifications",
        "credentials",
        "educational background",
    )
}


def preprocess_jd(raw_jd: str) -> str:
    """Normalize, extract key sections and keywords, and build a compact JD context.

    Args:
        raw_jd: Raw job description text.

    Returns:
        A compact structured job description string.
    """
    # 1. Normalize
    normalized = jd_normalizer.normalize(raw_jd)

    # 2. Heuristic section splitting
    lines = normalized.splitlines()

    sections: dict[str, list[str]] = {
        "responsibilities": [],
        "required_skills": [],
        "preferred_skills": [],
        "qualifications": [],
        "other": []
    }

    current_section = "other"

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue

        cleaned_line = stripped_line.lower().rstrip(":.- ")

        # Check if line is a heading
        matched_section = None
        for sec_name, aliases in _JD_HEADING_MAP.items():
            if cleaned_line in aliases:
                matched_section = sec_name
                break
            # Fuzzy/partial match for short header lines
            if len(cleaned_line) < 30 and any(a in cleaned_line for a in aliases if len(a) > 5):
                matched_section = sec_name
                break

        if matched_section:
            current_section = matched_section
        else:
            # Clean list bullets
            content_line = re.sub(r"^\s*[-*•\d.]+\s+", "", stripped_line)
            sections[current_section].append(content_line)

    # 3. Extract Keywords and Skills
    extracted_keywords = keyword_extractor.extract(normalized)
    keywords_str = ", ".join(extracted_keywords[:30])

    skills_res = skill_extractor.extract(normalized)
    skills_list = []
    for cat, items in skills_res.by_category.items():
        skills_list.append(f"{cat}: {', '.join(items)}")
    skills_str = "\n".join(skills_list)

    # 4. Build compact JD context
    context_parts = []
    if sections["responsibilities"]:
        context_parts.append("### Key Responsibilities:")
        context_parts.append("\n".join(f"- {l}" for l in sections["responsibilities"][:15]))
    if sections["required_skills"]:
        context_parts.append("### Required Skills & Experience:")
        context_parts.append("\n".join(f"- {l}" for l in sections["required_skills"][:15]))
    if sections["preferred_skills"]:
        context_parts.append("### Preferred / Nice-to-Have Skills:")
        context_parts.append("\n".join(f"- {l}" for l in sections["preferred_skills"][:10]))
    if sections["qualifications"]:
        context_parts.append("### Education & Qualifications:")
        context_parts.append("\n".join(f"- {l}" for l in sections["qualifications"][:8]))

    # If we couldn't split anything because there were no headers, use a truncated clean version
    if not any(sections[k] for k in ["responsibilities", "required_skills", "preferred_skills", "qualifications"]):
        context_parts.append("### Job Details:")
        context_parts.append(normalized[:1200])

    if keywords_str:
        context_parts.append("### Extracted Target Keywords:")
        context_parts.append(keywords_str)
    if skills_str:
        context_parts.append("### Extracted Target Skills:")
        context_parts.append(skills_str)

    return "\n\n".join(context_parts)
