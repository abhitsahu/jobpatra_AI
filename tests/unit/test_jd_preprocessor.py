"""Unit tests for the deterministic Job Description Preprocessor (Phase 10.4)."""

from __future__ import annotations

from app.analysis.normalization.jd_preprocessor import preprocess_jd


def test_preprocess_jd_with_valid_structure() -> None:
    """Verify that preprocess_jd extracts and organizes structured sections from a standard JD."""
    raw_jd = (
        "Role: Senior Software Engineer\n"
        "We are looking for a backend developer.\n\n"
        "Responsibilities:\n"
        "- Build highly scalable web applications in Python.\n"
        "- Design robust PostgreSQL database schemas.\n\n"
        "Requirements & Skills:\n"
        "- 5+ years of experience with Python.\n"
        "- Familiarity with Docker and Kubernetes.\n\n"
        "Education:\n"
        "- Bachelor's degree in Computer Science.\n"
    )

    result = preprocess_jd(raw_jd)

    assert "### Key Responsibilities:" in result
    assert "Build highly scalable web applications" in result
    assert "### Required Skills & Experience:" in result
    assert "5+ years of experience with Python" in result
    assert "### Education & Qualifications:" in result
    assert "Bachelor's degree in Computer Science" in result
    assert "### Extracted Target Skills:" in result


def test_preprocess_jd_empty_or_malformed() -> None:
    """Verify that preprocess_jd degrades gracefully when input is empty or lacks headers."""
    empty_result = preprocess_jd("")
    assert "### Job Details:" in empty_result

    short_result = preprocess_jd("Just a simple line of description.")
    assert "### Job Details:" in short_result
    assert "Just a simple line of description." in short_result


def test_preprocess_jd_truncation_fallback() -> None:
    """Verify that a massive JD without headings is truncated to maximum 1200 characters."""
    massive_jd = "x" * 2000
    result = preprocess_jd(massive_jd)

    # The raw text inside Job Details should be at most 1200 characters.
    # Total result contains headers. Let's find "### Job Details:\n\n" and get that content.
    start = result.find("### Job Details:\n\n") + len("### Job Details:\n\n")
    # Find next header
    end = result.find("###", start)
    if end != -1:
        details_part = result[start:end].strip()
    else:
        details_part = result[start:].strip()

    assert len(details_part) <= 1200
