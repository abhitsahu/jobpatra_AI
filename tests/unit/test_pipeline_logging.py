"""Unit tests for the end-to-end ATS pipeline logging/debugging instrumentation."""

import logging
import pytest
import os
from app.services.ats_service import PipelineDebugger, _get_rejected_tokens

def test_pipeline_debugger_logging(caplog):
    """Verify that PipelineDebugger logs correctly under various stages."""
    os.environ["DEBUG_ATS_PIPELINE"] = "true"

    debugger = PipelineDebugger(
        filename="test_resume.pdf",
        file_bytes=b"%PDF-1.4 test resume bytes",
        text_input=None,
        jd_text="Looking for a Python Developer with SQL experience.",
    )

    with caplog.at_level(logging.INFO, logger="jobpatra"):
        debugger.log_resume_upload()
        debugger.log_job_description()
        debugger.log_parser("Test Page 1\n\nTest Page 2", 15.5)
        debugger.log_normalizer("Raw unicode \u2022 bullet", "Cleaned bullet", 5.0)

    log_messages = [r.message for r in caplog.records if r.name == "jobpatra"]
    
    # Check that stage headers are printed
    assert any("RESUME UPLOAD" in m for m in log_messages)
    assert any("JOB DESCRIPTION" in m for m in log_messages)
    assert any("test_resume.pdf" in m for m in log_messages)
    assert any("DirectoryLoader + PyMuPDFLoader" in m for m in log_messages)
    assert any("Unicode artifacts removed" in m for m in log_messages)


def test_rejected_tokens():
    """Verify that _get_rejected_tokens identifies and labels rejected tokens correctly."""
    text = "a the showing C++ C# invalid_token!@#"
    rejected = _get_rejected_tokens(text)
    
    # Stop words or length-1 tokens should be rejected
    rejected_tokens = [tok for tok, reason in rejected]
    assert "a" in rejected_tokens
    assert "the" in rejected_tokens
    
    # Whitelisted C++ or C# should NOT be rejected
    assert "C++" not in rejected_tokens
    assert "C#" not in rejected_tokens
