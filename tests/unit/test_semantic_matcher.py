"""Unit tests for semantic_matcher and its integration with keyword_matcher.

All tests are fully deterministic — a ``FakeEmbeddingProvider`` injects
pre-computed vectors without any network calls.

The fake embeddings are hand-crafted 3-dimensional unit vectors designed so
that semantically similar pairs have high cosine similarity and dissimilar
pairs have low cosine similarity.  This removes any dependency on a real
embedding model while still exercising the full cosine-similarity + threshold
logic.

Deterministic. No AI. No network.
"""

import logging
import math
from typing import Any

import pytest

from app.analysis.matching import keyword_matcher, semantic_matcher
from app.analysis.matching.semantic_matcher import (
    EmbeddingProvider,
    Embedding,
    SemanticMatchResult,
    _cosine_similarity,
    match_unresolved,
    SIMILARITY_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Fake embedding provider — deterministic, no network
# ---------------------------------------------------------------------------


def _unit(v: list[float]) -> list[float]:
    """Normalise a vector to unit length."""
    mag = math.sqrt(sum(x * x for x in v))
    return [x / mag for x in v]


# Hand-crafted embeddings — similar concepts share close directions
_FAKE_VECTORS: dict[str, Embedding] = {
    # Very similar — should always be above threshold
    "REST API":           _unit([1.0, 0.8, 0.1]),
    "RESTful Services":   _unit([0.95, 0.82, 0.05]),
    "Backend Development":_unit([0.7, 0.9, 0.2]),
    "Built scalable APIs":_unit([0.72, 0.88, 0.22]),
    "Microservices":      _unit([0.6, 0.8, 0.5]),
    "Distributed Systems":_unit([0.55, 0.78, 0.55]),
    # Very different — should always be below threshold
    "Python":             _unit([0.0, 0.0, 1.0]),
    "Project Management": _unit([0.9, 0.0, 0.0]),
}


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic provider backed by ``_FAKE_VECTORS``."""

    def embed(self, texts: list[str]) -> list[Embedding]:
        result = []
        for text in texts:
            if text not in _FAKE_VECTORS:
                # Unknown text — return a near-zero vector (will not match)
                result.append([0.001, 0.001, 0.001])
            else:
                result.append(_FAKE_VECTORS[text])
        return result


_PROVIDER = FakeEmbeddingProvider()


# ---------------------------------------------------------------------------
# Tests — _cosine_similarity (pure math)
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_return_one(self) -> None:
        v = _unit([1.0, 2.0, 3.0])
        assert _cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors_return_zero(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_similar_vectors_high_score(self) -> None:
        a = _unit([1.0, 0.8, 0.1])
        b = _unit([0.95, 0.82, 0.05])
        score = _cosine_similarity(a, b)
        assert score > 0.99  # These vectors are very close


# ---------------------------------------------------------------------------
# Tests — match_unresolved (semantic_matcher public API)
# ---------------------------------------------------------------------------


class TestMatchUnresolved:
    def test_rest_api_matches_restful_services(self) -> None:
        """'REST API' in JD should match 'RESTful Services' on resume."""
        results, remaining_jd = match_unresolved(
            resume_keywords=["RESTful Services"],
            jd_keywords=["REST API"],
            provider=_PROVIDER,
        )
        assert len(results) == 1
        assert results[0].matched is True
        assert results[0].matchType == "RELATED"
        assert results[0].similarity > SIMILARITY_THRESHOLD
        assert remaining_jd == []

    def test_backend_development_matches_built_scalable_apis(self) -> None:
        """'Backend Development' (JD) should match 'Built scalable APIs' (resume)."""
        results, remaining_jd = match_unresolved(
            resume_keywords=["Built scalable APIs"],
            jd_keywords=["Backend Development"],
            provider=_PROVIDER,
        )
        assert results[0].matched is True
        assert results[0].matchType == "RELATED"
        assert remaining_jd == []

    def test_microservices_matches_distributed_systems(self) -> None:
        """'Microservices' and 'Distributed Systems' are semantically related."""
        results, _ = match_unresolved(
            resume_keywords=["Microservices"],
            jd_keywords=["Distributed Systems"],
            provider=_PROVIDER,
        )
        assert results[0].matched is True

    def test_python_does_not_match_project_management(self) -> None:
        """'Python' and 'Project Management' are unrelated — must NOT match."""
        results, remaining_jd = match_unresolved(
            resume_keywords=["Python"],
            jd_keywords=["Project Management"],
            provider=_PROVIDER,
        )
        assert results[0].matched is False
        assert results[0].matchType == "MISSING"
        assert "Project Management" in remaining_jd

    def test_python_does_not_count_as_functional_programming(self) -> None:
        """A named language is not evidence of functional-programming practice."""
        results, remaining_jd = match_unresolved(
            resume_keywords=["Python"],
            jd_keywords=["Functional Programming"],
            provider=_PROVIDER,
            threshold=0.60,
        )
        assert results[0].matched is False
        assert remaining_jd == ["Functional Programming"]

    def test_similarity_score_returned(self) -> None:
        """Similarity score must always be populated (for debugging)."""
        results, _ = match_unresolved(
            resume_keywords=["RESTful Services"],
            jd_keywords=["REST API"],
            provider=_PROVIDER,
        )
        assert isinstance(results[0].similarity, float)
        assert 0.0 <= results[0].similarity <= 1.0

    def test_threshold_respected_at_high_value(self) -> None:
        """At threshold=1.0 nothing matches (unless vectors are identical)."""
        results, remaining_jd = match_unresolved(
            resume_keywords=["RESTful Services"],
            jd_keywords=["REST API"],
            provider=_PROVIDER,
            threshold=1.0,
        )
        assert results[0].matched is False
        assert "REST API" in remaining_jd

    def test_empty_resume_keywords(self) -> None:
        results, remaining_jd = match_unresolved(
            resume_keywords=[],
            jd_keywords=["REST API"],
            provider=_PROVIDER,
        )
        assert results == []
        assert remaining_jd == ["REST API"]

    def test_empty_jd_keywords(self) -> None:
        results, remaining_jd = match_unresolved(
            resume_keywords=["RESTful Services"],
            jd_keywords=[],
            provider=_PROVIDER,
        )
        assert results[0].matched is False
        assert remaining_jd == []

    def test_returns_semantic_match_result_objects(self) -> None:
        results, _ = match_unresolved(
            resume_keywords=["RESTful Services"],
            jd_keywords=["REST API"],
            provider=_PROVIDER,
        )
        for r in results:
            assert isinstance(r, SemanticMatchResult)

    def test_logs_top_three_semantic_candidates(self, caplog: pytest.LogCaptureFixture) -> None:
        """Semantic diagnostics include the best three JD candidates per keyword."""
        caplog.set_level(logging.INFO, logger="jobpatra")
        match_unresolved(
            resume_keywords=["RESTful Services"],
            jd_keywords=["REST API", "Backend Development", "Project Management"],
            provider=_PROVIDER,
        )
        assert "[SemanticMatcher] Top matches for 'RESTful Services'" in caplog.text


# ---------------------------------------------------------------------------
# Tests — keyword_matcher integration
# ---------------------------------------------------------------------------


class TestKeywordMatcherWithSemantic:
    """Verify the full four-step pipeline with the fake provider."""

    def test_semantic_match_type_assigned(self) -> None:
        """'RESTful Services' vs 'REST API' — falls through exact/synonym/fuzzy,
        then resolves via semantic matching."""
        result = keyword_matcher.match(
            resume_keywords=["RESTful Services"],
            jd_keywords=["REST API"],
            embedding_provider=_PROVIDER,
        )
        assert result.matched == []
        assert result.related[0].keyword == "RESTful Services"
        assert result.related[0].matchType == "RELATED"
        assert result.related[0].similarity is not None
        assert result.related[0].is_related_concept is True
        assert result.missing == ["REST API"]
        assert result.unresolved == []

    def test_no_unresolved_with_provider(self) -> None:
        """When a provider is supplied, unresolved list must always be empty."""
        result = keyword_matcher.match(
            resume_keywords=["RESTful Services", "Python"],
            jd_keywords=["REST API", "Project Management"],
            embedding_provider=_PROVIDER,
        )
        assert result.unresolved == []

    def test_full_pipeline_all_match_types(self) -> None:
        """
        Resume:  React, NodeJS, RESTful Services
        JD:      React, Node.js, REST API, AWS
        Expected:
          React          → EXACT
          NodeJS         → EXACT
          RESTful Services → SEMANTIC
          AWS            → MISSING
        """
        result = keyword_matcher.match(
            resume_keywords=["React", "NodeJS", "RESTful Services"],
            jd_keywords=["React", "Node.js", "REST API", "AWS"],
            embedding_provider=_PROVIDER,
        )
        match_map = {m.keyword: m.matchType for m in result.matched}

        assert match_map.get("React") == "EXACT"
        assert match_map.get("NodeJS") == "EXACT"
        assert match_map.get("RESTful Services") is None
        assert result.related[0].keyword == "RESTful Services"
        assert result.related[0].matched_jd_keyword == "REST API"
        assert "AWS" in result.missing
        assert result.unresolved == []

    def test_without_provider_unresolved_preserved(self) -> None:
        """When no provider is given, unresolved keywords are returned as-is."""
        result = keyword_matcher.match(
            resume_keywords=["RESTful Services"],
            jd_keywords=["REST API"],
            embedding_provider=None,
        )
        # 'RESTful Services' fails exact/synonym/fuzzy against 'REST API'
        assert result.unresolved == ["RESTful Services"]
        assert result.missing == ["REST API"]

    def test_similarity_field_populated_for_semantic(self) -> None:
        result = keyword_matcher.match(
            resume_keywords=["RESTful Services"],
            jd_keywords=["REST API"],
            embedding_provider=_PROVIDER,
        )
        sem_matches = [m for m in result.matched if m.matchType == "SEMANTIC"]
        for m in sem_matches:
            assert isinstance(m.similarity, float)

    def test_semantic_threshold_override(self) -> None:
        """At threshold=1.0 nothing should semantic-match."""
        result = keyword_matcher.match(
            resume_keywords=["RESTful Services"],
            jd_keywords=["REST API"],
            embedding_provider=_PROVIDER,
            semantic_threshold=1.0,
        )
        sem_matches = [m for m in result.matched if m.matchType == "SEMANTIC"]
        assert sem_matches == []
        # The keyword should not be in matched at all
        assert result.unresolved == []         # provider was given
        assert "REST API" in result.missing    # JD keyword unmatched
