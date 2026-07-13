"""Semantic matcher — Step 4 (final) of the keyword matching pipeline.

This module resolves keywords that survived the three deterministic passes
(exact, synonym, fuzzy) by computing embedding-based cosine similarity.

Architecture
------------
``semantic_matcher`` depends on an **EmbeddingProvider** abstraction, not on
any vendor SDK directly.  This means:

  - The production provider (e.g. OpenAI, Gemini) can be swapped by passing
    a different ``EmbeddingProvider`` implementation.
  - Unit tests inject a deterministic ``FakeEmbeddingProvider`` without
    making any network calls.
  - A future local model (e.g. sentence-transformers) requires only a new
    provider class — ``semantic_matcher.py`` is untouched.

Pipeline position
-----------------
This module MUST only receive keywords that failed all three previous steps.
It is never called with keywords that are already matched.

Similarity threshold (``SIMILARITY_THRESHOLD = 0.82``)
-------------------------------------------------------
Chosen so that semantically close pairs like "REST API" / "RESTful Services"
(cosine ≈ 0.91) or "Backend Development" / "Built scalable APIs" pass, while
unrelated pairs like "Python" / "Project Management" do not.
This value is intentionally configurable — calibrate with real production data.

Returns
-------
``SemanticMatchResult`` per comparison:
  - ``keyword``   : the resume keyword tested
  - ``matched``   : bool
  - ``matchType`` : ``"SEMANTIC"`` when matched
  - ``similarity``: cosine similarity score (0–1, for debugging)

This module does NOT:
  - run ATS scoring
  - generate AI explanations
  - import from FastAPI
  - make direct vendor SDK calls
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Configurable threshold
# ---------------------------------------------------------------------------

# Minimum cosine similarity [0.0–1.0] to accept a semantic match.
# Calibrate with production data before adjusting.
SIMILARITY_THRESHOLD: float = 0.82

# Type alias for embedding vectors
Embedding = list[float]


# ---------------------------------------------------------------------------
# Embedding Provider abstraction
# ---------------------------------------------------------------------------


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers.

    Implement this interface to plug in any embedding backend
    (OpenAI, Gemini, a local sentence-transformers model, etc.)
    without changing ``semantic_matcher.py``.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[Embedding]:
        """Generate embedding vectors for a list of texts.

        Args:
            texts: List of strings to embed.  Must be non-empty.

        Returns:
            List of embedding vectors, one per input text, in the same order.
            Each vector must have the same dimension.

        Raises:
            EmbeddingError: If the provider cannot generate embeddings.
        """


class EmbeddingError(Exception):
    """Raised when an embedding provider fails to generate vectors."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class SemanticMatchResult:
    """Result of one semantic comparison attempt."""

    keyword: str
    """The resume keyword that was tested."""
    matched: bool
    """True if similarity exceeded ``SIMILARITY_THRESHOLD``."""
    matchType: str
    """``'SEMANTIC'`` when matched, ``'MISSING'`` otherwise."""
    similarity: float
    """Cosine similarity score between the resume and JD keyword embeddings."""
    matched_jd_keyword: str | None = None
    """The JD keyword this was matched against, or None if not matched."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def match_unresolved(
    resume_keywords: list[str],
    jd_keywords: list[str],
    provider: EmbeddingProvider,
    threshold: float = SIMILARITY_THRESHOLD,
) -> tuple[list[SemanticMatchResult], list[str]]:
    """Attempt semantic matching for all unresolved resume keywords.

    Embeds all resume and JD keywords in two batched API calls, then uses
    cosine similarity to find the best JD match for each resume keyword.
    A keyword resolved here is removed from ``jd_keywords`` before the next
    resume keyword is compared (greedy first-best strategy).

    Args:
        resume_keywords: Resume keywords that survived exact/synonym/fuzzy.
        jd_keywords: JD keywords still unmatched after previous passes.
        provider: An ``EmbeddingProvider`` instance that generates vectors.
        threshold: Minimum cosine similarity to accept as a match.
            Defaults to ``SIMILARITY_THRESHOLD``.

    Returns:
        A 2-tuple of:
          - results: ``SemanticMatchResult`` for every resume keyword.
          - remaining_jd: JD keywords still unmatched after this pass.

    Raises:
        EmbeddingError: Propagated from the provider if embedding fails.
    """
    if not resume_keywords or not jd_keywords:
        results = [
            SemanticMatchResult(
                keyword=rk,
                matched=False,
                matchType="MISSING",
                similarity=0.0,
            )
            for rk in resume_keywords
        ]
        return results, list(jd_keywords)

    # Batch embed: resume keywords first, then JD keywords
    all_texts = resume_keywords + jd_keywords
    all_vectors = provider.embed(all_texts)

    resume_vectors = all_vectors[: len(resume_keywords)]
    jd_vectors = all_vectors[len(resume_keywords) :]

    remaining_jd_indices: list[int] = list(range(len(jd_keywords)))
    results: list[SemanticMatchResult] = []

    for rk, rv in zip(resume_keywords, resume_vectors):
        best_score = 0.0
        best_jd_idx: int | None = None

        for jd_idx in remaining_jd_indices:
            score = _cosine_similarity(rv, jd_vectors[jd_idx])
            if score > best_score:
                best_score = score
                best_jd_idx = jd_idx

        if best_jd_idx is not None and best_score >= threshold:
            results.append(
                SemanticMatchResult(
                    keyword=rk,
                    matched=True,
                    matchType="SEMANTIC",
                    similarity=round(best_score, 4),
                    matched_jd_keyword=jd_keywords[best_jd_idx],
                )
            )
            remaining_jd_indices.remove(best_jd_idx)
        else:
            results.append(
                SemanticMatchResult(
                    keyword=rk,
                    matched=False,
                    matchType="MISSING",
                    similarity=round(best_score, 4),
                )
            )

    remaining_jd = [jd_keywords[i] for i in remaining_jd_indices]
    return results, remaining_jd


# ---------------------------------------------------------------------------
# Private helper — pure math, no dependencies
# ---------------------------------------------------------------------------


def _cosine_similarity(a: Embedding, b: Embedding) -> float:
    """Compute the cosine similarity between two vectors.

    Args:
        a: First embedding vector.
        b: Second embedding vector.  Must have the same length as ``a``.

    Returns:
        Cosine similarity in [0.0, 1.0].  Returns 0.0 if either vector is
        the zero vector (to avoid division by zero).
    """
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)
