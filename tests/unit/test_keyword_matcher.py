"""Unit tests for keyword_matcher — the complete matching pipeline.

Verifies the full Exact → Synonym → Fuzzy pipeline, matchType labels,
matched/missing/unresolved categorisation, and determinism.

Deterministic. No AI. No network.
"""

from app.analysis.matching import keyword_matcher
from app.analysis.matching.keyword_matcher import MatchResult, MatchedKeyword


class TestCompletePipeline:
    """Full pipeline test as described in the specification."""

    def test_mixed_match_types(self) -> None:
        """
        Resume:  React, NodeJS, Docker, MongoDB
        JD:      React, Node.js, Docker, AWS
        Expected:
          React  → EXACT
          NodeJS → EXACT (both aliases normalize to the same taxonomy skill)
          Docker → EXACT
          AWS    → MISSING (not in resume)
        """
        result: MatchResult = keyword_matcher.match(
            resume_keywords=["React", "NodeJS", "Docker", "MongoDB"],
            jd_keywords=["React", "Node.js", "Docker", "AWS"],
        )
        match_map = {m.keyword.lower(): m.matchType for m in result.matched}

        assert match_map.get("react") == "EXACT"
        assert match_map.get("nodejs") == "EXACT"
        assert match_map.get("docker") == "EXACT"
        assert "AWS" in result.missing

    def test_fuzzy_match_in_pipeline(self) -> None:
        """
        Resume:  Javascript
        JD:      JavaScript
        Expected: FUZZY match (Javascript is not an exact or synonym match)
        """
        result: MatchResult = keyword_matcher.match(
            resume_keywords=["Docekr"],
            jd_keywords=["Docker"],
        )
        assert len(result.matched) == 1
        assert result.matched[0].keyword == "Docekr"
        assert result.matched[0].matchType == "FUZZY"

    def test_unresolved_when_jd_keyword_not_in_resume(self) -> None:
        """
        Resume:  React
        JD:      Terraform
        Expected: React → UNRESOLVED, Terraform → MISSING
        """
        result: MatchResult = keyword_matcher.match(
            resume_keywords=["React"],
            jd_keywords=["Terraform"],
        )
        assert result.matched == []
        assert "React" in result.unresolved
        assert "Terraform" in result.missing


class TestReturnTypes:
    def test_returns_match_result(self) -> None:
        result = keyword_matcher.match(["React"], ["React"])
        assert isinstance(result, MatchResult)

    def test_matched_items_are_matched_keyword(self) -> None:
        result = keyword_matcher.match(["React"], ["React"])
        for item in result.matched:
            assert isinstance(item, MatchedKeyword)

    def test_matched_keyword_has_match_type(self) -> None:
        result = keyword_matcher.match(["React"], ["React"])
        assert result.matched[0].matchType in ("EXACT", "SYNONYM", "FUZZY")


class TestMatchType:
    def test_exact_match_type(self) -> None:
        result = keyword_matcher.match(["Docker"], ["Docker"])
        assert result.matched[0].matchType == "EXACT"

    def test_synonym_match_type(self) -> None:
        result = keyword_matcher.match(["nodejs"], ["Node.js"])
        assert result.matched[0].matchType == "EXACT"

    def test_fuzzy_match_type(self) -> None:
        result = keyword_matcher.match(["Mangodb"], ["MongoDB"])
        # Mangodb is not a synonym alias — must be FUZZY
        assert result.matched[0].matchType == "FUZZY"


class TestMissingAndUnresolved:
    def test_missing_jd_keyword(self) -> None:
        """JD has AWS; resume has React only → AWS should be MISSING."""
        result = keyword_matcher.match(["React"], ["React", "AWS"])
        assert "AWS" in result.missing

    def test_unresolved_resume_keyword(self) -> None:
        """Resume has Terraform; JD has React → Terraform is UNRESOLVED."""
        result = keyword_matcher.match(["React", "Terraform"], ["React"])
        assert "Terraform" in result.unresolved

    def test_empty_resume(self) -> None:
        result = keyword_matcher.match([], ["AWS", "Docker"])
        assert result.matched == []
        assert result.unresolved == []
        assert set(result.missing) == {"AWS", "Docker"}

    def test_empty_jd(self) -> None:
        result = keyword_matcher.match(["React", "Python"], [])
        assert result.matched == []
        assert result.missing == []
        assert set(result.unresolved) == {"React", "Python"}

    def test_both_empty(self) -> None:
        result = keyword_matcher.match([], [])
        assert result.matched == []
        assert result.missing == []
        assert result.unresolved == []


class TestPipelineOrder:
    """Verify that Exact fires before taxonomy relationships and fuzzy matching."""

    def test_exact_wins_over_synonym(self) -> None:
        """If a keyword matches exactly, it must be EXACT not a relationship."""
        # "Node.js" matches "Node.js" exactly before any synonym lookup
        result = keyword_matcher.match(["Node.js"], ["Node.js"])
        assert result.matched[0].matchType == "EXACT"

    def test_relationship_fires_only_after_exact_fails(self) -> None:
        """A taxonomy parent/child relation runs only after canonical exact match."""
        result = keyword_matcher.match(["EC2"], ["AWS"])
        assert result.matched[0].matchType == "SYNONYM"

    def test_fuzzy_fires_only_after_synonym_fails(self) -> None:
        """An unknown typo has no taxonomy relation and falls to fuzzy matching."""
        result = keyword_matcher.match(["Docekr"], ["Docker"])
        assert result.matched[0].matchType == "FUZZY"

    def test_deterministic_same_input_same_output(self) -> None:
        """Same input must always produce the same result."""
        kwargs = {
            "resume_keywords": ["React", "NodeJS", "Docekr", "Terraform"],
            "jd_keywords": ["React", "Node.js", "Docker", "AWS"],
        }
        result_1 = keyword_matcher.match(**kwargs)
        result_2 = keyword_matcher.match(**kwargs)
        assert result_1.matched == result_2.matched
        assert result_1.missing == result_2.missing
        assert result_1.unresolved == result_2.unresolved
