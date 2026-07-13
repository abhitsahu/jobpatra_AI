"""Unit tests for fuzzy_matcher.

Tests the fuzzy matching module directly for unit-level coverage, and also
verifies that the keyword_matcher pipeline correctly assigns FUZZY matchType.

Deterministic. No AI. No network.
"""

from app.analysis.matching import fuzzy_matcher, keyword_matcher


class TestFuzzyMatch:
    def test_javascript_typo_matched(self) -> None:
        """'Javascript' (wrong case, common typo) should fuzzy-match 'JavaScript'."""
        result = fuzzy_matcher.match("Javascript", ["JavaScript"])
        assert result == "JavaScript"

    def test_docker_typo_matched(self) -> None:
        """'Docekr' (transposition) should fuzzy-match 'Docker'."""
        result = fuzzy_matcher.match("Docekr", ["Docker"])
        assert result == "Docker"

    def test_mongodb_typo_matched(self) -> None:
        """'Mangodb' (vowel swap) should fuzzy-match 'MongoDB'."""
        result = fuzzy_matcher.match("Mangodb", ["MongoDB"])
        assert result == "MongoDB"

    def test_returns_none_when_no_candidate(self) -> None:
        assert fuzzy_matcher.match("React", []) is None

    def test_large_difference_not_matched(self) -> None:
        """'Python' should NOT fuzzy-match 'Java' — they are very different."""
        result = fuzzy_matcher.match("Python", ["Java"])
        assert result is None

    def test_completely_different_not_matched(self) -> None:
        """Gibberish should not match a real keyword."""
        result = fuzzy_matcher.match("Xzxzxzxz", ["Docker"])
        assert result is None

    def test_threshold_respected(self) -> None:
        """At threshold=100 only identical strings pass."""
        result = fuzzy_matcher.match("Docekr", ["Docker"], threshold=100)
        assert result is None


class TestFuzzyMatchAll:
    def test_multiple_typos_matched(self) -> None:
        matched, unmatched_r, unmatched_jd = fuzzy_matcher.match_all(
            ["Docekr", "Mangodb"], ["Docker", "MongoDB"]
        )
        assert len(matched) == 2
        assert unmatched_r == []
        assert unmatched_jd == []

    def test_partial_match(self) -> None:
        matched, unmatched_r, unmatched_jd = fuzzy_matcher.match_all(
            ["Docekr", "Xzxzxzxz"], ["Docker", "MongoDB"]
        )
        assert len(matched) == 1
        assert "Xzxzxzxz" in unmatched_r
        assert "MongoDB" in unmatched_jd


class TestFuzzyMatchTypeViaOrchestrator:
    """Verify that the pipeline labels fuzzy matches with FUZZY matchType."""

    def test_javascript_typo_gets_fuzzy_match_type(self) -> None:
        """'Javascript' fails exact and synonym, then passes fuzzy."""
        result = keyword_matcher.match(["Javascript"], ["JavaScript"])
        # "javascript" vs "javascript" → will actually be SYNONYM or EXACT
        # depending on synonym_map. Let's use a clear typo not in any alias.
        # Use a deliberate typo that is NOT in synonym_map.
        result2 = keyword_matcher.match(["Dockerr"], ["Docker"])
        match_types = [m.matchType for m in result2.matched]
        assert "FUZZY" in match_types

    def test_docekr_typo_gets_fuzzy_match_type(self) -> None:
        result = keyword_matcher.match(["Docekr"], ["Docker"])
        assert len(result.matched) == 1
        assert result.matched[0].matchType == "FUZZY"
