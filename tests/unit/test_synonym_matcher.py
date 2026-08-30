"""Unit tests for synonym matching via keyword_matcher's synonym pass.

These tests exercise the synonym resolution logic directly by calling
keyword_matcher.match() and checking that SYNONYM matchType is returned.
They intentionally do NOT import exact_matcher or fuzzy_matcher directly —
callers should always go through keyword_matcher.

Deterministic. No AI. No network.
"""

from app.analysis.matching import keyword_matcher
from app.analysis.matching.keyword_matcher import MatchResult


class TestSynonymMatch:
    def test_node_alias_matches_nodejs(self) -> None:
        """'node' on resume should match 'Node.js' in JD via synonym."""
        result: MatchResult = keyword_matcher.match(["node"], ["Node.js"])
        assert len(result.matched) == 1
        assert result.matched[0].keyword == "node"
        assert result.matched[0].matchType == "SYNONYM"
        assert result.missing == []

    def test_nodejs_alias_matches_node(self) -> None:
        """'nodejs' on resume should match 'node' in JD via synonym."""
        result: MatchResult = keyword_matcher.match(["nodejs"], ["node"])
        assert len(result.matched) == 1
        assert result.matched[0].matchType == "SYNONYM"

    def test_js_alias_matches_javascript(self) -> None:
        """'JS' on resume should match 'JavaScript' in JD via synonym."""
        result: MatchResult = keyword_matcher.match(["JS"], ["JavaScript"])
        assert len(result.matched) == 1
        assert result.matched[0].matchType == "SYNONYM"

    def test_react_alias_matches_reactjs(self) -> None:
        """'React' should match 'reactjs' in JD via synonym."""
        result: MatchResult = keyword_matcher.match(["React"], ["reactjs"])
        assert len(result.matched) == 1
        assert result.matched[0].matchType == "SYNONYM"

    def test_mongodb_alias_matches_mongo(self) -> None:
        """'MongoDB' should match 'mongo' in JD."""
        result: MatchResult = keyword_matcher.match(["MongoDB"], ["mongo"])
        assert len(result.matched) == 1
        assert result.matched[0].matchType == "SYNONYM"

    def test_postgres_alias_matches_postgresql(self) -> None:
        """'postgres' should match 'PostgreSQL'."""
        result: MatchResult = keyword_matcher.match(["postgres"], ["PostgreSQL"])
        assert len(result.matched) == 1
        assert result.matched[0].matchType == "SYNONYM"

    def test_cicd_alias_matches_continuous_integration(self) -> None:
        """'CI/CD' should match 'continuous integration' via synonym."""
        result: MatchResult = keyword_matcher.match(
            ["CI/CD"], ["continuous integration"]
        )
        assert len(result.matched) == 1
        assert result.matched[0].matchType == "SYNONYM"

    def test_distributed_systems_matches_microservices(self) -> None:
        """Fintech domain aliases resolve before fuzzy or semantic matching."""
        result: MatchResult = keyword_matcher.match(
            ["Microservices"], ["Distributed Systems"]
        )
        assert len(result.matched) == 1
        assert result.matched[0].matchType == "SYNONYM"

    def test_payment_orchestration_matches_payment_gateway(self) -> None:
        """Payment orchestration aliases resolve through the synonym stage."""
        result: MatchResult = keyword_matcher.match(
            ["Payment Gateway"], ["Payment Orchestration"]
        )
        assert len(result.matched) == 1
        assert result.matched[0].matchType == "SYNONYM"

    def test_overlapping_aliases_preserve_all_synonym_groups(self) -> None:
        """An alias must not lose CI/CD membership to a later map entry."""
        result: MatchResult = keyword_matcher.match(["GitHub Actions"], ["CI/CD"])
        assert len(result.matched) == 1
        assert result.matched[0].matchType == "SYNONYM"

    def test_unknown_keyword_not_synonym_matched(self) -> None:
        """Keywords not in the synonym map should not be matched via synonyms."""
        result: MatchResult = keyword_matcher.match(
            ["Blorfizz"], ["XYZ123"]
        )
        assert result.matched == []
        assert "Blorfizz" in result.unresolved
        assert "XYZ123" in result.missing
