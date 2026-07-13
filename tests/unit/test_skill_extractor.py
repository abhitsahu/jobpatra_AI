"""Unit tests for skill_extractor.

Deterministic. No AI. No network.
"""

from app.analysis.extraction import skill_extractor
from app.analysis.extraction.skill_extractor import SkillExtractionResult


class TestSkillIdentification:
    def test_returns_skill_extraction_result(self) -> None:
        result = skill_extractor.extract("React Node.js Docker")
        assert isinstance(result, SkillExtractionResult)

    def test_known_skill_identified(self) -> None:
        result = skill_extractor.extract("Python developer")
        canonicals = [s.canonical for s in result.skills]
        assert "Python" in canonicals

    def test_multiple_skills_identified(self) -> None:
        result = skill_extractor.extract("React Node.js Docker PostgreSQL")
        canonicals = [s.canonical for s in result.skills]
        assert "React" in canonicals
        assert "Node.js" in canonicals
        assert "Docker" in canonicals
        assert "PostgreSQL" in canonicals

    def test_alias_resolves_to_canonical(self) -> None:
        """'nodejs' alias should resolve to 'Node.js' canonical."""
        result = skill_extractor.extract("nodejs developer")
        canonicals = [s.canonical for s in result.skills]
        assert "Node.js" in canonicals

    def test_case_insensitive_alias_matching(self) -> None:
        result = skill_extractor.extract("PYTHON and DOCKER")
        canonicals = [s.canonical for s in result.skills]
        assert "Python" in canonicals
        assert "Docker" in canonicals


class TestUnknownTerms:
    def test_unknown_term_ignored(self) -> None:
        result = skill_extractor.extract("Blorfizz XYZ123 foobar")
        assert result.skills == []

    def test_empty_string_returns_empty(self) -> None:
        result = skill_extractor.extract("")
        assert result.skills == []
        assert result.by_category == {}


class TestCategorization:
    def test_skills_categorized_correctly(self) -> None:
        result = skill_extractor.extract("React Vue.js")
        assert "Frontend" in result.by_category

    def test_docker_in_devops_category(self) -> None:
        result = skill_extractor.extract("Docker Kubernetes")
        assert "DevOps" in result.by_category

    def test_by_category_contains_canonicals(self) -> None:
        result = skill_extractor.extract("PostgreSQL MongoDB")
        assert "Databases" in result.by_category
        assert "PostgreSQL" in result.by_category["Databases"]
        assert "MongoDB" in result.by_category["Databases"]


class TestDeduplication:
    def test_same_skill_via_two_aliases_not_duplicated(self) -> None:
        """'react' and 'reactjs' both map to 'React' — should appear once."""
        result = skill_extractor.extract("react reactjs")
        canonicals = [s.canonical for s in result.skills]
        assert canonicals.count("React") == 1

    def test_multiword_skill_detected(self) -> None:
        """'machine learning' is a bigram skill."""
        result = skill_extractor.extract("machine learning engineer")
        canonicals = [s.canonical for s in result.skills]
        assert "Machine Learning" in canonicals
