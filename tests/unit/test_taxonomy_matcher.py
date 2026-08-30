"""Unit tests for taxonomy normalization in the four-stage keyword matcher."""

from app.analysis.matching import keyword_matcher


def test_aliases_are_canonicalized_before_exact_matching() -> None:
    result = keyword_matcher.match(["ReactJS", "node"], ["React.js", "Node.js"])

    assert [match.matchType for match in result.matched] == ["EXACT", "EXACT"]
    assert result.missing == []


def test_exact_match_pairs_keep_the_correct_original_jd_term() -> None:
    result = keyword_matcher.match(
        ["JavaScript", "React", "HTML", "CSS"],
        ["HTML", "CSS", "React", "JavaScript"],
    )

    assert [(match.keyword, match.matched_jd_keyword) for match in result.matched] == [
        ("JavaScript", "JavaScript"),
        ("React", "React"),
        ("HTML", "HTML"),
        ("CSS", "CSS"),
    ]


def test_parent_child_and_related_skills_use_stage_two() -> None:
    result = keyword_matcher.match(
        ["Amazon EC2", "Git", "Pydantic"],
        ["AWS", "GitHub", "FastAPI"],
    )

    assert [match.matchType for match in result.matched] == ["SYNONYM", "SYNONYM", "SYNONYM"]
    assert result.missing == []


def test_unknown_terms_do_not_gain_a_taxonomy_relationship() -> None:
    result = keyword_matcher.match(["Blorfizz"], ["XYZ123"])

    assert result.matched == []
    assert result.unresolved == ["Blorfizz"]
    assert result.missing == ["XYZ123"]
