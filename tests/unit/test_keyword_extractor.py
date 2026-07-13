"""Unit tests for keyword_extractor.

Deterministic. No AI. No network.
"""

from app.analysis.extraction import keyword_extractor


class TestBasicExtraction:
    def test_returns_list(self) -> None:
        assert isinstance(keyword_extractor.extract("React developer"), list)

    def test_extracts_known_terms(self) -> None:
        result = keyword_extractor.extract("React Developer with Docker and AWS experience.")
        assert "React" in result
        assert "Docker" in result
        assert "AWS" in result

    def test_empty_string_returns_empty_list(self) -> None:
        assert keyword_extractor.extract("") == []

    def test_stop_words_only_returns_empty(self) -> None:
        result = keyword_extractor.extract("and the a an or but")
        assert result == []


class TestDeduplication:
    def test_exact_duplicate_removed(self) -> None:
        result = keyword_extractor.extract("Python Python Python")
        assert result.count("Python") == 1

    def test_case_insensitive_deduplication(self) -> None:
        result = keyword_extractor.extract("python Python PYTHON")
        # Only one variant should appear — set membership, not count
        lower_list = [k.lower() for k in result]
        assert lower_list.count("python") == 1

    def test_first_seen_casing_preserved(self) -> None:
        result = keyword_extractor.extract("React react REACT")
        assert result[0] == "React"

    def test_no_duplicates_across_sentence(self) -> None:
        text = "Docker is great. I love Docker and Docker is fast."
        result = keyword_extractor.extract(text)
        assert result.count("Docker") == 1


class TestFiltering:
    def test_single_char_tokens_excluded(self) -> None:
        result = keyword_extractor.extract("a b c d Python")
        assert "a" not in result
        assert "b" not in result
        assert "Python" in result

    def test_stop_words_excluded(self) -> None:
        result = keyword_extractor.extract("I am responsible for React development")
        assert "I" not in result
        assert "am" not in result
        assert "for" not in result
        assert "React" in result

    def test_preserves_technical_terms_with_dots(self) -> None:
        result = keyword_extractor.extract("Node.js and ASP.NET developer")
        assert "Node.js" in result or "Node" in result


class TestOutputOrder:
    def test_first_seen_order_preserved(self) -> None:
        result = keyword_extractor.extract("Docker React Python")
        assert result.index("Docker") < result.index("React")
        assert result.index("React") < result.index("Python")
