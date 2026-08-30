"""Tests for the startup-owned Tanova taxonomy service."""

import pytest

from app.services.taxonomy_service import TaxonomyService, get_taxonomy_service


def test_normalization_categories_hierarchy_and_weights() -> None:
    taxonomy = get_taxonomy_service()

    assert taxonomy.normalize("ReactJS") == "React"
    assert taxonomy.get_category("AWS") == "cloud"
    assert taxonomy.is_parent_of("RDS", "AWS")
    assert taxonomy.get_transferability("Microservices", "Distributed Systems") >= 0.75
    assert taxonomy.get_weight("Python") == 1.0
    assert taxonomy.get_weight("Black") == 0.4
    assert taxonomy.get_weight("Mypy") == 0.4
    assert taxonomy.get_weight("enthusiastic") == 0.1


def test_invalid_taxonomy_fails_fast(tmp_path) -> None:
    invalid_taxonomy = tmp_path / "taxonomy.json"
    invalid_taxonomy.write_text("not json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Unable to load required skills taxonomy"):
        TaxonomyService(invalid_taxonomy)
