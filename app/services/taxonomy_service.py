"""In-memory access to the vendored Tanova Open Skills Taxonomy.

The taxonomy is loaded once and indexed at application startup.  All public
lookup methods are read-only and deterministic, so they are safe to use in
the synchronous ATS matching and scoring pipeline.
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.logging import logger


_TAXONOMY_PATH = Path(__file__).resolve().parents[2] / "data" / "taxonomy.json"
_EXTENSIONS_PATH = Path(__file__).resolve().parents[2] / "data" / "taxonomy_extensions.json"
_FULL_WEIGHT_CATEGORIES = frozenset(
    {
        "programming_languages",
        "frontend_development",
        "backend_development",
        "backend_frameworks",
        "databases",
        "cloud",
        "cloud_platforms",
        "data_infrastructure",
        "ai_ml",
        "software_architecture",
        "architecture",
    }
)
_PREFERRED_CATEGORIES = frozenset({"tools", "devops", "testing", "design_tools", "cms"})
_FEEDBACK_CATEGORIES = frozenset(
    {
        "unknown",
        "soft_skills",
        "culture",
        "management",
        "project_management",
        "strategy",
        "sales",
        "marketing",
        "finance",
        "domain_knowledge",
    }
)


def _normalise(value: str) -> str:
    """Return a stable key for taxonomy lookups."""
    return " ".join(value.casefold().split())


@dataclass(frozen=True)
class TaxonomySkill:
    """The indexed subset of an upstream taxonomy skill record."""

    skill_id: str
    canonical_name: str
    category: str
    subcategory: str
    tags: tuple[str, ...]
    aliases: tuple[str, ...]
    parent_skills: tuple[str, ...]
    child_skills: tuple[str, ...]
    related_skills: tuple[str, ...]
    transferability: dict[str, float]


class TaxonomyService:
    """Read-only singleton-ready service backed by ``data/taxonomy.json``."""

    def __init__(self, taxonomy_path: Path = _TAXONOMY_PATH) -> None:
        try:
            raw = json.loads(taxonomy_path.read_text(encoding="utf-8"))
            self.version = str(raw["version"])
            self._skills_by_id = self._flatten_skills(raw)
            self._skills_by_id.update(self._load_extensions())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.critical("[Taxonomy] Failed to load %s: %s", taxonomy_path, exc)
            raise RuntimeError(f"Unable to load required skills taxonomy: {taxonomy_path}") from exc

        if not self._skills_by_id:
            logger.critical("[Taxonomy] %s did not contain any skills", taxonomy_path)
            raise RuntimeError("Required skills taxonomy is empty")

        self._skill_by_lookup = self._build_lookup(self._skills_by_id)
        self._term_pattern = self._build_term_pattern(self._skill_by_lookup)

    @staticmethod
    def _flatten_skills(raw: dict[str, Any]) -> dict[str, TaxonomySkill]:
        skills: dict[str, TaxonomySkill] = {}
        for category in raw.get("categories", {}).values():
            for subcategory in category.get("subcategories", {}).values():
                for item in subcategory.get("skills", []):
                    skill = TaxonomySkill(
                        skill_id=str(item["id"]),
                        canonical_name=str(item["canonical_name"]),
                        category=str(item.get("category", "unknown")),
                        subcategory=str(item.get("subcategory", "unknown")),
                        tags=tuple(str(tag) for tag in item.get("tags", [])),
                        aliases=tuple(str(alias) for alias in item.get("aliases", [])),
                        parent_skills=tuple(str(parent) for parent in item.get("parent_skills", [])),
                        child_skills=tuple(str(child) for child in item.get("child_skills", [])),
                        related_skills=tuple(str(related) for related in item.get("related_skills", [])),
                        transferability={
                            str(skill_id): float(score)
                            for skill_id, score in item.get("transferability", {}).items()
                        },
                    )
                    skills[skill.skill_id] = skill
        return skills

    @staticmethod
    def _load_extensions() -> dict[str, TaxonomySkill]:
        """Load schema-compatible project skills missing from the source release."""
        raw = json.loads(_EXTENSIONS_PATH.read_text(encoding="utf-8"))
        extensions: dict[str, TaxonomySkill] = {}
        for item in raw.get("skills", []):
            skill = TaxonomySkill(
                skill_id=str(item["id"]),
                canonical_name=str(item["canonical_name"]),
                category=str(item.get("category", "unknown")),
                subcategory=str(item.get("subcategory", "unknown")),
                tags=tuple(str(tag) for tag in item.get("tags", [])),
                aliases=tuple(str(alias) for alias in item.get("aliases", [])),
                parent_skills=tuple(str(parent) for parent in item.get("parent_skills", [])),
                child_skills=tuple(str(child) for child in item.get("child_skills", [])),
                related_skills=tuple(str(related) for related in item.get("related_skills", [])),
                transferability={
                    str(skill_id): float(score)
                    for skill_id, score in item.get("transferability", {}).items()
                },
            )
            extensions[skill.skill_id] = skill
        return extensions

    @staticmethod
    def _build_lookup(skills: dict[str, TaxonomySkill]) -> dict[str, TaxonomySkill]:
        lookup: dict[str, TaxonomySkill] = {}
        for skill in skills.values():
            for value in (skill.skill_id, skill.canonical_name, *skill.aliases):
                lookup.setdefault(_normalise(value), skill)
        return lookup

    @staticmethod
    def _build_term_pattern(lookup: dict[str, TaxonomySkill]) -> re.Pattern[str]:
        terms = sorted((term for term in lookup if len(term) > 1), key=len, reverse=True)
        escaped = "|".join(re.escape(term) for term in terms)
        return re.compile(rf"(?<![a-z0-9+#.])(?:{escaped})(?![a-z0-9+#])", re.IGNORECASE)

    def normalize(self, skill: str) -> str:
        """Return a taxonomy canonical name, or the cleaned unknown value."""
        cleaned = " ".join(skill.split())
        match = self._skill_by_lookup.get(_normalise(cleaned))
        return match.canonical_name if match else cleaned

    def get_category(self, skill: str) -> str:
        """Return the taxonomy subcategory, with cloud promoted from tags."""
        record = self._skill_by_lookup.get(_normalise(skill))
        if record is None:
            return "unknown"
        if "cloud" in {tag.casefold() for tag in record.tags}:
            return "cloud"
        return record.subcategory or "unknown"

    def is_parent_of(self, child: str, parent: str) -> bool:
        """Return whether ``parent`` occurs anywhere above ``child`` in the graph."""
        child_record = self._skill_by_lookup.get(_normalise(child))
        parent_record = self._skill_by_lookup.get(_normalise(parent))
        if child_record is None or parent_record is None or child_record.skill_id == parent_record.skill_id:
            return False

        pending = deque(child_record.parent_skills)
        visited: set[str] = set()
        while pending:
            candidate_id = pending.popleft()
            if candidate_id in visited:
                continue
            if candidate_id == parent_record.skill_id:
                return True
            visited.add(candidate_id)
            candidate = self._skills_by_id.get(candidate_id)
            if candidate is not None:
                pending.extend(candidate.parent_skills)
        return False

    def are_related(self, first: str, second: str) -> bool:
        """Return whether two known skills have a taxonomy graph relationship."""
        first_record = self._skill_by_lookup.get(_normalise(first))
        second_record = self._skill_by_lookup.get(_normalise(second))
        if first_record is None or second_record is None:
            return False
        if first_record.skill_id == second_record.skill_id:
            return True
        return (
            self.is_parent_of(first, second)
            or self.is_parent_of(second, first)
            or self.get_transferability(first, second) >= 0.75
        )

    def get_transferability(self, source: str, target: str) -> float:
        """Return the directed taxonomy transferability score in the range 0–1."""
        source_record = self._skill_by_lookup.get(_normalise(source))
        target_record = self._skill_by_lookup.get(_normalise(target))
        if source_record is None or target_record is None:
            return 0.0
        return max(
            source_record.transferability.get(target_record.skill_id, 0.0),
            target_record.transferability.get(source_record.skill_id, 0.0),
        )

    def get_weight(self, skill: str) -> float:
        """Return the scoring weight implied by the taxonomy category."""
        category = self.get_category(skill)
        if category in _FULL_WEIGHT_CATEGORIES:
            return 1.0
        if category in _PREFERRED_CATEGORIES:
            return 0.4
        return 0.1

    def is_required_skill(self, skill: str) -> bool:
        """Return whether a recognized skill belongs in the required denominator."""
        return self.get_category(skill) in _FULL_WEIGHT_CATEGORIES

    def is_preferred_skill(self, skill: str) -> bool:
        """Return whether a recognized skill is optional bonus evidence."""
        return self.get_category(skill) in _PREFERRED_CATEGORIES

    def is_feedback_only(self, skill: str) -> bool:
        """Return whether a term must be excluded from ATS score inputs."""
        return self.get_category(skill) in _FEEDBACK_CATEGORIES

    def recognized_terms(self, text: str) -> list[str]:
        """Extract stable, canonical taxonomy skills occurring in arbitrary text."""
        result: list[str] = []
        seen: set[str] = set()
        for match in self._term_pattern.finditer(text):
            canonical = self.normalize(match.group(0))
            key = _normalise(canonical)
            if key not in seen:
                seen.add(key)
                result.append(canonical)
        return result


_service: TaxonomyService | None = None


def initialize_taxonomy_service() -> TaxonomyService:
    """Create the global service once; errors deliberately prevent startup."""
    global _service
    if _service is None:
        _service = TaxonomyService()
        logger.info(
            "[Taxonomy] Loaded Tanova skills taxonomy v%s with %d skills",
            _service.version,
            len(_service._skills_by_id),
        )
    return _service


def get_taxonomy_service() -> TaxonomyService:
    """Return the startup singleton, initializing once for direct library use."""
    return initialize_taxonomy_service()
