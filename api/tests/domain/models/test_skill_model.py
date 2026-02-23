from __future__ import annotations

from app.domain.models.skill import SkillSourceType, build_skill_key


def test_skill_source_type_contains_local() -> None:
    assert SkillSourceType.LOCAL.value == "local"
    assert SkillSourceType.GITHUB.value == "github"


def test_build_skill_key_is_stable_and_slug_based() -> None:
    key1 = build_skill_key("demo-skill", SkillSourceType.GITHUB, "github:owner/repo")
    key2 = build_skill_key("demo-skill", SkillSourceType.GITHUB, "github:owner/repo")

    assert key1 == key2
    assert key1.startswith("demo-skill--")
