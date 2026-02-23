from __future__ import annotations

from app.application.services.skill_selector import SkillSelector
from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType


def _build_skill(
    skill_id: str,
    name: str,
    description: str,
    skill_md: str,
) -> Skill:
    return Skill(
        id=skill_id,
        slug=skill_id,
        name=name,
        description=description,
        source_type=SkillSourceType.LOCAL,
        source_ref=f"local:{skill_id}",
        runtime_type=SkillRuntimeType.NATIVE,
        manifest={
            "name": name,
            "runtime_type": "native",
            "skill_md": skill_md,
            "tools": [{"name": "run", "description": description}],
        },
        enabled=True,
    )


def test_skill_selector_prioritizes_keyword_match() -> None:
    selector = SkillSelector(default_top_k=12)
    skills = [
        _build_skill("code-review", "Code Review", "review pull request", "review coding style"),
        _build_skill("travel", "Travel", "plan trip", "book hotel"),
    ]

    selected = selector.select(
        skills=skills,
        user_message="请帮我 review 这个 PR 的代码",
    )

    assert selected
    assert selected[0].id == "code-review"


def test_skill_selector_respects_top_k() -> None:
    selector = SkillSelector(default_top_k=1)
    skills = [
        _build_skill("s1", "One", "desc", "guide"),
        _build_skill("s2", "Two", "desc", "guide"),
    ]

    selected = selector.select(skills=skills, user_message="random")
    assert len(selected) == 1


def test_skill_selector_prefers_context_blob_when_available() -> None:
    selector = SkillSelector(default_top_k=12)
    skills = [
        _build_skill("pptx", "PPTX", "slides", "generic content"),
        _build_skill("sql", "SQL", "db", "generic content"),
    ]
    skills[0].manifest["context_blob"] = "build slide deck with template"
    skills[1].manifest["context_blob"] = "optimize query and index"

    selected = selector.select(skills=skills, user_message="请帮我做一个 slide deck")

    assert selected
    assert selected[0].id == "pptx"
