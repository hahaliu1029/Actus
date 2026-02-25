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


def test_tokenize_cjk_bigram_examples_snapshot() -> None:
    selector = SkillSelector()

    assert selector._tokenize("流程图") == {"流程图", "流程", "程图"}
    assert selector._tokenize("画流程图") == {"画流程图", "画流", "流程", "程图"}
    assert selector._tokenize("sql优化") == {"sql", "优化"}
    assert selector._tokenize("查日志") == {"查日志", "查日", "日志"}
    assert selector._tokenize("图") == {"图"}


def test_tokenize_mixed_sql_cjk() -> None:
    selector = SkillSelector()
    tokens = selector._tokenize("请做 SQL 优化并查日志")
    assert "sql" in tokens
    assert "优化" in tokens
    assert "查日" in tokens
    assert "日志" in tokens


def test_no_unigram_for_multi_char_cjk_span() -> None:
    selector = SkillSelector()
    tokens = selector._tokenize("好的继续")
    assert "好的继续" in tokens
    assert "好的" in tokens
    assert "继续" in tokens
    assert "好" not in tokens
    assert "的" not in tokens


def test_select_with_meta_uses_dynamic_threshold_for_short_intent() -> None:
    selector = SkillSelector(default_top_k=12, base_threshold=3)
    skills = [
        _build_skill("sql", "SQL", "sql optimize", "query and index"),
        _build_skill("draw", "Draw", "diagram", "draw flow chart"),
    ]

    meta = selector.select_with_meta(skills=skills, user_message="sql优化")

    assert meta.token_count == 2
    assert meta.effective_threshold == 1
    assert meta.has_positive_match is True
    assert meta.selected_skills[0].id == "sql"
