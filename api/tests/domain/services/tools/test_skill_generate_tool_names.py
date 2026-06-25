from unittest.mock import MagicMock

from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType
from app.domain.services.tools.skill import SkillTool


def _skill(slug, tools, runtime=SkillRuntimeType.MCP):
    # [codex-R1-F3] Skill requires source_type + source_ref (no defaults).
    return Skill(
        id=slug, slug=slug, name=slug, runtime_type=runtime, enabled=True,
        source_type=SkillSourceType.LOCAL, source_ref=slug,
        manifest={"tools": tools},
    )


def test_generated_names_are_slug_prefixed():
    s = _skill("repo-map", [{"name": "search"}, {"name": "tree"}])
    names = SkillTool.generate_tool_names([s])
    assert names == ["skill_repo_map_search", "skill_repo_map_tree"]


def test_resolver_matches_a_real_skilltool_get_tools():
    # ANTI-DRIFT LOCK (§17): the pure resolver MUST equal what a constructed
    # SkillTool.initialize/get_tools produces for the same skills.
    import asyncio
    s = _skill("repo-map", [{"name": "search"}, {"name": "tree"}])
    st = SkillTool(sandbox=MagicMock(), mcp_tool=MagicMock(), a2a_tool=MagicMock())
    # Brief used the deprecated asyncio.get_event_loop().run_until_complete(...),
    # which raises "no current event loop" on Python 3.12 + pytest-asyncio strict
    # mode. asyncio.run(...) is the repo-wide idiom and preserves the test intent.
    asyncio.run(st.initialize([s]))
    built = [t["function"]["name"] for t in st.get_tools()]
    assert SkillTool.generate_tool_names([s]) == built


def test_model_invocable_false_skipped():
    s = _skill("repo-map", [{"name": "search", "policy": {"model_invocable": False}}])
    assert SkillTool.generate_tool_names([s]) == []
