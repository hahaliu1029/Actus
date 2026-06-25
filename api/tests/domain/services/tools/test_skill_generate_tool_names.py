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


# --- EPIC-FIX-2: cross-pool generated-name collision carve-out regression --- #
#
# Two DIFFERENT (slug, tool) pairs that NORMALIZE to the SAME base
# ``skill_foo_bar_baz``:
#   - NON-member skill: slug "foo-bar", tool "baz"  → skill_foo_bar_baz
#   - MEMBER skill:     slug "foo",     tool "bar-baz" → skill_foo_bar_baz
# The expander predicts the MEMBER's tool name from a per-member EMPTY index
# (generate_tool_names over only the member's skills) → the BARE name. The child
# builds tools over the FULL selected pool. If the member is indexed AFTER the
# colliding non-member, the member gets ``_1`` and the prediction breaks (the §12
# #1 risk / §13 R7-2 violation). The floor must place the member FIRST so the
# child reproduces the bare prediction.


def _collide_skills():
    import asyncio  # noqa: F401  (imported lazily by callers below)

    member = _skill("foo", [{"name": "bar-baz"}])         # → skill_foo_bar_baz
    nonmember = _skill("foo-bar", [{"name": "baz"}])      # → skill_foo_bar_baz
    return member, nonmember


def test_expander_member_name_equals_child_built_name_after_floor():
    """The expander's per-member generate_tool_names prediction EQUALS the member's
    ACTUAL built name in the child once the floor prepends the member skill —
    and the built-vs-bound assertion does NOT raise."""
    import asyncio

    from app.domain.services.agent_task_runner import (
        _apply_member_skill_floor,
        _assert_member_tools_built,
    )

    member, nonmember = _collide_skills()

    # 1) Expander side: predict the member's tool names from an EMPTY index over
    #    ONLY the member's skills (mirrors team_expander.resolve_team_member_map).
    member_skill_tools = set(SkillTool.generate_tool_names([member]))
    assert member_skill_tools == {"skill_foo_bar_baz"}   # bare base, no _N suffix

    # 2) Child side: a query-driven selector picked the colliding NON-member skill;
    #    the floor unions the MEMBER skill back — and places it FIRST.
    pool = [nonmember, member]
    selected = [nonmember]
    floored = _apply_member_skill_floor(selected, pool, member_slugs=(member.slug,))
    assert [s.slug for s in floored] == ["foo", "foo-bar"]  # member indexed first

    # 3) Child builds the real SkillTool over the floored selection (same order
    #    that reaches initialize() via _apply_preselected_skills dedup).
    st = SkillTool(sandbox=MagicMock(), mcp_tool=MagicMock(), a2a_tool=MagicMock())
    asyncio.run(st.initialize(floored))
    built = {t["function"]["name"] for t in st.get_tools()}
    bindings = {name: b["skill"].slug for name, b in st._tool_bindings.items()}

    # The member's predicted name IS among the actually-built names AND is bound
    # to the MEMBER skill (not the colliding non-member) → carve-out holds.
    assert member_skill_tools <= built, (
        f"member prediction {member_skill_tools} not in built {built}"
    )
    assert bindings["skill_foo_bar_baz"] == "foo"          # member owns the bare name
    assert bindings["skill_foo_bar_baz_1"] == "foo-bar"    # non-member absorbed _N
    _assert_member_tools_built(required=member_skill_tools, built_names=built)


def test_append_last_order_misidentifies_member_tool_proving_floor_fix():
    """RED-evidence guard: with the OLD append-LAST order the member skill is
    indexed AFTER the colliding non-member, so the bare name the expander
    predicted (``skill_foo_bar_baz``) is bound to the WRONG skill (the
    non-member), and the member's REAL tool gets ``_1``. This is the §12 #1 risk
    in its worst form: ``_assert_member_tools_built`` is membership-based and does
    NOT raise (the bare name exists in the built set) — yet it is the non-member's
    tool, so the member's actual tool (``skill_foo_bar_baz_1``) is silently absent
    from the member allowlist. This test pins WHY the prepend fix is required by
    asserting the broken append-last binding, NOT the production helper."""
    import asyncio

    member, nonmember = _collide_skills()
    member_skill_tools = set(SkillTool.generate_tool_names([member]))
    assert member_skill_tools == {"skill_foo_bar_baz"}

    # Simulate the OLD append-LAST selection order (non-member first, member last).
    append_last_order = [nonmember, member]
    st = SkillTool(sandbox=MagicMock(), mcp_tool=MagicMock(), a2a_tool=MagicMock())
    asyncio.run(st.initialize(append_last_order))
    bindings = {name: b["skill"].slug for name, b in st._tool_bindings.items()}

    # The predicted bare name is bound to the NON-member skill (the bug):
    assert bindings["skill_foo_bar_baz"] == "foo-bar"      # non-member stole it
    # ...and the MEMBER's real tool was demoted to the _1 suffix, which the
    # expander never predicted ⇒ NOT in the member allowlist ⇒ silent absence.
    assert bindings["skill_foo_bar_baz_1"] == "foo"        # member demoted
    assert "skill_foo_bar_baz_1" not in member_skill_tools
