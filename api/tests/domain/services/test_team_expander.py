import pytest

from app.domain.models.agent_team import TeamBundle, TeamMember
from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType
from app.domain.services.team_expander import (
    MemberCapability,
    TeamCapabilityError,
    resolve_team_member_map,
)


class _FakeSkillRepo:
    def __init__(self, skills):
        self._by_slug = {s.slug: s for s in skills}

    async def get_by_slug(self, slug):
        return self._by_slug.get(slug)


def _skill(slug, runtime=SkillRuntimeType.MCP, tools=None, enabled=True, policy=None):
    manifest = {"tools": tools or [{"name": "go"}]}
    if policy is not None:
        manifest["policy"] = policy
    # [codex-R1-F3] Skill requires source_type + source_ref (no defaults).
    return Skill(id=slug, slug=slug, name=slug, runtime_type=runtime, enabled=enabled,
                 source_type=SkillSourceType.LOCAL, source_ref=slug, manifest=manifest)


def _team(*members):
    return TeamBundle(slug="t", name="T", members=tuple(members))


@pytest.mark.asyncio
async def test_mcp_member_skill_with_c2_child_safe_resolves():
    repo = _FakeSkillRepo([_skill("safe-mcp", policy={"c2_child_safe": True})])
    m = TeamMember(role="r", description="d", system_prompt="p", skills=("safe-mcp",))
    out = await resolve_team_member_map(team=_team(m), skill_repository=repo)
    cap = out["r"]
    assert isinstance(cap, MemberCapability)
    assert cap.member_skill_tools == frozenset({"skill_safe_mcp_go"})
    assert cap.member_skill_slugs == ("safe-mcp",)
    assert cap.native_skill_slugs == ()  # mcp skill ⇒ not native


@pytest.mark.asyncio
async def test_mcp_member_skill_without_c2_child_safe_fails_closed():
    repo = _FakeSkillRepo([_skill("unsafe-mcp")])  # no policy.c2_child_safe
    m = TeamMember(role="r", description="d", system_prompt="p", skills=("unsafe-mcp",))
    with pytest.raises(TeamCapabilityError):
        await resolve_team_member_map(team=_team(m), skill_repository=repo)


@pytest.mark.asyncio
async def test_native_member_skill_resolves_and_is_tracked_pr5():
    # [S4 §13 PR-5] native member skills RESOLVE (no longer rejected at resolve);
    # their slug is collected into native_skill_slugs and their generated tool
    # name is included in member_skill_tools. The per-unit capability gate
    # (shell+write) is enforced later in the dispatch path (5.2), NOT here.
    repo = _FakeSkillRepo([_skill("nat", runtime=SkillRuntimeType.NATIVE)])
    m = TeamMember(role="r", description="d", system_prompt="p", skills=("nat",),
                   shell_mode=True)
    out = await resolve_team_member_map(team=_team(m), skill_repository=repo)
    cap = out["r"]
    assert cap.native_skill_slugs == ("nat",)
    # _skill("nat") manifest tools default to [{"name": "go"}] →
    # generated name is skill_{slug}_{tool} = skill_nat_go.
    assert "skill_nat_go" in cap.member_skill_tools  # native tool name included


@pytest.mark.asyncio
async def test_unknown_or_disabled_skill_fails_closed():
    repo = _FakeSkillRepo([_skill("dis", enabled=False, policy={"c2_child_safe": True})])
    m = TeamMember(role="r", description="d", system_prompt="p", skills=("dis",))
    with pytest.raises(TeamCapabilityError):
        await resolve_team_member_map(team=_team(m), skill_repository=repo)
    m2 = TeamMember(role="r", description="d", system_prompt="p", skills=("ghost",))
    with pytest.raises(TeamCapabilityError):
        await resolve_team_member_map(team=_team(m2), skill_repository=repo)


@pytest.mark.asyncio
async def test_prompt_only_member_no_skills():
    repo = _FakeSkillRepo([])
    m = TeamMember(role="r", description="d", system_prompt="p")  # skills=()
    out = await resolve_team_member_map(team=_team(m), skill_repository=repo)
    assert out["r"].member_skill_tools == frozenset()
    assert out["r"].member_skill_slugs == ()
