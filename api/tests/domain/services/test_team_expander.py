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
    assert cap.native_skill_slugs == ()  # PR-3: native always empty


@pytest.mark.asyncio
async def test_mcp_member_skill_without_c2_child_safe_fails_closed():
    repo = _FakeSkillRepo([_skill("unsafe-mcp")])  # no policy.c2_child_safe
    m = TeamMember(role="r", description="d", system_prompt="p", skills=("unsafe-mcp",))
    with pytest.raises(TeamCapabilityError):
        await resolve_team_member_map(team=_team(m), skill_repository=repo)


@pytest.mark.asyncio
async def test_native_member_skill_rejected_in_pr3():
    repo = _FakeSkillRepo([_skill("nat", runtime=SkillRuntimeType.NATIVE)])
    m = TeamMember(role="r", description="d", system_prompt="p", skills=("nat",))
    with pytest.raises(TeamCapabilityError):
        await resolve_team_member_map(team=_team(m), skill_repository=repo)


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
