"""C2-full S4 team expander (spec §10/§13). Domain-pure: resolves a TeamBundle
into a role → MemberCapability map, fail-closed on any capability violation.
Runs parent-side in _run_parallel_backend (no sandbox)."""
from __future__ import annotations

from dataclasses import dataclass

from app.domain.models.agent_team import TeamBundle
from app.domain.models.skill import SkillRuntimeType
from app.domain.repositories.skill_repository import SkillRepository
from app.domain.services.tools.skill import SkillTool


class TeamCapabilityError(Exception):
    """A team/member/skill failed resolve-time capability validation. Caught in
    _run_parallel_backend alongside CoordinatorPathContractError → graceful
    FAILED step (re-plannable), never an uncaught agent-run abort."""


@dataclass(frozen=True)
class MemberCapability:
    system_prompt: str
    member_skill_tools: frozenset[str]   # GENERATED skill_{slug}_{tool} names (bind floor)
    member_skill_slugs: tuple[str, ...]  # source slugs (child-side carve-out carrier)
    native_skill_slugs: tuple[str, ...]  # ⊆ member_skill_slugs; the native ones (PR-5 gate)
    shell_mode: bool


async def resolve_team_member_map(
    *, team: TeamBundle, skill_repository: SkillRepository,
) -> dict[str, MemberCapability]:
    out: dict[str, MemberCapability] = {}
    for member in team.members:
        slugs = tuple(dict.fromkeys(member.skills))  # dedupe a member's slug list
        resolved = []
        native_slugs: list[str] = []
        for slug in slugs:
            skill = await skill_repository.get_by_slug(slug)
            if skill is None or not skill.enabled:
                raise TeamCapabilityError(
                    f"team {team.slug!r} member {member.role!r}: skill {slug!r} "
                    f"is not installed/enabled"
                )
            # ---- member-skill capability policy (§13, fail-closed) ----------
            if skill.runtime_type == SkillRuntimeType.NATIVE:
                # [S4 §13 PR-5] native is allowed but capability-gated PER-UNIT
                # (post-coercion shell_mode + write) in the dispatch path, NOT
                # here. Resolve tracks the slug; the gate enforces shell+write.
                native_slugs.append(slug)
            else:  # mcp / a2a → require the unverified author attestation
                policy = (skill.manifest or {}).get("policy", {})
                if not (isinstance(policy, dict) and policy.get("c2_child_safe") is True):
                    raise TeamCapabilityError(
                        f"team {team.slug!r} member {member.role!r}: mcp/a2a member "
                        f"skill {slug!r} must declare manifest.policy.c2_child_safe == true"
                    )
            resolved.append(skill)
        tool_names = SkillTool.generate_tool_names(resolved)
        if len(tool_names) != len(set(tool_names)):  # R7-3 defensive (near-impossible)
            raise TeamCapabilityError(
                f"team {team.slug!r} member {member.role!r}: generated skill tool "
                f"name collision: {tool_names}"
            )
        out[member.role] = MemberCapability(
            system_prompt=member.system_prompt,
            member_skill_tools=frozenset(tool_names),
            member_skill_slugs=slugs,
            native_skill_slugs=tuple(native_slugs),  # ⊆ member_skill_slugs; the native ones (PR-5 gate)
            shell_mode=member.shell_mode,
        )
    return out
