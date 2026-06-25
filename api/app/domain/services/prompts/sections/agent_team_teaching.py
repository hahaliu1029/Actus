"""agent_team_teaching section — flag+team gated planner/updater teaching (S4).

Injects the agent-team member roster + a "set ``role`` on each work unit"
instruction into the planner + updater prompts ONLY when ALL THREE hold:

  1. ``ACTUS_C2_COORDINATOR_ENABLED`` is truthy (the master coordinator flag,
     load-bearing for the executor ``assert_coordinator_enabled()`` hard-gate),
  2. ``ACTUS_C2_AGENT_TEAMS_ENABLED`` is truthy (the S4 team flag), AND
  3. ``ctx.team_members`` is non-empty (a team was resolved for this session;
     ``planner_node`` loads this best-effort + STRUCTURAL-ONLY — see
     ``section.RenderContext.team_members``).

Any of the three missing ⇒ ``SectionOutput(text=None)`` so the planner prompt
is byte-identical to pre-S4 (INV-0). The ``role`` instruction is explicitly
OPTIONAL: an un-tagged work unit runs exactly as it does today — tagging only
selects a team member's persona + tool surface in ``_run_parallel_backend``.

Mirrors the structure of ``parallel_work_units_teaching.py``: a sync
``_render(ctx)`` gated on the flags, returning a ``SectionOutput``, plus a
module-level ``Section`` singleton registered in the planner/updater bundles.
Unlike that section the per-member roster is built from ``ctx.team_members`` at
render time, so the teaching prose is rendered around the roster rather than
pulled from a frozen constant.
"""
from __future__ import annotations

from app.domain.services.agent_teams_flag import is_agent_teams_enabled
from app.domain.services.coordinator_feature_flag import is_coordinator_enabled
from app.domain.services.prompts.section import (
    RenderContext,
    Section,
    SectionOutput,
)


# Bilingual teaching framing. The roster lines are interpolated between the
# header and the ``role`` instruction at render time. Schema key (``role``) is
# kept in English in both languages so the LLM does not translate it
# (schema-drift guard, same rationale as parallel_work_units_teaching).
_TEAM_TEACHING_HEADER_EN = "## Agent Team Members (advanced — optional per work unit)"
_TEAM_TEACHING_HEADER_ZH = "## Agent 团队成员 (Agent Team，高级用法，按 work_unit 选择)"

_TEAM_TEACHING_INTRO_EN = (
    "A named agent team is active for this session. Each member below is a "
    "specialist persona with its own system prompt + skill tool surface."
)
_TEAM_TEACHING_INTRO_ZH = (
    "本次会话启用了一个具名 agent 团队。下列每个成员都是带有专属 system "
    "prompt 与技能工具集的专家角色。"
)

_TEAM_TEACHING_ROLE_INSTRUCTION_EN = (
    "To dispatch a parallel work_unit to a specific member, set "
    '`"role": "<member_role>"` on that work_unit (one of the roles listed '
    "above). This is OPTIONAL — a work_unit with no `role` runs as today with "
    "the default child persona/tools."
)
_TEAM_TEACHING_ROLE_INSTRUCTION_ZH = (
    "若要把某个并行 work_unit 派发给特定成员，在该 work_unit 上设置 "
    '`"role": "<member_role>"`（取上方列出的某个 role）。这是可选的 —— '
    "未设置 `role` 的 work_unit 与今天一致，使用默认的子 agent 角色/工具。"
)


def _render(ctx: RenderContext) -> SectionOutput:
    """Flag+team gated: emit the member roster + role instruction.

    Returns ``SectionOutput(text=None)`` unless the coordinator flag AND the
    agent-team flag AND ``ctx.team_members`` are all present, keeping every
    non-team / flag-off call site byte-identical (INV-0)."""
    if not is_coordinator_enabled():
        return SectionOutput(text=None)
    if not is_agent_teams_enabled():
        return SectionOutput(text=None)
    team_members = getattr(ctx, "team_members", None)
    if not team_members:
        return SectionOutput(text=None)

    is_en = ctx.lang == "en"
    header = _TEAM_TEACHING_HEADER_EN if is_en else _TEAM_TEACHING_HEADER_ZH
    intro = _TEAM_TEACHING_INTRO_EN if is_en else _TEAM_TEACHING_INTRO_ZH
    role_instruction = (
        _TEAM_TEACHING_ROLE_INSTRUCTION_EN
        if is_en
        else _TEAM_TEACHING_ROLE_INSTRUCTION_ZH
    )

    roster_lines = [
        f"- `{role}`: {description}" for role, description in team_members
    ]
    roster = "\n".join(roster_lines)

    text = f"{header}\n\n{intro}\n\n{roster}\n\n{role_instruction}"
    return SectionOutput(text=text, metadata={"agent_team_teaching": True})


class AgentTeamTeachingSection:
    """Thin OO wrapper exposing ``_render`` for direct-render tests.

    The render logic lives in the module-level ``_render`` so the wrapper and
    the ``Section`` singleton (``render=_render``) share a single
    implementation — there is no per-instance state."""

    def _render(self, ctx: RenderContext) -> SectionOutput:
        return _render(ctx)


agent_team_teaching_section = Section(
    id="agent_team_teaching",
    # priority=8 == CRITICAL_PRIORITY_MIN — same rationale as
    # parallel_work_units_teaching_section: PROTECTED from budget-driven
    # dropping so a flag-on team roster can't silently vanish under prompt
    # budget pressure. Inert (text=None) when off, so this protection only
    # ever applies to flag-on team sessions.
    priority=8,
    cacheable=False,  # runtime-flag + per-session-team gated output
    dynamic=True,
    render=_render,
)
