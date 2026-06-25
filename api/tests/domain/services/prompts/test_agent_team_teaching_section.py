"""S4 PR-4: direct-render tests for the agent-team teaching section.

The section renders the member roster + a "set `role` on each work unit"
instruction ONLY when ``is_coordinator_enabled() AND is_agent_teams_enabled()
AND ctx.team_members``; otherwise it emits ``SectionOutput(text=None)`` so a
flag-OFF / no-team planner prompt stays byte-identical (INV-0).
"""
from __future__ import annotations

from app.domain.services.prompts.sections.agent_team_teaching import (
    AgentTeamTeachingSection,
)


def _ctx(team_members):
    # minimal RenderContext stub exposing .team_members + .lang
    class _C:
        pass

    c = _C()
    c.team_members = team_members
    c.lang = "en"
    return c


def test_no_team_members_renders_nothing(monkeypatch):
    monkeypatch.setenv("ACTUS_C2_COORDINATOR_ENABLED", "true")
    monkeypatch.setenv("ACTUS_C2_AGENT_TEAMS_ENABLED", "true")
    out = AgentTeamTeachingSection()._render(_ctx(None))
    assert out.text is None


def test_flag_off_renders_nothing(monkeypatch):
    monkeypatch.setenv("ACTUS_C2_COORDINATOR_ENABLED", "true")
    monkeypatch.delenv("ACTUS_C2_AGENT_TEAMS_ENABLED", raising=False)
    out = AgentTeamTeachingSection()._render(_ctx((("explorer", "map the code"),)))
    assert out.text is None


def test_coordinator_off_renders_nothing(monkeypatch):
    monkeypatch.delenv("ACTUS_C2_COORDINATOR_ENABLED", raising=False)
    monkeypatch.setenv("ACTUS_C2_AGENT_TEAMS_ENABLED", "true")
    out = AgentTeamTeachingSection()._render(_ctx((("explorer", "map the code"),)))
    assert out.text is None


def test_renders_member_roster_and_role_instruction(monkeypatch):
    monkeypatch.setenv("ACTUS_C2_COORDINATOR_ENABLED", "true")
    monkeypatch.setenv("ACTUS_C2_AGENT_TEAMS_ENABLED", "true")
    out = AgentTeamTeachingSection()._render(_ctx((("explorer", "map the code"),)))
    assert "explorer" in out.text
    assert "map the code" in out.text
    assert "role" in out.text.lower()
