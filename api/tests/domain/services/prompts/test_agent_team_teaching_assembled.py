"""S4 PR-4 [codex-R5]: assembled-bundle tests for the agent-team teaching.

A direct-render test alone misses the "registered but never assembled"
dead-feature failure mode. These tests assemble the actual planner registry
via ``get_prompt_section_bundle`` so a section that is registered in the
wrong place (or not reaching the assembled prompt) is caught.
"""
from __future__ import annotations

import pytest

from app.domain.services.graphs.token_estimator import TokenEstimator
from app.domain.services.prompts import get_prompt_section_bundle
from app.domain.services.prompts.assembler import PromptAssembler
from app.domain.services.prompts.budget import SystemPromptBudget
from app.domain.services.prompts.section import PromptMode, RenderContext


def _assembler() -> PromptAssembler:
    return PromptAssembler(
        budget=SystemPromptBudget(max_tokens=10_000),
        token_estimator=TokenEstimator(strategy="hybrid"),
        telemetry=None,
    )


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_team_teaching_in_assembled_planner_prompt(monkeypatch, lang):
    monkeypatch.setenv("ACTUS_C2_COORDINATOR_ENABLED", "true")
    monkeypatch.setenv("ACTUS_C2_AGENT_TEAMS_ENABLED", "true")
    ctx = RenderContext(lang=lang, team_members=(("explorer", "map the code"),))
    out = _assembler().assemble(
        get_prompt_section_bundle(lang).planner,
        ctx,
        PromptMode.FULL,
        fallback_used=False,
    )
    assert "explorer" in out.text and "map the code" in out.text


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_team_teaching_absent_when_off(monkeypatch, lang):
    monkeypatch.delenv("ACTUS_C2_AGENT_TEAMS_ENABLED", raising=False)
    ctx = RenderContext(lang=lang, team_members=None)
    out = _assembler().assemble(
        get_prompt_section_bundle(lang).planner,
        ctx,
        PromptMode.FULL,
        fallback_used=False,
    )
    assert "map the code" not in out.text  # section inert ⇒ INV-0 planner prompt


# ---- EPIC-FIX-1: updater registry parity ----------------------------- #
# The teaching section is registered in BOTH the planner AND updater
# registries (bundles/{en,zh}.py). updater_node is the re-plan loop that
# RE-EMITS role-bearing parallel_work_units, so the teaching must reach the
# *assembled updater prompt* too — otherwise role-tagging degrades on every
# replan round. These mirror the planner-side tests against ``.updater``.


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_team_teaching_in_assembled_updater_prompt(monkeypatch, lang):
    monkeypatch.setenv("ACTUS_C2_COORDINATOR_ENABLED", "true")
    monkeypatch.setenv("ACTUS_C2_AGENT_TEAMS_ENABLED", "true")
    ctx = RenderContext(lang=lang, team_members=(("explorer", "map the code"),))
    out = _assembler().assemble(
        get_prompt_section_bundle(lang).updater,
        ctx,
        PromptMode.FULL,
        fallback_used=False,
    )
    assert "explorer" in out.text and "map the code" in out.text


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_team_teaching_absent_when_off_updater(monkeypatch, lang):
    monkeypatch.delenv("ACTUS_C2_AGENT_TEAMS_ENABLED", raising=False)
    ctx = RenderContext(lang=lang, team_members=None)
    out = _assembler().assemble(
        get_prompt_section_bundle(lang).updater,
        ctx,
        PromptMode.FULL,
        fallback_used=False,
    )
    assert "map the code" not in out.text  # section inert ⇒ INV-0 updater prompt
