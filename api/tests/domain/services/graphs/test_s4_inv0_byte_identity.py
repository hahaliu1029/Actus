"""[S4 §14/§17/R6-F5 + codex-R1-F1] INV-0 CI byte-identity guard.

Proves ALL FOUR S4 surfaces are byte-identical when the feature is OFF (no
flag / no team) or when a team is selected but a given work unit carries no
role. The four surfaces:

  (a) manifest bytes  — a no-role unit serializes exactly like a pre-S4
      WorkUnit, INCLUDING a duplicated ``allowed_tools`` payload surviving
      verbatim (the INV-0 trap — NO ``dict.fromkeys`` dedupe on the no-merge
      branch).
  (b) child prompt    — ``build_minimal_for_coordinator_child`` with no member
      params is identical to passing ``member_system_prompt=None``.
  (c) tool_filter     — ``_apply_member_skill_bind`` with an empty member tool
      set is the identity on the bind floor.
  (d) planner prompt  — the teaching section is the ONLY S4 addition to the
      planner prompt, so its ``_render`` returning ``text=None`` when off IS
      the planner-prompt-identity proof.

Plus an AST guard that BOTH ``input_for_graph`` dict literals in
``planner_react.py`` thread the ``team_slug`` key.
"""
import ast
import json
import pathlib

from app.application.services.child_agent_runner_factory import _apply_member_skill_bind
from app.domain.models.tool_filter_presets import COORDINATOR_STEP_PRESET
from app.domain.models.work_unit import WorkUnit, WorkUnitRequest
from app.domain.services.graphs.parallel_execution_subgraph import (
    _build_work_units_from_requests, _serialize_spawn_manifest,
)
from app.domain.services.prompts.assembler import PromptAssembler


class _EmptyCpc:
    member_skill_tools = frozenset()


def test_manifest_byte_identical_no_role_with_duplicate_allowed_tools():
    # (a) the INV-0 trap: an LLM-emitted duplicate must survive verbatim (no fromkeys).
    # codex R2 F2: use a deliberately OUT-OF-ORDER duplicate payload so the oracle
    # also catches a future ``sorted()`` regression (which a pre-sorted all-equal
    # list like ``["file_read", "file_read"]`` would not).
    req = WorkUnitRequest(objective="o", phase="exploration",
                          allowed_tools=["z_tool", "a_tool", "z_tool"])
    units = _build_work_units_from_requests([req], "h", 0, team_member_map=None)
    pre_s4 = WorkUnit(work_unit_id="h.a0.0", objective="o", phase="exploration",
                      allowed_tools=["z_tool", "a_tool", "z_tool"])
    assert _serialize_spawn_manifest(units[0]) == _serialize_spawn_manifest(pre_s4)
    data = json.loads(_serialize_spawn_manifest(units[0]))
    # Exact serialized order must survive: a ``sorted()`` regression would yield
    # ``["a_tool", "z_tool", "z_tool"]`` and a dedup would yield
    # ``["z_tool", "a_tool"]`` — both FAIL this assertion.
    assert data["allowed_tools"] == ["z_tool", "a_tool", "z_tool"]
    assert "member_skill_tools" not in data and "member_skill_slugs" not in data


def test_child_prompt_identity_no_member():
    # (b) child prompt unchanged when no member params supplied
    base = dict(objective="o", phase="exploration", allowed_paths=[], work_unit_id="h.a0.0")
    assert (PromptAssembler.build_minimal_for_coordinator_child(**base)
            == PromptAssembler.build_minimal_for_coordinator_child(**base, member_system_prompt=None))
    # True oracle: the None path must yield a NON-empty legacy body with NO member
    # block — default==None alone would pass even if the default body regressed.
    out = PromptAssembler.build_minimal_for_coordinator_child(**base)
    assert "## Member role instructions (advisory" not in out  # no advisory block when off
    assert out  # non-empty pre-S4 body present


def test_tool_filter_identity_empty_member_tools():
    # (c) bind floor unchanged when member_skill_tools empty
    base = frozenset({"file_read"})
    assert _apply_member_skill_bind(base, _EmptyCpc(), COORDINATOR_STEP_PRESET) == base


def test_planner_prompt_teaching_section_inert_when_off(monkeypatch):
    # (d) the teaching section is the ONLY S4 addition to the planner prompt;
    # text=None when flag-OFF/no-team ⇒ planner prompt byte-identical.
    from app.domain.services.graphs.token_estimator import TokenEstimator
    from app.domain.services.prompts import get_prompt_section_bundle
    from app.domain.services.prompts.budget import SystemPromptBudget
    from app.domain.services.prompts.section import PromptMode, RenderContext
    from app.domain.services.prompts.sections.agent_team_teaching import (
        AgentTeamTeachingSection,
    )
    monkeypatch.delenv("ACTUS_C2_AGENT_TEAMS_ENABLED", raising=False)

    # direct-render: section is inert (text=None) when off/no-team
    class _Ctx: team_members = None
    assert AgentTeamTeachingSection()._render(_Ctx()).text is None

    # true oracle: ASSEMBLE the full planner bundle off/no-team and prove the team
    # teaching text is absent ⇒ planner prompt byte-identical (does not rely on the
    # unverified premise "teaching is the ONLY S4 planner addition").
    assembler = PromptAssembler(
        budget=SystemPromptBudget(max_tokens=10_000),
        token_estimator=TokenEstimator(strategy="hybrid"),
        telemetry=None,
    )
    for lang in ("zh", "en"):
        out = assembler.assemble(
            get_prompt_section_bundle(lang).planner,
            RenderContext(lang=lang, team_members=None),
            PromptMode.FULL,
            fallback_used=False,
        )
        assert "map the code" not in out.text  # teaching text absent ⇒ planner byte-identical when off


def test_both_input_for_graph_sites_thread_team_slug():
    # AST guard (robust, not a string count): count ALL `input_for_graph = {...}`
    # dict-literal assignment sites and assert EVERY one threads "team_slug" — so a
    # future 3rd site that forgets the key fails (count-only would still pass at 2).
    tree = ast.parse(
        pathlib.Path("app/domain/services/flows/planner_react.py").read_text()
    )
    total = 0
    with_key = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "input_for_graph"
                        for t in node.targets)
                and isinstance(node.value, ast.Dict)):
            total += 1
            keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
            if "team_slug" in keys:
                with_key += 1
    assert total >= 2, f"expected >=2 input_for_graph dict literals, found {total}"
    assert with_key == total, (
        f"every input_for_graph dict literal must thread team_slug, "
        f"{with_key}/{total} do"
    )
