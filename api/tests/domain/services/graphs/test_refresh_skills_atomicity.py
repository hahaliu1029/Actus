"""B5 C5a: atomicity tests for _refresh_skill_context_for_step refactor.

Eight scenarios from the design doc, exercising the split
``_compute_refreshed_skills`` (pure) + ``_apply_refreshed_skills``
(atomic mutation) + ``_build_step_react_graph`` (3-phase caller).

Invariant under test: the four persistent fields
- ``self._last_skill_context``
- ``self._last_skill_ids``
- ``self._last_bound_tool_names``
- ``self._last_initialized_skill_ids``
must be updated **atomically**: either ALL four reflect the new state,
or ALL four reflect the pre-call snapshot. No partial updates.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.app_config import (
    A2AConfig,
    AgentConfig,
    MCPConfig,
)
from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType
from app.domain.services.agent_task_runner import AgentTaskRunner
from app.domain.services.graphs.step_metadata import (
    RefreshedSkillsResult,
    StepMetadata,
)


pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


# ---- Test fixtures (minimal runner + fakes) ---------------------------- #


def _build_skill(skill_id: str, *, name: str = "") -> Skill:
    return Skill(
        id=skill_id,
        slug=skill_id,
        name=name or skill_id,
        description=f"{skill_id} description",
        source_type=SkillSourceType.GITHUB,
        source_ref=f"github:test/{skill_id}",
        runtime_type=SkillRuntimeType.NATIVE,
        manifest={"runtime_type": "native", "tools": []},
        enabled=True,
    )


class _FakeSkillTool:
    """Records calls to initialize() so tests can assert call patterns."""

    def __init__(self) -> None:
        self.initialize_calls: list[list[Skill]] = []
        self.raise_on_initialize: Exception | None = None
        self.side_effect_before_raise: Any = None  # callable to simulate partial mutation

    async def initialize(self, skills: list[Skill]) -> None:
        if self.side_effect_before_raise is not None:
            self.side_effect_before_raise()
        if self.raise_on_initialize is not None:
            raise self.raise_on_initialize
        self.initialize_calls.append(list(skills))

    async def cleanup(self) -> None:
        return None

    def get_tools(self) -> list[dict[str, Any]]:
        return []


class _FakeSandbox:
    async def ensure_sandbox(self) -> None:
        return None


class _FakeMCPTool:
    async def initialize(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def cleanup(self) -> None:
        return None

    def get_tools(self) -> list[Any]:
        return []


class _FakeA2ATool:
    def __init__(self) -> None:
        self.manager = None

    async def initialize(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def cleanup(self) -> None:
        return None


class _FakeSkillBundleSync:
    sandbox_skill_root = "/sandbox/skills"

    def get_file_listing_all(self) -> dict[str, Any]:
        return {}


def _make_runner(monkeypatch: pytest.MonkeyPatch) -> AgentTaskRunner:
    """Build a minimally-initialized AgentTaskRunner for atomicity tests.

    We bypass the full lifecycle (which spins up PlannerReActFlow and
    registers events) by monkey-patching PlannerReActFlow to a dummy
    before the runner is constructed.
    """

    class _DummyFlow:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self._overflow_config = SimpleNamespace(tool_result_max_chars=8000)
            self._assembler = None
            self._telemetry = None
            self._memory_config = SimpleNamespace(half_life_days=30, mmr_lambda=0.5)

        def set_skill_context(self, skill_context: str) -> None:
            pass

        async def close(self) -> None:
            pass

    monkeypatch.setattr(
        "app.domain.services.agent_task_runner.PlannerReActFlow",
        _DummyFlow,
    )

    runner = AgentTaskRunner(
        uow_factory=lambda: MagicMock(),
        llm=object(),
        agent_config=AgentConfig(
            max_iterations=100, max_retries=3, max_search_results=10
        ),
        mcp_config=MCPConfig(mcpServers={}),
        a2a_config=A2AConfig(a2a_servers=[]),
        session_id="atom-test",
        user_id="user-atom",
        file_storage=object(),
        browser=object(),
        search_engine=object(),
        sandbox=_FakeSandbox(),
    )
    runner._mcp_tool = _FakeMCPTool()
    runner._a2a_tool = _FakeA2ATool()
    runner._skill_tool = _FakeSkillTool()
    runner._skill_bundle_sync = _FakeSkillBundleSync()
    runner._session_skill_pool = []
    runner._current_message_text = ""
    runner._embedding_available = False
    runner._embedding_index = None

    # Stub _skill_selector so _compute_refreshed_skills has a fallback path
    runner._skill_selector = SimpleNamespace(select=lambda pool, query: [])
    # Stub _build_runtime_system_context so we don't need real skill state
    monkeypatch.setattr(
        runner,
        "_build_runtime_system_context",
        lambda skills, scores=None: (
            "## Active Skills\n" + "\n".join(f"- {s.id}" for s in skills)
            if skills
            else ""
        ),
    )
    # Stub tool building so _build_step_react_graph doesn't need real infrastructure
    monkeypatch.setattr(
        runner,
        "_build_lc_tools_for_step",
        lambda: [
            SimpleNamespace(name="shell_execute"),
            SimpleNamespace(name="file_read"),
        ],
    )
    monkeypatch.setattr(
        runner,
        "_build_minimal_lc_tools_for_step",
        lambda: [SimpleNamespace(name="shell_execute")],
    )
    # Stub build_react_graph so we don't need real LLM/tools plumbing
    monkeypatch.setattr(
        "app.domain.services.graphs.react_graph.build_react_graph",
        lambda **kwargs: MagicMock(name="compiled_graph"),
    )

    return runner


def _seed_pre_call_state(runner: AgentTaskRunner) -> dict[str, Any]:
    """Set up a 'previous step' state and return the snapshot for later assertions."""
    runner._last_skill_context = "## Previous context"
    runner._last_skill_ids = ("prev_skill_1",)
    runner._last_bound_tool_names = frozenset({"shell_execute", "skill_prev"})
    runner._last_initialized_skill_ids = ("prev_skill_1",)
    return {
        "skill_context": runner._last_skill_context,
        "skill_ids": runner._last_skill_ids,
        "bound_tool_names": runner._last_bound_tool_names,
        "initialized_skill_ids": runner._last_initialized_skill_ids,
    }


def _assert_state_matches_snapshot(
    runner: AgentTaskRunner, snapshot: dict[str, Any]
) -> None:
    assert runner._last_skill_context == snapshot["skill_context"]
    assert runner._last_skill_ids == snapshot["skill_ids"]
    assert runner._last_bound_tool_names == snapshot["bound_tool_names"]
    assert runner._last_initialized_skill_ids == snapshot["initialized_skill_ids"]


# ---- Scenario 1: Normal path --------------------------------------------- #


async def test_scenario_1_normal_path_updates_all_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_compute returns new result → _apply commits → all 3 _last_* updated."""
    runner = _make_runner(monkeypatch)
    _seed_pre_call_state(runner)

    new_skills = (_build_skill("s1"), _build_skill("s2"))
    refreshed = RefreshedSkillsResult(
        skills=new_skills,
        context="## New context\n- s1, s2",
        skill_ids=("s1", "s2"),
        scores=None,
    )

    async def fake_compute(step_desc: str) -> RefreshedSkillsResult:
        return refreshed

    monkeypatch.setattr(runner, "_compute_refreshed_skills", fake_compute)

    step_react, metadata = await runner._build_step_react_graph("new step")

    assert runner._last_skill_context == "## New context\n- s1, s2"
    assert runner._last_skill_ids == ("s1", "s2")
    assert runner._last_bound_tool_names == frozenset({"shell_execute", "file_read"})
    assert runner._last_initialized_skill_ids == ("s1", "s2")
    assert metadata.skill_ids == ("s1", "s2")
    assert metadata.bound_tool_names == frozenset({"shell_execute", "file_read"})
    assert metadata.skill_context == "## New context\n- s1, s2"


# ---- Scenario 2: Low-score sticky ---------------------------------------- #


async def test_scenario_2_sticky_preserves_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_compute returns None (sticky) → _apply is skipped → _last_* unchanged."""
    runner = _make_runner(monkeypatch)
    snapshot = _seed_pre_call_state(runner)

    async def fake_compute(step_desc: str) -> None:
        return None  # sticky

    monkeypatch.setattr(runner, "_compute_refreshed_skills", fake_compute)

    step_react, metadata = await runner._build_step_react_graph("low-confidence step")

    # _last_skill_context / _skill_ids / _initialized_skill_ids unchanged
    assert runner._last_skill_context == snapshot["skill_context"]
    assert runner._last_skill_ids == snapshot["skill_ids"]
    assert runner._last_initialized_skill_ids == snapshot["initialized_skill_ids"]
    # But _last_bound_tool_names IS allowed to advance (Phase 2 succeeded)
    assert runner._last_bound_tool_names == frozenset({"shell_execute", "file_read"})
    # StepMetadata reflects the sticky skill state + fresh bound tools
    assert metadata.skill_context == snapshot["skill_context"]
    assert metadata.skill_ids == snapshot["skill_ids"]


# ---- Scenario 3: _apply raises via _initialize_skill_tool_if_needed ------- #


async def test_scenario_3_initialize_raises_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_initialize_skill_tool_if_needed raises → _apply_refreshed_skills rollback.

    After the call:
    - All 4 _last_* fields equal their pre-call snapshot
    - No exception propagates to the caller
    - StepMetadata.bound_tool_names == pre-call (because refresh_failed=True)
    """
    runner = _make_runner(monkeypatch)
    snapshot = _seed_pre_call_state(runner)

    refreshed = RefreshedSkillsResult(
        skills=(_build_skill("s_new"),),
        context="## New",
        skill_ids=("s_new",),
        scores=None,
    )

    async def fake_compute(step_desc: str) -> RefreshedSkillsResult:
        return refreshed

    # Patch _initialize_skill_tool_if_needed to raise
    async def fake_initialize(skills: list[Skill]) -> None:
        raise RuntimeError("simulated initialize failure")

    monkeypatch.setattr(runner, "_compute_refreshed_skills", fake_compute)
    monkeypatch.setattr(runner, "_initialize_skill_tool_if_needed", fake_initialize)

    step_react, metadata = await runner._build_step_react_graph("new step")

    # All 4 fields rolled back to pre-call state
    _assert_state_matches_snapshot(runner, snapshot)
    # Metadata reflects the rolled-back state
    assert metadata.skill_context == snapshot["skill_context"]
    assert metadata.skill_ids == snapshot["skill_ids"]
    assert metadata.bound_tool_names == snapshot["bound_tool_names"]


# ---- Scenario 4: _compute raises ----------------------------------------- #


async def test_scenario_4_compute_raises_preserves_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_compute_refreshed_skills raises → _apply never called → state unchanged."""
    runner = _make_runner(monkeypatch)
    snapshot = _seed_pre_call_state(runner)

    async def fake_compute(step_desc: str) -> RefreshedSkillsResult:
        raise RuntimeError("simulated compute failure")

    monkeypatch.setattr(runner, "_compute_refreshed_skills", fake_compute)

    step_react, metadata = await runner._build_step_react_graph("new step")

    _assert_state_matches_snapshot(runner, snapshot)


# ---- Scenario 5: First call (empty state) -------------------------------- #


async def test_scenario_5_first_call_no_previous_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First call with no previous state → normal-path write."""
    runner = _make_runner(monkeypatch)
    # Do NOT seed pre-call state — _last_* stay at __init__ defaults

    refreshed = RefreshedSkillsResult(
        skills=(_build_skill("first"),),
        context="## First",
        skill_ids=("first",),
        scores=None,
    )

    async def fake_compute(step_desc: str) -> RefreshedSkillsResult:
        return refreshed

    monkeypatch.setattr(runner, "_compute_refreshed_skills", fake_compute)

    step_react, metadata = await runner._build_step_react_graph("first step")

    assert runner._last_skill_context == "## First"
    assert runner._last_skill_ids == ("first",)
    assert runner._last_bound_tool_names == frozenset(
        {"shell_execute", "file_read"}
    )
    assert runner._last_initialized_skill_ids == ("first",)


# ---- Scenario 6: Same skill_ids, initialize is no-op --------------------- #


async def test_scenario_6_same_skill_ids_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_compute returns the SAME skill_ids as pre-call → initialize skipped
    by the early-return guard, but _last_skill_context and _last_skill_ids
    still get updated by the atomic commit.
    """
    runner = _make_runner(monkeypatch)
    _seed_pre_call_state(runner)
    # Pre-call: _last_initialized_skill_ids == ("prev_skill_1",)

    # Compute returns the same skill id, with an updated context (e.g. new scores)
    refreshed = RefreshedSkillsResult(
        skills=(_build_skill("prev_skill_1"),),
        context="## Updated context same skill",
        skill_ids=("prev_skill_1",),
        scores=None,
    )

    async def fake_compute(step_desc: str) -> RefreshedSkillsResult:
        return refreshed

    monkeypatch.setattr(runner, "_compute_refreshed_skills", fake_compute)

    # Track whether initialize was called
    original_initialize = runner._skill_tool.initialize
    init_call_count = 0

    async def tracking_initialize(skills: list[Skill]) -> None:
        nonlocal init_call_count
        init_call_count += 1
        await original_initialize(skills)

    runner._skill_tool.initialize = tracking_initialize  # type: ignore[method-assign]

    step_react, metadata = await runner._build_step_react_graph("same-skill step")

    # _initialize_skill_tool_if_needed early-returned due to skill_ids match:
    # it did not call self._skill_tool.initialize
    assert init_call_count == 0
    # _last_initialized_skill_ids unchanged (early return)
    assert runner._last_initialized_skill_ids == ("prev_skill_1",)
    # BUT _last_skill_context and _last_skill_ids were atomically committed
    assert runner._last_skill_context == "## Updated context same skill"
    assert runner._last_skill_ids == ("prev_skill_1",)
    # Phase 2 succeeded → _last_bound_tool_names advanced
    assert runner._last_bound_tool_names == frozenset(
        {"shell_execute", "file_read"}
    )


# ---- Scenario 7: Partial mutation on initialize raise -------------------- #


async def test_scenario_7_initialize_partial_mutation_then_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_skill_tool.initialize() writes some internal state, then raises.

    This exercises the documented limitation: SkillTool's internal state
    may be partially updated, but the runner's 4 bookkeeping fields are
    still rolled back atomically. StepMetadata.bound_tool_names equals
    the pre-call value (lower bound); step_react may technically bind
    a superset, but the contract is "LLM only uses what the prompt
    advertises", which is governed by StepMetadata.
    """
    runner = _make_runner(monkeypatch)
    snapshot = _seed_pre_call_state(runner)

    refreshed = RefreshedSkillsResult(
        skills=(_build_skill("s_partial"),),
        context="## partial",
        skill_ids=("s_partial",),
        scores=None,
    )

    async def fake_compute(step_desc: str) -> RefreshedSkillsResult:
        return refreshed

    # Simulate partial mutation: imagine initialize() wrote to some
    # internal _activated_tool_names-style field, THEN raised
    partial_mutation_log: list[str] = []

    async def fake_initialize(skills: list[Skill]) -> None:
        partial_mutation_log.append("partial write before raise")
        raise RuntimeError("partial mutation failure")

    monkeypatch.setattr(runner, "_compute_refreshed_skills", fake_compute)
    monkeypatch.setattr(runner, "_initialize_skill_tool_if_needed", fake_initialize)

    step_react, metadata = await runner._build_step_react_graph("partial step")

    # Partial mutation DID happen
    assert partial_mutation_log == ["partial write before raise"]
    # But all 4 runner bookkeeping fields are rolled back
    _assert_state_matches_snapshot(runner, snapshot)
    # StepMetadata is the LOWER BOUND — matches rolled-back state
    assert metadata.bound_tool_names == snapshot["bound_tool_names"]
    assert metadata.skill_context == snapshot["skill_context"]
    assert metadata.skill_ids == snapshot["skill_ids"]


# ---- Scenario 8: lc_tools construction raises on partial SkillTool ------- #


async def test_scenario_8_lc_tools_construction_failure_degrades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 2 lc_tools construction raises → degrades to minimal path.

    Should NOT propagate the exception. StepMetadata.bound_tool_names
    equals the pre-call value. Actual lc_tools equals the minimal set.
    """
    runner = _make_runner(monkeypatch)
    snapshot = _seed_pre_call_state(runner)

    refreshed = RefreshedSkillsResult(
        skills=(_build_skill("s_lc_fail"),),
        context="## lc fail",
        skill_ids=("s_lc_fail",),
        scores=None,
    )

    async def fake_compute(step_desc: str) -> RefreshedSkillsResult:
        return refreshed

    monkeypatch.setattr(runner, "_compute_refreshed_skills", fake_compute)

    # Phase 1 succeeds (apply commits). Phase 2 raises.
    def raising_build_lc_tools() -> list[Any]:
        raise RuntimeError("dynamic skill tool construction failed")

    monkeypatch.setattr(runner, "_build_lc_tools_for_step", raising_build_lc_tools)

    step_react, metadata = await runner._build_step_react_graph("lc-fail step")

    # Phase 2 degradation: refresh_failed=True branch kicked in
    # → stable_bound_tool_names == pre-call _last_bound_tool_names
    assert metadata.bound_tool_names == snapshot["bound_tool_names"]
    # But Phase 1 already committed the new skill_context/skill_ids
    assert runner._last_skill_context == "## lc fail"
    assert runner._last_skill_ids == ("s_lc_fail",)
    # And _last_bound_tool_names stayed pinned to pre-call
    assert runner._last_bound_tool_names == snapshot["bound_tool_names"]
    # StepMetadata mirrors the atomic commit for skill fields
    assert metadata.skill_context == "## lc fail"
    assert metadata.skill_ids == ("s_lc_fail",)


# ---- Scenario 9: Split-brain carry-over ---------------------------------- #


async def test_scenario_9_split_brain_carries_over_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a Scenario-8-style split-brain (skill_context advanced, but
    bound_tool_names stayed pinned to pre-call), a SECOND follow-up call
    with normal-path success must correctly advance ALL four fields to
    the NEXT state — not resurrect the stale pinned bound_tool_names.

    Guards against the failure mode where Phase 3's "refresh_failed
    preserves pre-call bound_tool_names" logic inadvertently freezes
    the field across calls.
    """
    runner = _make_runner(monkeypatch)
    pre_snapshot = _seed_pre_call_state(runner)

    # First call: scenario-8 split brain
    refreshed_1 = RefreshedSkillsResult(
        skills=(_build_skill("s_split"),),
        context="## split ctx",
        skill_ids=("s_split",),
        scores=None,
    )

    async def compute_1(step_desc: str) -> RefreshedSkillsResult:
        return refreshed_1

    monkeypatch.setattr(runner, "_compute_refreshed_skills", compute_1)

    def raise_lc_tools() -> list[Any]:
        raise RuntimeError("phase 2 degrade")

    monkeypatch.setattr(runner, "_build_lc_tools_for_step", raise_lc_tools)

    _, meta1 = await runner._build_step_react_graph("first")
    # After call 1: skill fields advanced, bound_tool_names stayed pinned
    assert runner._last_skill_ids == ("s_split",)
    assert runner._last_skill_context == "## split ctx"
    assert runner._last_bound_tool_names == pre_snapshot["bound_tool_names"]
    assert meta1.bound_tool_names == pre_snapshot["bound_tool_names"]

    # Second call: normal path. Must advance ALL fields cleanly — no stale
    # bound_tool_names carryover from the split-brain state.
    refreshed_2 = RefreshedSkillsResult(
        skills=(_build_skill("s_next"),),
        context="## next ctx",
        skill_ids=("s_next",),
        scores=None,
    )

    async def compute_2(step_desc: str) -> RefreshedSkillsResult:
        return refreshed_2

    monkeypatch.setattr(runner, "_compute_refreshed_skills", compute_2)
    # Restore the working _build_lc_tools_for_step stub (from _make_runner)
    monkeypatch.setattr(
        runner,
        "_build_lc_tools_for_step",
        lambda: [
            SimpleNamespace(name="shell_execute"),
            SimpleNamespace(name="file_read"),
        ],
    )

    _, meta2 = await runner._build_step_react_graph("second")

    # All fields match the NEW state, not the pre-snapshot AND not the
    # split-brain state from call 1.
    assert runner._last_skill_ids == ("s_next",)
    assert runner._last_skill_context == "## next ctx"
    assert runner._last_bound_tool_names == frozenset(
        {"shell_execute", "file_read"}
    )
    assert meta2.skill_ids == ("s_next",)
    assert meta2.skill_context == "## next ctx"
    assert meta2.bound_tool_names == frozenset({"shell_execute", "file_read"})


# ---- Scenario 10: build_react_graph raises after Phase 1 commit --------- #


async def test_lc_tools_degradation_writes_to_runner_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-audit MEDIUM #6 regression: the lc_tools degradation path
    (Phase 2 fallback to _build_minimal_lc_tools_for_step) must record
    a ``record_lc_tools_degradation`` event via the runner's own
    ``_prompt_telemetry`` port. Pre-fix, the code read
    ``self._flow._telemetry`` which was never wired, so the degradation
    signal silently dropped."""
    runner = _make_runner(monkeypatch)
    _seed_pre_call_state(runner)

    # Inject a recording telemetry port on the runner
    class _Spy:
        def __init__(self) -> None:
            self.degradation_calls: list[str] = []

        def record_assembly(self, **kwargs: Any) -> None:
            pass

        def record_llm_invocation(self, **kwargs: Any) -> None:
            pass

        def record_lc_tools_degradation(self, *, reason: str) -> None:
            self.degradation_calls.append(reason)

    spy = _Spy()
    runner._prompt_telemetry = spy

    # Stub _compute_refreshed_skills to return a clean result so Phase 1
    # succeeds, then make Phase 2 fail to trigger the degradation path.
    refreshed = RefreshedSkillsResult(
        skills=(_build_skill("s_degrade"),),
        context="## degrade",
        skill_ids=("s_degrade",),
        scores=None,
    )

    async def fake_compute(step_desc: str) -> RefreshedSkillsResult:
        return refreshed

    monkeypatch.setattr(runner, "_compute_refreshed_skills", fake_compute)

    # Make the normal lc_tools path raise so the degraded path fires
    def raising_build_lc_tools() -> list[Any]:
        raise RuntimeError("simulated lc_tools build failure")

    monkeypatch.setattr(runner, "_build_lc_tools_for_step", raising_build_lc_tools)

    await runner._build_step_react_graph("degrade step")

    # Degradation telemetry was recorded on the runner's port
    assert spy.degradation_calls == ["lc_tools_build_failed"]


async def test_scenario_10_build_react_graph_exception_rolls_back_all_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-audit HIGH #1 regression: if Phase 1 of
    ``_build_step_react_graph`` successfully commits (skill refresh +
    lc_tools + fresh_bound_tool_names), but the final
    ``build_react_graph`` call raises, ALL FOUR ``_last_*`` bookkeeping
    fields must be rolled back to their pre-call snapshot. The exception
    must propagate (not be swallowed), but the runner must be left in
    a consistent state so the next step can start cleanly.

    This was the hole Codex identified: earlier code only guarded
    ``_last_bound_tool_names`` at the Phase 3 write site, but the Phase 1
    commits to ``_last_skill_context`` / ``_last_skill_ids`` /
    ``_last_initialized_skill_ids`` leaked through on exception.
    """
    runner = _make_runner(monkeypatch)
    snapshot = _seed_pre_call_state(runner)

    # Phase 1 succeeds with fresh values
    refreshed = RefreshedSkillsResult(
        skills=(_build_skill("s_phase3_fail"),),
        context="## phase3 fail ctx",
        skill_ids=("s_phase3_fail",),
        scores=None,
    )

    async def fake_compute(step_desc: str) -> RefreshedSkillsResult:
        return refreshed

    monkeypatch.setattr(runner, "_compute_refreshed_skills", fake_compute)

    # Phase 3 raises
    def raising_build_react_graph(**kwargs: Any) -> None:
        raise RuntimeError("simulated build_react_graph failure")

    monkeypatch.setattr(
        "app.domain.services.graphs.react_graph.build_react_graph",
        raising_build_react_graph,
    )

    # The whole call should raise (not swallow), and all 4 fields must
    # be rolled back to the pre-call snapshot.
    with pytest.raises(RuntimeError, match="simulated build_react_graph failure"):
        await runner._build_step_react_graph("trigger phase 3 failure")

    _assert_state_matches_snapshot(runner, snapshot)


async def test_scenario_10_partial_phase3_rollback_keeps_next_call_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After Scenario 10's rollback, the very next call to
    ``_build_step_react_graph`` must work cleanly against pre-call
    state — i.e. the rollback fully repaired the runner, not just the
    4 fields but also any implicit state affecting the cache path."""
    runner = _make_runner(monkeypatch)
    snapshot = _seed_pre_call_state(runner)

    # First call: Phase 3 raises
    refreshed_bad = RefreshedSkillsResult(
        skills=(_build_skill("s_bad"),),
        context="## bad",
        skill_ids=("s_bad",),
        scores=None,
    )

    async def compute_bad(step_desc: str) -> RefreshedSkillsResult:
        return refreshed_bad

    monkeypatch.setattr(runner, "_compute_refreshed_skills", compute_bad)

    call_count = {"n": 0}

    def flaky_build_react_graph(**kwargs: Any) -> Any:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("first call fails")
        return MagicMock(name="compiled_graph_ok")

    monkeypatch.setattr(
        "app.domain.services.graphs.react_graph.build_react_graph",
        flaky_build_react_graph,
    )

    # Call 1: expect the exception
    with pytest.raises(RuntimeError, match="first call fails"):
        await runner._build_step_react_graph("bad step")
    _assert_state_matches_snapshot(runner, snapshot)

    # Call 2: successful path — should see pre-call state on entry and
    # advance to new values (not the failed s_bad from call 1).
    refreshed_good = RefreshedSkillsResult(
        skills=(_build_skill("s_good"),),
        context="## good",
        skill_ids=("s_good",),
        scores=None,
    )

    async def compute_good(step_desc: str) -> RefreshedSkillsResult:
        return refreshed_good

    monkeypatch.setattr(runner, "_compute_refreshed_skills", compute_good)

    _, metadata = await runner._build_step_react_graph("good step")

    assert runner._last_skill_ids == ("s_good",)
    assert runner._last_skill_context == "## good"
    assert metadata.skill_ids == ("s_good",)
    # Crucially: the skill_ids advanced from the snapshot value, NOT from
    # the failed s_bad state (which leaked in the pre-fix version).
    assert runner._last_skill_ids != snapshot["skill_ids"]
    assert runner._last_skill_ids != ("s_bad",)


# ---- Bonus: legacy _refresh_skill_context_for_step still works ---------- #


async def test_legacy_refresh_wrapper_preserves_old_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_refresh_skill_context_for_step`` (the legacy name) must still
    return the current skill_context string so the updater-side consumer
    (``skill_context_refresher`` injection) keeps working."""
    runner = _make_runner(monkeypatch)
    _seed_pre_call_state(runner)

    refreshed = RefreshedSkillsResult(
        skills=(_build_skill("leg1"),),
        context="## legacy wrapper",
        skill_ids=("leg1",),
        scores=None,
    )

    async def fake_compute(step_desc: str) -> RefreshedSkillsResult:
        return refreshed

    monkeypatch.setattr(runner, "_compute_refreshed_skills", fake_compute)

    result = await runner._refresh_skill_context_for_step("legacy entry")
    assert result == "## legacy wrapper"
    assert runner._last_skill_context == "## legacy wrapper"
    assert runner._last_skill_ids == ("leg1",)
