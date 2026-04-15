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
from app.domain.services.tools.tool_source_resolver import resolve_tool_source


def _fake_tool(name: str) -> SimpleNamespace:
    """Build a stub tool with _actus_source metadata populated.

    R1.16: ``agent_task_runner._build_step_react_graph`` now filters
    ``lc_tools`` via ``resolve_tool_source_from_tool(t).category == 'skill'``.
    Real production tools always carry this metadata (set by factory calls to
    ``annotate_and_register_tool_source``), so test stubs must mirror it.
    """
    return SimpleNamespace(
        name=name,
        metadata={"_actus_source": resolve_tool_source(name)},
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
    """Test fake that models SkillTool internal state for #27 regression tests.

    Tracks which skill IDs are currently "loaded" (mirroring SkillTool's
    self._tools / self._tool_bindings shape at a coarse granularity). Tests
    can inspect ``loaded_skill_ids`` to verify the outer rollback path
    actively reset internal state, not just the 4 bookkeeping fields.
    """

    def __init__(self) -> None:
        self.initialize_calls: list[list[Skill]] = []
        self.raise_on_initialize: Exception | None = None
        self._loaded_skill_ids: tuple[str, ...] = ()

    async def initialize(self, skills: list[Skill]) -> None:
        self.initialize_calls.append(list(skills))
        if self.raise_on_initialize is not None:
            raise self.raise_on_initialize
        # Success path: commit the new loaded state (atomic, mirrors
        # real SkillTool.initialize() contract from #27).
        self._loaded_skill_ids = tuple(s.id for s in skills)

    @property
    def loaded_skill_ids(self) -> tuple[str, ...]:
        """Inspector for tests — what's currently loaded inside SkillTool."""
        return self._loaded_skill_ids

    async def cleanup(self) -> None:
        self._loaded_skill_ids = ()

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
            _fake_tool("shell_execute"),
            _fake_tool("file_read"),
        ],
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
    - 3 skill-related _last_* fields equal their pre-call snapshot
    - _last_bound_tool_names IS allowed to advance (Phase 2 still succeeds)
    - No exception propagates to the caller
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

    # 3 skill-related fields rolled back to pre-call state
    assert runner._last_skill_context == snapshot["skill_context"]
    assert runner._last_skill_ids == snapshot["skill_ids"]
    assert runner._last_initialized_skill_ids == snapshot["initialized_skill_ids"]
    # _last_bound_tool_names advances (Phase 2 succeeded post-#27)
    assert runner._last_bound_tool_names == frozenset({"shell_execute", "file_read"})
    # Metadata reflects the rolled-back skill state + fresh bound tools
    assert metadata.skill_context == snapshot["skill_context"]
    assert metadata.skill_ids == snapshot["skill_ids"]
    assert metadata.bound_tool_names == frozenset({"shell_execute", "file_read"})


# ---- Scenario 4: _compute raises ----------------------------------------- #


async def test_scenario_4_compute_raises_preserves_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_compute_refreshed_skills raises → _apply never called → skill state unchanged.

    Post-#27: _last_bound_tool_names IS allowed to advance since Phase 2
    still runs successfully — only the 3 skill-related fields stay sticky.
    """
    runner = _make_runner(monkeypatch)
    snapshot = _seed_pre_call_state(runner)

    async def fake_compute(step_desc: str) -> RefreshedSkillsResult:
        raise RuntimeError("simulated compute failure")

    monkeypatch.setattr(runner, "_compute_refreshed_skills", fake_compute)

    step_react, metadata = await runner._build_step_react_graph("new step")

    # Skill-related fields unchanged (sticky)
    assert runner._last_skill_context == snapshot["skill_context"]
    assert runner._last_skill_ids == snapshot["skill_ids"]
    assert runner._last_initialized_skill_ids == snapshot["initialized_skill_ids"]
    # But _last_bound_tool_names advances (Phase 2 succeeded)
    assert runner._last_bound_tool_names == frozenset({"shell_execute", "file_read"})


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


# ---- Scenario 7: SkillTool.initialize raise preserves runner state ------- #


async def test_skill_tool_initialize_raise_preserves_runner_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 7 (rewritten for #27): with SkillTool.initialize() now atomic,
    the runner's 4 _last_* bookkeeping fields must remain at their pre-call
    snapshot when _initialize_skill_tool_if_needed raises.

    This no longer needs to simulate a partial-state SkillTool (that state is
    impossible post-#27). It exercises the simpler contract: runner rollback
    works because the field assignments in _apply_refreshed_skills all happen
    AFTER the await to _initialize_skill_tool_if_needed.
    """
    runner = _make_runner(monkeypatch)
    snapshot = _seed_pre_call_state(runner)

    # Inject: _FakeSkillTool.initialize raises via the existing flag
    runner._skill_tool.raise_on_initialize = RuntimeError(
        "simulated SkillTool.initialize failure"
    )

    # _compute_refreshed_skills returns a new selection, _apply_refreshed_skills
    # will try to initialize and fail
    async def return_new_refreshed(step_description: str) -> RefreshedSkillsResult:
        return RefreshedSkillsResult(
            skills=(_build_skill("s_new"),),
            context="## new context",
            skill_ids=("s_new",),
            scores=None,
        )

    monkeypatch.setattr(
        runner, "_compute_refreshed_skills", return_new_refreshed
    )

    # Phase 1 catch converts the raise into a logged warning and sticky fallback
    step_react, metadata = await runner._build_step_react_graph(
        step_description="do something"
    )

    # 3 skill-related fields preserved (sticky).
    # _last_bound_tool_names is allowed to advance because Phase 2 still ran
    # successfully with the sticky skill selection (same behavior as
    # scenarios 3 and 4).
    assert runner._last_skill_context == snapshot["skill_context"]
    assert runner._last_skill_ids == snapshot["skill_ids"]
    assert runner._last_initialized_skill_ids == snapshot["initialized_skill_ids"]
    # StepMetadata reflects sticky values for skill fields
    assert metadata.skill_context == snapshot["skill_context"]
    assert metadata.skill_ids == snapshot["skill_ids"]


async def test_runner_recovers_after_skill_tool_initialize_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery: after a failed initialize sticky-falls-back once, the next
    successful build_step_react_graph call must advance all 4 _last_* fields
    cleanly (no residue from the failed attempt).

    Replaces scenario 9 (split-brain recovery), which was specific to the
    now-deleted minimal-path state.
    """
    runner = _make_runner(monkeypatch)
    initial_snapshot = _seed_pre_call_state(runner)

    # First call: SkillTool.initialize raises → sticky fallback
    call_count = {"n": 0}

    async def flaky_initialize(skills: list[Skill]) -> None:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("transient failure")
        # Second call: succeed

    monkeypatch.setattr(
        runner._skill_tool, "initialize", flaky_initialize
    )

    refreshed_result = RefreshedSkillsResult(
        skills=(_build_skill("s_recovered"),),
        context="## recovered context",
        skill_ids=("s_recovered",),
        scores=None,
    )

    async def return_refreshed(step_description: str) -> RefreshedSkillsResult:
        return refreshed_result

    monkeypatch.setattr(
        runner, "_compute_refreshed_skills", return_refreshed
    )

    # First call: initialize fails, sticky fallback
    await runner._build_step_react_graph(step_description="step 1")
    # The 3 skill-related fields stay sticky (bound_tool_names may advance — see Task 9)
    assert runner._last_skill_context == initial_snapshot["skill_context"]
    assert runner._last_skill_ids == initial_snapshot["skill_ids"]
    assert runner._last_initialized_skill_ids == initial_snapshot["initialized_skill_ids"]

    # Second call: initialize succeeds, all 4 fields advance cleanly
    _, metadata = await runner._build_step_react_graph(step_description="step 2")
    assert runner._last_skill_ids == ("s_recovered",)
    assert runner._last_skill_context == "## recovered context"
    assert runner._last_initialized_skill_ids == ("s_recovered",)
    assert metadata.skill_ids == ("s_recovered",)


async def test_mcp_activation_survives_refresh_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HIGH 1 regression: a newly-activated MCP tool (activated during step N
    via get_mcp_tool) must surface in step N+1's StepMetadata.bound_tool_names
    even when step N+1's Phase 1 skill refresh fails.

    Pre-#27 code pinned bound_tool_names to stale self._last_bound_tool_names
    (snapshotted before step N started executing, i.e., before the MCP
    activation), which hid the newly-activated MCP tool for one step. #27
    removes that pin; this test locks in the new behavior.

    See design doc §已知行为变化 for the full timeline analysis.
    """
    runner = _make_runner(monkeypatch)

    from app.domain.services.tools.tool_source_resolver import ToolSource

    class _FakeLCTool:
        def __init__(self, name: str, *, source: str, category: str) -> None:
            self.name = name
            # R1.16: _build_step_react_graph filters by canonical category via
            # resolve_tool_source_from_tool → tool.metadata["_actus_source"].
            # Since these stub names are deliberately non-canonical (to isolate
            # the MCP activation behavior change under test), we bake the
            # ToolSource directly rather than going through the registry.
            self.metadata = {
                "_actus_source": ToolSource(
                    source=source, category=category, canonical_name=name,
                ),
            }

    # Override the default _build_lc_tools_for_step stub so that it actually
    # reflects _activated_mcp_tools (instead of the fixture's fixed native
    # tool list). This is the minimum faithful model needed to exercise the
    # behavior change.
    def dynamic_build_lc_tools_for_step() -> list[Any]:
        tools: list[Any] = [
            _FakeLCTool("native_shell", source="native", category="shell"),
            _FakeLCTool("native_file_read", source="native", category="file"),
        ]
        for mcp_name in sorted(runner._activated_mcp_tools):
            tools.append(_FakeLCTool(mcp_name, source="mcp", category="mcp"))
        return tools

    monkeypatch.setattr(
        runner, "_build_lc_tools_for_step", dynamic_build_lc_tools_for_step
    )

    # Seed: step N committed _last_bound_tool_names BEFORE "mcp_foo" was
    # activated. Then step N's LLM called get_mcp_tool("mcp_foo") during
    # execution, mutating _activated_mcp_tools. Now step N+1 is starting.
    runner._last_bound_tool_names = frozenset(
        {"native_shell", "native_file_read"}
    )
    runner._last_skill_ids = ("skill_a",)
    runner._last_skill_context = "## skill_a context"
    runner._last_initialized_skill_ids = ("skill_a",)
    runner._activated_mcp_tools.add("mcp_foo")

    # Step N+1: Phase 1 raises (e.g., embedding query transient timeout)
    async def fail_refresh(*args: Any, **kwargs: Any) -> RefreshedSkillsResult:
        raise RuntimeError("embedding query timeout")

    monkeypatch.setattr(runner, "_compute_refreshed_skills", fail_refresh)

    _, metadata = await runner._build_step_react_graph(
        step_description="do something"
    )

    # StepMetadata advertises the activated MCP tool (refactored behavior)
    assert "mcp_foo" in metadata.bound_tool_names
    assert "native_shell" in metadata.bound_tool_names
    # Sticky skills context preserved on refresh failure
    assert metadata.skill_context == "## skill_a context"
    assert metadata.skill_ids == ("skill_a",)


# ---- Scenario 10: build_react_graph raises after Phase 1 commit --------- #


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


# ---- #27 hotfix: SkillTool internal state rollback ---------------------- #


async def test_phase3_failure_rollback_restores_skill_tool_internal_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex review HIGH finding (#27 hotfix).

    Reproduces the leak: when ``_build_step_react_graph``'s Phase 3 fails
    after Phase 1 successfully refreshed skills (mutating SkillTool internal
    state), the outer except must actively reset SkillTool back to the
    snapshot state AND clear ``_lc_tools_cache``. Without this, sticky
    follow-up calls would read the stale SkillTool (cache miss path) and
    leak the failed step's tools into ``metadata.bound_tool_names`` while
    ``metadata.skill_ids`` stays on the rolled-back old IDs.

    Pre-fix: ``_FakeSkillTool.loaded_skill_ids`` would still be the new
    skills after the failed ``_build_step_react_graph`` call.
    After fix: ``loaded_skill_ids`` is back to the snapshot state, AND
    ``_lc_tools_cache`` is empty, AND ``_last_initialized_skills`` is
    rolled back to the snapshot list.
    """
    runner = _make_runner(monkeypatch)

    # Seed: step 1 already ran successfully with skill_1 — SkillTool is
    # initialized, _last_initialized_skill_ids/_last_initialized_skills
    # reflect it, the 4 bookkeeping fields are committed.
    skill_1 = _build_skill("skill_1", name="Skill One")
    await runner._initialize_skill_tool_if_needed([skill_1])
    runner._last_skill_context = "## skill_1 context"
    runner._last_skill_ids = ("skill_1",)
    runner._last_bound_tool_names = frozenset({"shell_execute", "file_read"})
    # Prime _lc_tools_cache to simulate state from the successful step 1
    cache_key_step1 = (
        ("skill_1",),
        frozenset(),
    )
    runner._lc_tools_cache[cache_key_step1] = [
        _fake_tool("shell_execute"),
        _fake_tool("file_read"),
    ]

    # Sanity pre-check: SkillTool internally loaded skill_1
    assert runner._skill_tool.loaded_skill_ids == ("skill_1",)
    assert runner._last_initialized_skills == [skill_1]

    # Step 2: Phase 1 refreshes to skill_2 (successful SkillTool.initialize
    # mutation). Phase 3 (build_react_graph) then raises.
    skill_2 = _build_skill("skill_2", name="Skill Two")
    refreshed = RefreshedSkillsResult(
        skills=(skill_2,),
        context="## skill_2 context",
        skill_ids=("skill_2",),
        scores=None,
    )

    async def fake_compute(step_desc: str) -> RefreshedSkillsResult:
        return refreshed

    monkeypatch.setattr(runner, "_compute_refreshed_skills", fake_compute)

    # Phase 3 raises — build_react_graph failure triggers the outer rollback
    def raising_build_react_graph(**kwargs: Any) -> None:
        raise RuntimeError("simulated phase 3 failure")

    monkeypatch.setattr(
        "app.domain.services.graphs.react_graph.build_react_graph",
        raising_build_react_graph,
    )

    # The call should raise
    with pytest.raises(RuntimeError, match="simulated phase 3 failure"):
        await runner._build_step_react_graph("step 2 with phase 3 failure")

    # ---- Pre-fix would FAIL these assertions --------------------- #

    # #27 hotfix: SkillTool internal state was reset to skill_1
    # (not leaked as skill_2).
    assert runner._skill_tool.loaded_skill_ids == ("skill_1",), (
        f"SkillTool internal state leaked. Expected ('skill_1',) after "
        f"rollback, got {runner._skill_tool.loaded_skill_ids}. "
        f"initialize_calls={runner._skill_tool.initialize_calls}"
    )

    # #27 hotfix: _last_initialized_skills also rolled back
    assert runner._last_initialized_skills == [skill_1]

    # #27 hotfix: _lc_tools_cache cleared — prior step 1 cache entry is gone
    assert runner._lc_tools_cache == {}, (
        f"_lc_tools_cache not cleared; still has: {runner._lc_tools_cache}"
    )

    # The 4 bookkeeping fields rolled back (existing Scenario 10 contract)
    assert runner._last_skill_context == "## skill_1 context"
    assert runner._last_skill_ids == ("skill_1",)
    assert runner._last_bound_tool_names == frozenset(
        {"shell_execute", "file_read"}
    )
    assert runner._last_initialized_skill_ids == ("skill_1",)

    # Post-rollback: _skill_tool.initialize was called at least TWICE —
    # once by Phase 1 (→ skill_2), once by the rollback path (→ skill_1).
    assert len(runner._skill_tool.initialize_calls) >= 2
    # Last call should be the rollback restoration to skill_1
    last_call = runner._skill_tool.initialize_calls[-1]
    assert [s.id for s in last_call] == ["skill_1"]


async def test_phase2_failure_rollback_restores_skill_tool_internal_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex review HIGH finding — Phase 2 (lc_tools build) variant.

    Same scenario as the Phase 3 test but with Phase 2 (``_build_lc_tools_for_step``)
    as the failure point. Post-#27, ``_build_lc_tools_for_step`` errors also
    propagate to the outer rollback and should trigger SkillTool state reset.
    """
    runner = _make_runner(monkeypatch)

    skill_1 = _build_skill("skill_1", name="Skill One")
    await runner._initialize_skill_tool_if_needed([skill_1])
    runner._last_skill_context = "## skill_1 context"
    runner._last_skill_ids = ("skill_1",)
    runner._last_bound_tool_names = frozenset({"shell_execute", "file_read"})

    assert runner._skill_tool.loaded_skill_ids == ("skill_1",)

    skill_2 = _build_skill("skill_2", name="Skill Two")
    refreshed = RefreshedSkillsResult(
        skills=(skill_2,),
        context="## skill_2 context",
        skill_ids=("skill_2",),
        scores=None,
    )

    async def fake_compute(step_desc: str) -> RefreshedSkillsResult:
        return refreshed

    monkeypatch.setattr(runner, "_compute_refreshed_skills", fake_compute)

    # Phase 2 raises — lc_tools build failure
    def raising_build_lc_tools_for_step() -> list[Any]:
        raise RuntimeError("simulated phase 2 lc_tools failure")

    monkeypatch.setattr(
        runner, "_build_lc_tools_for_step", raising_build_lc_tools_for_step
    )

    with pytest.raises(RuntimeError, match="simulated phase 2 lc_tools failure"):
        await runner._build_step_react_graph("step 2 with phase 2 failure")

    # SkillTool reset to skill_1
    assert runner._skill_tool.loaded_skill_ids == ("skill_1",)
    assert runner._last_initialized_skills == [skill_1]
    assert runner._lc_tools_cache == {}
    assert runner._last_initialized_skill_ids == ("skill_1",)


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


# ---- Scenario 13: Session bootstrap via _apply_preselected_skills --------- #


async def test_scenario_13_session_bootstrap_advances_three_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct call to _apply_preselected_skills mirrors what session bootstrap does.

    Verifies that after the call, all three skill-related _last_* fields advance
    in sync. _last_bound_tool_names is intentionally untouched.
    """
    runner = _make_runner(monkeypatch)
    initial_skills = [_build_skill("init_a"), _build_skill("init_b")]

    await runner._apply_preselected_skills(initial_skills)

    assert runner._last_skill_ids == ("init_a", "init_b")
    assert runner._last_initialized_skill_ids == ("init_a", "init_b")
    assert runner._last_skill_context == (
        "## Active Skills\n- init_a\n- init_b"
    )  # from the _build_runtime_system_context stub in _make_runner
    # _last_bound_tool_names is unchanged from default
    assert runner._last_bound_tool_names == frozenset()


async def test_scenario_13_session_bootstrap_init_failure_keeps_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If SkillTool.initialize raises during bootstrap, all 3 fields stay default."""
    runner = _make_runner(monkeypatch)
    runner._skill_tool.raise_on_initialize = RuntimeError("simulated")

    with pytest.raises(RuntimeError, match="simulated"):
        await runner._apply_preselected_skills([_build_skill("a")])

    # All three remain at their default empty values
    assert runner._last_skill_context == ""
    assert runner._last_skill_ids == ()
    assert runner._last_initialized_skill_ids == ()


# ---- Scenario 14: New-message boundary via _apply_preselected_skills ------ #


async def test_scenario_14_new_message_boundary_with_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-message apply forwards scores to _build_runtime_system_context."""
    runner = _make_runner(monkeypatch)
    captured_scores: list[list[float] | None] = []

    monkeypatch.setattr(
        runner,
        "_build_runtime_system_context",
        lambda skills, scores=None: (
            captured_scores.append(scores)
            or "## with scores"
        ),
    )

    await runner._apply_preselected_skills(
        [_build_skill("scored")], scores=[0.9, 0.7]
    )

    assert captured_scores == [[0.9, 0.7]]
    assert runner._last_skill_context == "## with scores"
    assert runner._last_skill_ids == ("scored",)


async def test_scenario_14_new_message_boundary_init_failure_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Init failure during per-message apply preserves prior state."""
    runner = _make_runner(monkeypatch)
    _seed_pre_call_state(runner)

    runner._skill_tool.raise_on_initialize = RuntimeError("init boom")

    with pytest.raises(RuntimeError, match="init boom"):
        await runner._apply_preselected_skills([_build_skill("new")])

    assert runner._last_skill_context == "## Previous context"
    assert runner._last_skill_ids == ("prev_skill_1",)
    assert runner._last_initialized_skill_ids == ("prev_skill_1",)


# ---- Scenario 15: Step-lock activation via _activate_step_skills --------- #


async def test_scenario_15_step_lock_activation_routes_through_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_activate_step_skills now routes through _apply_preselected_skills.

    Verifies that after step lock activation, the three _last_* fields advance.
    """
    runner = _make_runner(monkeypatch)
    target_skills = [_build_skill("step_a"), _build_skill("step_b")]

    await runner._activate_step_skills(
        step_id="step-1",
        user_message="dummy message",
        selected_skills=target_skills,
    )

    assert runner._last_skill_ids == ("step_a", "step_b")
    assert runner._last_initialized_skill_ids == ("step_a", "step_b")
    assert "step_a" in runner._last_skill_context
    assert runner._step_skill_state is not None
    assert runner._step_skill_state.step_id == "step-1"


async def test_scenario_15_step_lock_init_failure_no_state_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If apply raises in _activate_step_skills, _step_skill_state is NOT created."""
    runner = _make_runner(monkeypatch)
    runner._skill_tool.raise_on_initialize = RuntimeError("init boom")
    snapshot = _seed_pre_call_state(runner)

    with pytest.raises(RuntimeError, match="init boom"):
        await runner._activate_step_skills(
            step_id="step-fail",
            user_message="dummy",
            selected_skills=[_build_skill("new")],
        )

    # _step_skill_state was NOT created because apply raised before line that sets it
    assert runner._step_skill_state is None
    # Three _last_* fields preserved (rollback on init failure)
    assert runner._last_skill_context == snapshot["skill_context"]
    assert runner._last_skill_ids == snapshot["skill_ids"]
    assert runner._last_initialized_skill_ids == snapshot["initialized_skill_ids"]


# ---- Scenario 16: Unknown-tool reselect field split (Site 4) ------------- #


async def test_scenario_16_unknown_tool_reselect_apply_success_advances_locked_skills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive _handle_step_skill_lock through a real ToolEvent with the
    unknown-tool sentinel and verify the production field-split semantics
    on the success path: all three state fields advance.
    """
    from app.domain.models.event import ToolEvent, ToolEventStatus
    from app.domain.models.tool_result import ToolResult
    from app.domain.services.agent_task_runner import StepSkillActivationState

    runner = _make_runner(monkeypatch)
    # Pre-seed _step_skill_state with consecutive_unknown_tool_calls just
    # below threshold so the next event's increment trips the reselect path.
    threshold = runner._skill_selection_policy.step_skill_reselect_unknown_tool_threshold
    runner._step_skill_state = StepSkillActivationState(
        step_id="s-1",
        user_message="msg",
        locked_skills=[_build_skill("old1")],
    )
    runner._step_skill_state.consecutive_unknown_tool_calls = threshold - 1
    runner._step_skill_state.reselect_count = 0

    # Stub _select_skills_for_message to return new skills
    new_skills = [_build_skill("new1"), _build_skill("new2")]

    async def fake_select(pool, msg):
        return (list(new_skills), None)

    monkeypatch.setattr(runner, "_select_skills_for_message", fake_select)

    # Construct an unknown-tool ToolEvent
    event = ToolEvent(
        id="evt-1",
        tool_call_id="tc-1",
        tool_name="skills",
        function_name="missing_tool",
        function_args={},
        function_result=ToolResult(success=False, data={"code": "UNKNOWN_TOOL"}),
        status=ToolEventStatus.CALLED,
    )

    # Drive the production path
    await runner._handle_step_skill_lock(event, "user message")

    # Field-split assertions on the production state object
    state = runner._step_skill_state
    assert state.reselect_count == 1, "attempt counter advances on success"
    assert state.consecutive_unknown_tool_calls == 0, "reset on success"
    assert [s.id for s in state.locked_skills] == ["new1", "new2"], (
        "success commit applied"
    )
    # Helper-side: _apply_preselected_skills also wrote runner._last_*
    assert runner._last_skill_ids == ("new1", "new2")


async def test_scenario_16_unknown_tool_reselect_apply_failure_attempt_counter_advances_lock_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive _handle_step_skill_lock through a real ToolEvent and force
    _apply_preselected_skills to raise. Verify the production field-split
    semantics on the failure path:
      (a) state.locked_skills preserved (success commit semantics)
      (b) state.reselect_count IS advanced (attempt counter semantics —
          preserves max_reselect cap protection)
      (c) state.consecutive_unknown_tool_calls reset (also advanced before raise)
    """
    from app.domain.models.event import ToolEvent, ToolEventStatus
    from app.domain.models.tool_result import ToolResult
    from app.domain.services.agent_task_runner import StepSkillActivationState

    runner = _make_runner(monkeypatch)
    threshold = runner._skill_selection_policy.step_skill_reselect_unknown_tool_threshold
    pre_locked = [_build_skill("orig_a"), _build_skill("orig_b")]
    runner._step_skill_state = StepSkillActivationState(
        step_id="s-1",
        user_message="msg",
        locked_skills=list(pre_locked),
    )
    runner._step_skill_state.consecutive_unknown_tool_calls = threshold - 1
    runner._step_skill_state.reselect_count = 0

    new_skills = [_build_skill("new1")]

    async def fake_select(pool, msg):
        return (list(new_skills), None)

    monkeypatch.setattr(runner, "_select_skills_for_message", fake_select)

    runner._skill_tool.raise_on_initialize = RuntimeError("apply boom")

    event = ToolEvent(
        id="evt-2",
        tool_call_id="tc-2",
        tool_name="skills",
        function_name="missing_tool",
        function_args={},
        function_result=ToolResult(success=False, data={"code": "UNKNOWN_TOOL"}),
        status=ToolEventStatus.CALLED,
    )

    with pytest.raises(RuntimeError, match="apply boom"):
        await runner._handle_step_skill_lock(event, "user message")

    state = runner._step_skill_state
    # Attempt counter advanced (a) — max_reselect cap protection preserved
    assert state.reselect_count == 1, "reselect_count must advance on attempt"
    assert state.consecutive_unknown_tool_calls == 0, "consecutive counter reset"
    # Success commit NOT applied (b)
    assert [s.id for s in state.locked_skills] == [s.id for s in pre_locked], (
        "locked_skills preserved on apply failure"
    )
