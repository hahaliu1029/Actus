"""Tests for AgentTaskRunner._apply_preselected_skills helper.

Covers the new atomic entry point introduced by TODO #30
(see docs/superpowers/specs/2026-04-13-activate-step-skills-atomicity-design.md §3.1).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.domain.models.app_config import AgentConfig, A2AConfig, MCPConfig
from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType
from app.domain.services.agent_task_runner import AgentTaskRunner

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


def _build_skill(skill_id: str) -> Skill:
    return Skill(
        id=skill_id,
        slug=skill_id,
        name=skill_id,
        description=f"{skill_id} description",
        source_type=SkillSourceType.GITHUB,
        source_ref=f"github:test/{skill_id}",
        runtime_type=SkillRuntimeType.NATIVE,
        manifest={"runtime_type": "native", "tools": []},
        enabled=True,
    )


class _FakeSkillTool:
    def __init__(self) -> None:
        self.initialize_calls: list[list[Skill]] = []
        self._loaded_skill_ids: tuple[str, ...] = ()

    async def initialize(self, skills: list[Skill]) -> None:
        self.initialize_calls.append(list(skills))
        self._loaded_skill_ids = tuple(s.id for s in skills)

    async def cleanup(self) -> None:
        self._loaded_skill_ids = ()

    def get_tools(self) -> list[dict[str, Any]]:
        return [{"name": f"skill_{sid}"} for sid in self._loaded_skill_ids]


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


def _make_runner(monkeypatch: pytest.MonkeyPatch) -> AgentTaskRunner:
    """Build a minimally-initialized AgentTaskRunner for helper tests."""

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
        session_id="apply-preselected-test",
        user_id="user-test",
        file_storage=object(),
        browser=object(),
        search_engine=object(),
        sandbox=_FakeSandbox(),
    )
    runner._mcp_tool = _FakeMCPTool()
    runner._a2a_tool = _FakeA2ATool()
    runner._skill_tool = _FakeSkillTool()
    return runner


async def test_apply_preselected_skills_init_first_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Init MUST be called BEFORE _build_runtime_system_context.

    The context builder reads ``self._skill_tool.get_tools()`` for the
    available-tool summary, so reading before init would produce a
    "new skill guide + stale tool summary" mixed context.
    See Codex review round 2 HIGH #1.
    """
    runner = _make_runner(monkeypatch)
    skills = [_build_skill("s1"), _build_skill("s2")]

    call_order: list[str] = []

    async def mock_init(skill_list: list[Skill]) -> None:
        call_order.append("init")

    def mock_build_context(
        skill_list: list[Skill], scores: list[float] | None = None
    ) -> str:
        call_order.append("build_context")
        return "## context"

    monkeypatch.setattr(runner, "_initialize_skill_tool_if_needed", mock_init)
    monkeypatch.setattr(runner, "_build_runtime_system_context", mock_build_context)

    await runner._apply_preselected_skills(skills)

    assert call_order == ["init", "build_context"], (
        f"Expected init before build_context, got {call_order}"
    )


async def test_apply_preselected_skills_context_reads_fresh_skill_tool_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_build_runtime_system_context must see the post-init SkillTool state."""
    runner = _make_runner(monkeypatch)
    skills = [_build_skill("a"), _build_skill("b")]

    captured_tools: list[list[dict[str, Any]]] = []

    def fake_build_context(
        skill_list: list[Skill], scores: list[float] | None = None
    ) -> str:
        captured_tools.append(runner._skill_tool.get_tools())
        return "## context"

    monkeypatch.setattr(runner, "_build_runtime_system_context", fake_build_context)

    await runner._apply_preselected_skills(skills)

    assert len(captured_tools) == 1
    assert {t["name"] for t in captured_tools[0]} == {"skill_a", "skill_b"}, (
        "Expected build_context to see post-init SkillTool state with both skills"
    )


async def test_apply_preselected_skills_advances_three_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All three _last_* fields are written after a successful apply."""
    runner = _make_runner(monkeypatch)
    monkeypatch.setattr(
        runner,
        "_build_runtime_system_context",
        lambda skills, scores=None: "## new context",
    )
    skills = [_build_skill("x"), _build_skill("y")]

    await runner._apply_preselected_skills(skills)

    assert runner._last_skill_context == "## new context"
    assert runner._last_skill_ids == ("x", "y")
    assert runner._last_initialized_skill_ids == ("x", "y")


async def test_apply_preselected_skills_init_failure_rolls_back_all_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If _initialize_skill_tool_if_needed raises, all three fields stay pre-call."""
    runner = _make_runner(monkeypatch)
    runner._last_skill_context = "## prev"
    runner._last_skill_ids = ("prev_skill",)
    runner._last_initialized_skill_ids = ("prev_skill",)

    async def raising_init(skill_list: list[Skill]) -> None:
        raise RuntimeError("simulated init failure")

    monkeypatch.setattr(runner, "_initialize_skill_tool_if_needed", raising_init)
    monkeypatch.setattr(
        runner,
        "_build_runtime_system_context",
        lambda skills, scores=None: "## new",
    )

    with pytest.raises(RuntimeError, match="simulated init failure"):
        await runner._apply_preselected_skills([_build_skill("new")])

    assert runner._last_skill_context == "## prev"
    assert runner._last_skill_ids == ("prev_skill",)
    assert runner._last_initialized_skill_ids == ("prev_skill",)


async def test_apply_preselected_skills_context_failure_after_init_documented_split_brain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Document the residual split-brain risk when context build raises after init.

    This test exists to LOCK IN the contract — it does not assert correctness
    in the rollback sense, but documents that:
    - _last_initialized_skill_ids HAS advanced (init succeeded)
    - _last_skill_context / _last_skill_ids did NOT advance (assignment skipped)

    See spec §6.2 risk #1 and §3.1 docstring for rationale.
    """
    runner = _make_runner(monkeypatch)
    runner._last_skill_context = "## prev"
    runner._last_skill_ids = ("prev_skill",)
    runner._last_initialized_skill_ids = ("prev_skill",)

    def raising_build(
        skill_list: list[Skill], scores: list[float] | None = None
    ) -> str:
        raise RuntimeError("simulated context build failure")

    monkeypatch.setattr(runner, "_build_runtime_system_context", raising_build)

    new_skills = [_build_skill("new")]
    with pytest.raises(RuntimeError, match="simulated context build failure"):
        await runner._apply_preselected_skills(new_skills)

    # init advanced (post-#27 atomic guarantee + assignment is inside init helper)
    assert runner._last_initialized_skill_ids == ("new",)
    # The two assignments below the raise did not run
    assert runner._last_skill_context == "## prev"
    assert runner._last_skill_ids == ("prev_skill",)


async def test_apply_preselected_skills_threads_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scores parameter must be forwarded to _build_runtime_system_context."""
    runner = _make_runner(monkeypatch)
    captured_scores: list[list[float] | None] = []

    def fake_build(
        skill_list: list[Skill], scores: list[float] | None = None
    ) -> str:
        captured_scores.append(scores)
        return "## ctx"

    monkeypatch.setattr(runner, "_build_runtime_system_context", fake_build)

    await runner._apply_preselected_skills([_build_skill("s")], scores=[0.9, 0.8])

    assert captured_scores == [[0.9, 0.8]]


async def test_apply_preselected_skills_empty_skills_clears_skill_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty skills list is an EXPLICIT CLEAR — init is called, fields zero out.

    NOTE: _last_skill_context is NOT empty string — it equals the output of
    _build_runtime_system_context([]) which still includes the tool summary
    section for native / MCP / A2A / memory tools.
    """
    runner = _make_runner(monkeypatch)
    runner._last_skill_context = "## prev"
    runner._last_skill_ids = ("prev",)
    runner._last_initialized_skill_ids = ("prev",)

    expected_empty_context = "## Available Tools\n- shell\n- file_view"
    monkeypatch.setattr(
        runner,
        "_build_runtime_system_context",
        lambda skills, scores=None: expected_empty_context,
    )

    await runner._apply_preselected_skills([])

    assert runner._last_initialized_skill_ids == ()
    assert runner._last_skill_ids == ()
    assert runner._last_skill_context == expected_empty_context, (
        "Empty skills clears the selection but tool summary still appears"
    )
    # SkillTool.initialize was actually called with the empty list
    assert runner._skill_tool.initialize_calls == [[]]


async def test_apply_preselected_skills_non_empty_to_empty_transition_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bootstrap-style transition: first non-empty, then empty (e.g. config reload).

    Verifies that a second call with [] correctly clears state established by
    a prior call with skills.
    """
    runner = _make_runner(monkeypatch)
    monkeypatch.setattr(
        runner,
        "_build_runtime_system_context",
        lambda skills, scores=None: (
            "## Active Skills\n" + "\n".join(f"- {s.id}" for s in skills)
            if skills
            else "## Tools only"
        ),
    )

    # First call: [A, B, C]
    await runner._apply_preselected_skills(
        [_build_skill("A"), _build_skill("B"), _build_skill("C")]
    )
    assert runner._last_skill_ids == ("A", "B", "C")
    assert "Active Skills" in runner._last_skill_context

    # Second call: empty
    await runner._apply_preselected_skills([])
    assert runner._last_skill_ids == ()
    assert runner._last_skill_context == "## Tools only"
    assert "Active Skills" not in runner._last_skill_context


async def test_apply_preselected_skills_non_empty_to_empty_transition_new_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """New-message boundary transition with cache key implication.

    After the second (empty) call, the cache key tuple (_last_skill_ids,)
    is () — a valid empty key, NOT a stale stale ('A', 'B', 'C') key.
    """
    runner = _make_runner(monkeypatch)
    monkeypatch.setattr(
        runner,
        "_build_runtime_system_context",
        lambda skills, scores=None: f"ctx-{len(skills)}",
    )

    await runner._apply_preselected_skills(
        [_build_skill("A"), _build_skill("B")], scores=[0.8, 0.6]
    )
    first_cache_key = runner._last_skill_ids

    await runner._apply_preselected_skills([])
    second_cache_key = runner._last_skill_ids

    assert first_cache_key == ("A", "B")
    assert second_cache_key == ()
    assert first_cache_key != second_cache_key


async def test_apply_preselected_skills_non_empty_to_empty_transition_unknown_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown-tool reselect transition: SkillTool.initialize([]) is invoked."""
    runner = _make_runner(monkeypatch)
    monkeypatch.setattr(
        runner,
        "_build_runtime_system_context",
        lambda skills, scores=None: f"ctx-{len(skills)}",
    )

    await runner._apply_preselected_skills(
        [_build_skill("X"), _build_skill("Y"), _build_skill("Z")]
    )
    assert runner._skill_tool.initialize_calls[-1] == [
        _build_skill("X"),
        _build_skill("Y"),
        _build_skill("Z"),
    ] or [s.id for s in runner._skill_tool.initialize_calls[-1]] == ["X", "Y", "Z"]

    await runner._apply_preselected_skills([])
    # _initialize_skill_tool_if_needed checks if skill_ids match; () != ("X","Y","Z")
    # so it forwards the call to SkillTool.initialize([])
    assert runner._skill_tool.initialize_calls[-1] == []
    assert runner._last_skill_ids == ()


async def test_apply_preselected_skills_does_not_touch_last_bound_tool_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_last_bound_tool_names is owned by _build_step_react_graph Phase 3.

    The pre-selected helper must not touch it.
    """
    runner = _make_runner(monkeypatch)
    monkeypatch.setattr(
        runner,
        "_build_runtime_system_context",
        lambda skills, scores=None: "ctx",
    )
    pre_call = frozenset({"shell_execute", "file_read"})
    runner._last_bound_tool_names = pre_call

    await runner._apply_preselected_skills([_build_skill("s")])

    assert runner._last_bound_tool_names == pre_call


async def test_apply_preselected_skills_dedups_by_skill_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate skill IDs in input are deduplicated (first-occurrence order).

    Without dedup, a caller passing [skill_a, skill_a, skill_b] would set
    _last_skill_ids to ("a", "a", "b"), and the next call with the
    deduplicated equivalent [skill_a, skill_b] would mismatch against the
    stale tuple and trigger spurious re-init.
    """
    runner = _make_runner(monkeypatch)
    monkeypatch.setattr(
        runner,
        "_build_runtime_system_context",
        lambda skill_list, scores=None: f"ctx-{len(skill_list)}",
    )

    skill_a = _build_skill("a")
    skill_b = _build_skill("b")

    # First call with duplicate "a"
    await runner._apply_preselected_skills([skill_a, skill_a, skill_b])

    # _last_skill_ids should NOT contain duplicates
    assert runner._last_skill_ids == ("a", "b"), (
        f"Expected dedup'd tuple ('a', 'b'), got {runner._last_skill_ids}"
    )
    assert runner._last_initialized_skill_ids == ("a", "b")
    # SkillTool.initialize was called with the dedup'd list (one call)
    assert len(runner._skill_tool.initialize_calls) == 1
    assert [s.id for s in runner._skill_tool.initialize_calls[0]] == ["a", "b"]

    # Second call with the already-deduplicated equivalent — must NOT re-init
    await runner._apply_preselected_skills([skill_a, skill_b])
    assert runner._last_skill_ids == ("a", "b")
    assert len(runner._skill_tool.initialize_calls) == 1, (
        "no spurious re-init: dedup'd tuples already match"
    )
