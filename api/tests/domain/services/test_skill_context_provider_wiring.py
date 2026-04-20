"""Tests for PlannerReActFlow._skill_context_provider callback and seed helper.

Covers the clock-2-field replacement introduced by TODO #30
(see docs/superpowers/specs/2026-04-13-activate-step-skills-atomicity-design.md §3.3).
"""
from __future__ import annotations

import pytest

from app.domain.services.flows.planner_react import PlannerReActFlow

pytestmark = pytest.mark.anyio


def _build_flow_minimal() -> PlannerReActFlow:
    """Construct a minimally-initialized PlannerReActFlow without a runner.

    Only fields needed by ``_get_skill_context_seed`` are populated.
    Other dependencies (LLM, sandbox, tools) are not exercised by these tests.
    """
    flow = PlannerReActFlow.__new__(PlannerReActFlow)
    flow._skill_context_provider = None
    return flow


def test_get_skill_context_seed_when_provider_none() -> None:
    """``_get_skill_context_seed`` returns ``""`` when no provider wired.

    Test harness scenarios that construct PlannerReActFlow without a runner
    must not crash on read.
    """
    flow = _build_flow_minimal()
    assert flow._get_skill_context_seed() == ""


def test_get_skill_context_seed_returns_provider_value() -> None:
    """When provider is wired, _get_skill_context_seed returns its result."""
    flow = _build_flow_minimal()
    flow._skill_context_provider = lambda: "## Active Skills\n- example"
    assert flow._get_skill_context_seed() == "## Active Skills\n- example"


def test_get_skill_context_seed_returns_empty_when_provider_returns_empty() -> None:
    """A provider that returns empty string is forwarded as-is."""
    flow = _build_flow_minimal()
    flow._skill_context_provider = lambda: ""
    assert flow._get_skill_context_seed() == ""


def test_provider_field_is_optional_callable_attribute() -> None:
    """The new field has correct shape: Callable[[], str] | None."""
    flow = _build_flow_minimal()
    # None is the unwired default
    assert flow._skill_context_provider is None
    # Can be assigned to a callable
    flow._skill_context_provider = lambda: "test"
    assert callable(flow._skill_context_provider)
    assert flow._skill_context_provider() == "test"


def test_provider_lambda_captures_runner_reference() -> None:
    """Verify the wiring pattern: lambda closes over a stable object reference.

    This mirrors how AgentTaskRunner will wire the callback:
    ``flow._skill_context_provider = lambda: self._last_skill_context``
    """
    class _FakeRunner:
        def __init__(self) -> None:
            self._last_skill_context = "initial"

    runner = _FakeRunner()
    flow = _build_flow_minimal()
    flow._skill_context_provider = lambda: runner._last_skill_context

    assert flow._get_skill_context_seed() == "initial"

    # When runner mutates its state, the seed reflects the new value
    runner._last_skill_context = "updated"
    assert flow._get_skill_context_seed() == "updated"


def test_input_for_graph_branches_use_provider_seed() -> None:
    """The three 'skill_context' dict-key sites at planner_react.py L763 /
    L1012 / L1035 must read via _get_skill_context_seed (which calls the
    provider). This is a STATIC regression check — it catches anyone who
    reverts to ``self._skill_context``.

    Runtime coverage for these read points lives in
    test_run_planner_for_detection_reads_provider (below) and
    test_resume_path_does_not_read_provider (below).
    """
    import ast
    from pathlib import Path

    pr_path = (
        Path(__file__).resolve().parents[3]
        / "app"
        / "domain"
        / "services"
        / "flows"
        / "planner_react.py"
    )
    source = pr_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Find all dict literal entries with "skill_context" as a key
    skill_context_value_strs: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values):
            if (
                isinstance(k, ast.Constant)
                and isinstance(k.value, str)
                and k.value == "skill_context"
            ):
                skill_context_value_strs.append(ast.unparse(v))

    # Production code should have exactly 3 occurrences (L763, L1012, L1035),
    # all calling _get_skill_context_seed() (NOT reading self._skill_context).
    assert len(skill_context_value_strs) == 3, (
        f"Expected 3 'skill_context' dict-key entries in planner_react.py, "
        f"got {len(skill_context_value_strs)}: {skill_context_value_strs}"
    )
    for value_str in skill_context_value_strs:
        assert "_get_skill_context_seed" in value_str, (
            f"Found 'skill_context' value that doesn't use _get_skill_context_seed: {value_str}"
        )


async def test_run_planner_for_detection_reads_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_run_planner_for_detection must read skill_context via
    _get_skill_context_seed (which calls the provider callback), not
    via the retired self._skill_context field.

    Approach: spy on _skill_context_provider; stub the prompt_assembler
    and LLM enough for the method to reach the detection_state
    construction site. Assert the spy was called and the value flowed
    through to the render context.
    """
    from unittest.mock import AsyncMock, MagicMock
    # NOTE: ``Message`` lives in ``app.domain.models.message`` (see
    # production import at planner_react.py:37). ``app.domain.models.event``
    # defines the event hierarchy, not the Message DTO — importing from
    # there would fail at test collection.
    from app.domain.models.message import Message
    from app.domain.services.flows.planner_react import PlannerReActFlow

    flow = PlannerReActFlow.__new__(PlannerReActFlow)
    # Spy provider
    provider_calls = []
    flow._skill_context_provider = lambda: (
        provider_calls.append(1) or "## TEST_SEED"
    )

    # Stub enough attributes for _run_planner_for_detection to reach the
    # detection_state construction site without blowing up.
    flow._allow_default_prompt_assembler = False
    flow._supports_vision = False
    flow._agent_config = MagicMock(
        tool_confirmation=MagicMock(enabled=False, timeout_seconds=300)
    )
    flow._llm = MagicMock()
    flow._session_id = "detection-test"
    # Added by commit 84e1853 — salvage_empty_memory_recall_plan(has_memory_tools=...)
    # reads this; production code sets it in __init__ L240 + _collect_all_tools L409,
    # but __new__ bypasses both.
    flow._has_memory_tools = False

    # Capture the state passed to build_render_context (which reads
    # detection_state["skill_context"]).
    captured = {}

    def fake_assemble(sections, ctx, mode, fallback_used=False):
        captured["ctx_skill_context"] = getattr(ctx, "skill_context", None)
        result = MagicMock()
        result.text = "stub prompt"
        return result

    flow._prompt_assembler = MagicMock()
    flow._prompt_assembler.assemble = fake_assemble

    # Short-circuit the structured_llm call to return a minimal plan
    fake_plan_response = MagicMock()
    fake_plan_response.title = "Task"
    fake_plan_response.goal = "goal"
    fake_plan_response.language = "en"
    fake_plan_response.steps = []
    fake_plan_response.message = "ok"
    flow._llm.with_structured_output = MagicMock(
        return_value=MagicMock(ainvoke=AsyncMock(return_value=fake_plan_response))
    )

    message = Message(
        message="detection test message",
        language="en",
        attachments=[],
        image_content_blocks=[],
    )

    await flow._run_planner_for_detection(message, summary_texts=[])

    # Verify the provider was called during detection
    assert len(provider_calls) >= 1, (
        "_run_planner_for_detection must call _skill_context_provider "
        "(via _get_skill_context_seed) when building detection_state"
    )
    # And the seed value flowed through to the render context
    assert captured.get("ctx_skill_context") == "## TEST_SEED", (
        f"ctx.skill_context should be '## TEST_SEED', got {captured.get('ctx_skill_context')!r}"
    )


async def test_resume_path_does_not_read_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """flow.resume() must NOT call _skill_context_provider.

    Spec §6.2 #4 claim: 'fresh-runner resume is safe with provider=None
    because resume reads state.skill_context from the Postgres checkpoint,
    not from the provider callback.' This test locks that invariant in.

    **Gotcha**: flow.resume() evaluates ``self._main_graph`` at the
    ``bridge.run(self._main_graph, ...)`` call site (planner_react.py
    L1073). Even though the bridge is monkey-patched to a no-op async
    generator, Python still looks up the attribute as an argument, so
    the test MUST set ``flow._main_graph`` (to any sentinel) before
    calling resume. Same for ``_session_id``, ``_build_config``,
    ``_uow_factory``, ``_ensure_graphs``, ``_persist_after_graph``,
    and the two ``_deferred_*`` fields (used in the ``finally`` block).
    """
    from unittest.mock import AsyncMock, MagicMock
    from langgraph.types import Command
    from app.domain.services.flows.planner_react import PlannerReActFlow

    flow = PlannerReActFlow.__new__(PlannerReActFlow)

    # --- Provider spy — must NEVER be called during resume --- #
    provider_spy = MagicMock(return_value="would-be-seed")
    flow._skill_context_provider = provider_spy

    # --- Minimum attribute stubs for flow.resume() to reach the
    #     bridge.run() no-op without AttributeError --- #
    flow._ensure_graphs = AsyncMock(return_value=None)
    flow._build_config = MagicMock(return_value={"configurable": {}})
    flow._session_id = "resume-test"
    # ``_main_graph`` is looked up as a positional arg to bridge.run()
    # at L1073 — must exist even though bridge.run() ignores it.
    flow._main_graph = MagicMock(name="main_graph_sentinel")

    class _FakeUow:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        session = MagicMock(get_summary=AsyncMock(return_value=[]))

    flow._uow_factory = lambda: _FakeUow()

    # Replace GraphEventBridge so its ``run`` is a no-op async generator
    # and ``final_state`` has the expected shape for the finally block.
    async def fake_bridge_run(*args, **kwargs):
        if False:
            yield  # empty async generator

    monkeypatch.setattr(
        "app.domain.services.flows.planner_react.GraphEventBridge",
        lambda: MagicMock(
            run=fake_bridge_run,
            final_state={"should_interrupt": False},
        ),
    )

    # Finally-block attributes
    flow._persist_after_graph = AsyncMock()
    flow._deferred_final_state = None
    flow._deferred_summaries = []

    command = Command(resume={"action": "approve", "scope": "once"})

    # Drive flow.resume to completion
    async for _ in flow.resume(command):
        pass

    # Critical assertion: the provider was never read during resume
    assert provider_spy.call_count == 0, (
        f"resume() must NOT call _skill_context_provider "
        f"(call_count={provider_spy.call_count}). Spec §6.2 #4 claim broken."
    )
