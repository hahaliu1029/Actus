"""[C2b rollout WS0 §3A.4] Flag-off hard sanitation: parallel_work_units in the
parsed LLM response must be cleared at the parse->Step boundary so no flag-off
session (planner OR detection OR updater) can route into the coordinator."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from app.domain.models.llm_responses import PlanResponse, StepDef
from app.domain.models.work_unit import ParallelWorkUnitGroupRequest, WorkUnitRequest
from app.domain.services.graphs.main_graph import _build_plan_from_response

_FLAG = "ACTUS_C2_COORDINATOR_ENABLED"


def _response_with_pwu() -> PlanResponse:
    pwu = ParallelWorkUnitGroupRequest(
        work_units=[
            WorkUnitRequest(
                objective="analyze foo",
                phase="exploration",
                allowed_tools=["file_read"],
            )
        ]
    )
    return PlanResponse(steps=[StepDef(description="parallel step", parallel_work_units=pwu)])


def test_flag_off_clears_parallel_work_units(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_FLAG, raising=False)
    plan = _build_plan_from_response(_response_with_pwu())
    assert plan.steps, "expected one step"
    assert all(s.parallel_work_units is None for s in plan.steps)


def test_flag_on_preserves_parallel_work_units(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_FLAG, "true")
    plan = _build_plan_from_response(_response_with_pwu())
    assert plan.steps[0].parallel_work_units is not None
    assert len(plan.steps[0].parallel_work_units.work_units) == 1


def test_detection_path_shares_sanitized_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """flows/planner_react.py:1239 calls _build_plan_from_response directly
    (detection planner bypasses planner_node). Pin that the helper — the single
    shared Step-construction point — sanitizes for that path too."""
    monkeypatch.delenv(_FLAG, raising=False)
    plan = _build_plan_from_response(_response_with_pwu(), plan_id="detection")
    assert all(s.parallel_work_units is None for s in plan.steps)


def test_child_runner_strips_parallel_work_units_even_flag_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[child-pwu fix] Coordinator children run WITHOUT
    configurable.parallel_execution_subgraph (depth cap = 1, child factory
    never wires coord_deps), so a child plan carrying parallel_work_units
    hard-crashes at the executor guard (main_graph.py:100-108), cancels the
    sibling (sibling_terminal_failed) and fails the reduce. The parse->Step
    boundary must therefore ALSO strip when the runner says dispatch is not
    allowed — even with the global flag ON."""
    monkeypatch.setenv(_FLAG, "true")
    plan = _build_plan_from_response(
        _response_with_pwu(), allow_parallel_work_units=False
    )
    assert plan.steps, "expected one step"
    assert all(s.parallel_work_units is None for s in plan.steps)


def test_parallel_dispatch_allowed_false_without_subgraph() -> None:
    """[child-pwu fix] No ``parallel_execution_subgraph`` in configurable ⇒
    this runner cannot dispatch (coordinator child / legacy runner without
    coord_deps). Keeping pwu would ALWAYS crash at the executor guard, so
    the only safe answer is False."""
    from app.domain.services.graphs.main_graph import _parallel_dispatch_allowed

    assert _parallel_dispatch_allowed(None) is False
    assert _parallel_dispatch_allowed({}) is False
    assert _parallel_dispatch_allowed({"configurable": {}}) is False
    assert _parallel_dispatch_allowed({"configurable": None}) is False


def test_parallel_dispatch_allowed_true_when_subgraph_wired() -> None:
    """[child-pwu fix] Root runners get the subgraph via coord_deps →
    PlannerReActFlow._build_config; presence = dispatch capability."""
    from app.domain.services.graphs.main_graph import _parallel_dispatch_allowed

    cfg = {"configurable": {"parallel_execution_subgraph": object()}}
    assert _parallel_dispatch_allowed(cfg) is True


def test_executor_gate_raises_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """[§3A.5] Last-line defense: assert_coordinator_enabled() raises flag-off.
    This is the invariant the dark-launch integration test used to cover via SSE;
    pinned here as a fast local unit test."""
    from app.domain.services.coordinator_feature_flag import assert_coordinator_enabled

    monkeypatch.delenv(_FLAG, raising=False)
    with pytest.raises(RuntimeError, match="ACTUS_C2_COORDINATOR_ENABLED"):
        assert_coordinator_enabled()


@pytest.mark.anyio
async def test_child_checkpoint_parallel_step_executes_sequentially(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child may resume a checkpoint written before sanitation.  Explicitly
    disabled dispatch must route that stale parallel step through normal ReAct
    instead of crashing on the intentionally absent coordinator subgraph.
    """
    from app.domain.services.graphs.main_graph import build_main_graph

    monkeypatch.setenv(_FLAG, "true")
    plan = _build_plan_from_response(
        _response_with_pwu(), allow_parallel_work_units=True
    )
    planner_llm = MagicMock()

    class MockReactGraph:
        async def astream(self, input_state, config=None, **kwargs):
            yield {
                "llm_node": {
                    "events": [],
                    "messages": [
                        AIMessage(
                            content=(
                                '{"success": true, "result": "done", '
                                '"attachments": []}'
                            )
                        )
                    ],
                    "should_interrupt": False,
                }
            }

    graph = build_main_graph(
        _allow_default_prompt_assembler=True,
        planner_llm=planner_llm,
        react_graph=MockReactGraph(),
        summary_llm=planner_llm,
        uow_factory=MagicMock(),
        session_id="child-session",
    )

    result = await graph.ainvoke(
        {
            "message": "analyze foo",
            "language": "en",
            "attachments": [],
            "image_content_blocks": [],
            "plan": plan,
            "current_step": plan.steps[0],
            "messages": [],
            "execution_summary": "",
            "events": [],
            "flow_status": "executing",
            "session_id": "child-session",
            "should_interrupt": False,
            "resume_value": None,
            "original_request": "",
            "skill_context": "",
            "conversation_summaries": [],
        },
        config={"configurable": {"parallel_dispatch_allowed": False}},
    )

    assert result["plan"].steps[0].status.value == "completed"


def _step_with_pwu_path(path: str | None):
    """Build a Step carrying pwu directly (bypasses the flag-gated parse
    boundary so these unit tests are env-independent). ``None`` = exploration
    unit with no proposed_paths."""
    from app.domain.models.plan import Step

    wu_kwargs: dict = dict(
        objective="w", phase="write", allowed_tools=["file_write"],
    )
    if path is not None:
        wu_kwargs["proposed_paths"] = [{"path": path, "op": "add"}]
    else:
        wu_kwargs["phase"] = "exploration"
        wu_kwargs["allowed_tools"] = ["file_read"]
    return Step(
        id="s1",
        description="write",
        parallel_work_units=ParallelWorkUnitGroupRequest(
            work_units=[WorkUnitRequest(**wu_kwargs)]
        ),
    )


def test_pwu_paths_contract_ok_accepts_valid_and_empty() -> None:
    """[dispatch-fallback fix] Directory-qualified relative paths pass; an
    exploration unit with no proposed_paths trivially passes."""
    from app.domain.services.graphs.main_graph import _pwu_paths_contract_ok

    assert _pwu_paths_contract_ok(_step_with_pwu_path("workspace/a.md")) is True
    assert _pwu_paths_contract_ok(_step_with_pwu_path(None)) is True


def test_pwu_paths_contract_ok_rejects_workspace_root_absolute() -> None:
    """[dispatch-fallback fix] Live 2026-07-13 repro: the user asked for
    /home/ubuntu/part_{a,b,c}.md, glm-5.2 proposed those absolute paths,
    dispatch rejected the WHOLE step pre-spawn and the single-step plan
    ended as a fake-completed run with ZERO files written. The pre-check
    must flag these so executor degrades to sequential ReAct instead."""
    from app.domain.services.graphs.main_graph import _pwu_paths_contract_ok

    assert _pwu_paths_contract_ok(
        _step_with_pwu_path("/home/ubuntu/part_a.md")
    ) is False
    assert _pwu_paths_contract_ok(_step_with_pwu_path("part_a.md")) is False


@pytest.mark.anyio
async def test_invalid_pwu_paths_fall_back_to_react_not_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[dispatch-fallback fix] Contract-invalid proposed_paths must route the
    step through the normal ReAct branch (work still gets done) and must NOT
    invoke the parallel subgraph — pre-spawn, zero coordinator side effects."""
    from unittest.mock import AsyncMock

    from app.domain.services.graphs.main_graph import build_main_graph

    monkeypatch.setenv(_FLAG, "true")

    pwu = ParallelWorkUnitGroupRequest(work_units=[
        WorkUnitRequest(
            objective="write ocean line", phase="write",
            allowed_tools=["file_write"],
            proposed_paths=[{"path": "/home/ubuntu/part_a.md", "op": "add"}],
        )
    ])
    create_response = PlanResponse(
        title="Parallel", goal="write 3 files", language="en",
        steps=[StepDef(id="s1", description="parallel step", parallel_work_units=pwu)],
        message="ok",
    )
    create_structured = AsyncMock()
    create_structured.ainvoke = AsyncMock(return_value=create_response)
    planner_llm = MagicMock()
    planner_llm.with_structured_output = MagicMock(return_value=create_structured)

    subgraph = AsyncMock()
    subgraph.ainvoke = AsyncMock()

    class MockReactGraph:
        async def astream(self, input_state, config=None, **kwargs):
            yield {
                "llm_node": {
                    "events": [],
                    "messages": [
                        AIMessage(
                            content=(
                                '{"success": true, "result": "done", '
                                '"attachments": []}'
                            )
                        )
                    ],
                    "should_interrupt": False,
                }
            }

    graph = build_main_graph(
        _allow_default_prompt_assembler=True,
        planner_llm=planner_llm,
        react_graph=MockReactGraph(),
        summary_llm=planner_llm,
        uow_factory=MagicMock(),
        session_id="root-session",
    )

    result = await graph.ainvoke(
        {
            "message": "write three files",
            "language": "en",
            "attachments": [],
            "image_content_blocks": [],
            "plan": None,
            "current_step": None,
            "messages": [],
            "execution_summary": "",
            "events": [],
            "flow_status": "idle",
            "session_id": "root-session",
            "should_interrupt": False,
            "resume_value": None,
            "original_request": "",
            "skill_context": "",
            "conversation_summaries": [],
        },
        config={"configurable": {
            "parallel_execution_subgraph": subgraph,
            "user_id": "u1",
        }},
    )

    subgraph.ainvoke.assert_not_awaited()
    assert result["plan"].steps[0].status.value == "completed"
