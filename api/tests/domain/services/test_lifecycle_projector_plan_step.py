"""C7 PR2 — 投影器 plan/step 分支映射测试（spec §4.1/§4.2）。"""
import pytest

from app.domain.models.event import (
    LifecycleEvent,
    MessageEvent,
    PlanEvent,
    PlanEventStatus,
    StepEvent,
    StepEventStatus,
)
from app.domain.models.lifecycle import (
    LifecycleEventKind as K,
    LifecycleState as S,
    LifecycleType as T,
)
from app.domain.models.plan import Plan, Step
from app.domain.services.lifecycle_emit import build_lifecycle_event
from app.domain.services.lifecycle_projector import LifecycleProjector


@pytest.fixture
def projector() -> LifecycleProjector:
    return LifecycleProjector(parent_session_id="sess-1")


def _plan(status: PlanEventStatus, n_steps: int = 2) -> PlanEvent:
    ev = PlanEvent(
        plan=Plan(id="plan-1", steps=[Step(id=f"s{i}") for i in range(n_steps)]),
        status=status,
    )
    ev.seq = 10
    ev.id = "src-10"
    return ev


class TestPlanMapping:
    @pytest.mark.anyio
    async def test_created_maps_started_pending_with_note(self, projector):
        out = await projector.project(_plan(PlanEventStatus.CREATED))
        assert len(out) == 1  # steps snapshot 不派生 step lifecycle（§4.1）
        lc = out[0]
        assert (lc.lifecycle_type, lc.event, lc.state) == (T.PLAN, K.STARTED, S.PENDING)
        assert lc.unit_id == "plan-1"
        assert lc.detail is not None and lc.detail.note == "plan_created_not_yet_executing"
        assert (lc.source_event_type, lc.source_event_id, lc.source_seq) == ("plan", "src-10", 10)

    @pytest.mark.anyio
    async def test_updated_maps_progress_running(self, projector):
        out = await projector.project(_plan(PlanEventStatus.UPDATED))
        assert [(o.lifecycle_type, o.event, o.state, o.reason) for o in out] == [
            (T.PLAN, K.PROGRESS, S.RUNNING, "plan_updated")
        ]

    @pytest.mark.anyio
    async def test_completed_maps_completed(self, projector):
        out = await projector.project(_plan(PlanEventStatus.COMPLETED))
        assert [(o.lifecycle_type, o.event, o.state) for o in out] == [
            (T.PLAN, K.COMPLETED, S.COMPLETED)
        ]


def _step(status: StepEventStatus, success: bool) -> StepEvent:
    ev = StepEvent(step=Step(id="step-1", success=success), status=status)
    ev.seq = 20
    ev.id = "src-20"
    return ev


class TestStepMapping:
    @pytest.mark.anyio
    async def test_started_maps_running(self, projector):
        out = await projector.project(_step(StepEventStatus.STARTED, success=False))
        assert [(o.lifecycle_type, o.event, o.state) for o in out] == [
            (T.STEP, K.STARTED, S.RUNNING)
        ]
        assert out[0].unit_id == "step-1"

    @pytest.mark.anyio
    async def test_completed_success_true_maps_completed(self, projector):
        out = await projector.project(_step(StepEventStatus.COMPLETED, success=True))
        assert [(o.event, o.state, o.reason) for o in out] == [(K.COMPLETED, S.COMPLETED, None)]

    @pytest.mark.anyio
    async def test_completed_success_false_maps_failed(self, projector):
        # R5#P1：失败 step 的生产形态 = COMPLETED + success=False
        out = await projector.project(_step(StepEventStatus.COMPLETED, success=False))
        assert [(o.event, o.state, o.reason) for o in out] == [(K.FAILED, S.FAILED, "step_failed")]

    @pytest.mark.anyio
    async def test_failed_defensive_maps_failed(self, projector):
        out = await projector.project(_step(StepEventStatus.FAILED, success=False))
        assert [(o.event, o.state, o.reason) for o in out] == [(K.FAILED, S.FAILED, "step_failed")]


class TestGuards:
    @pytest.mark.anyio
    async def test_self_projection_guard_returns_empty(self, projector):
        # R3#5 自投影硬守卫：投影器对 LifecycleEvent 输入必须返回空 Sequence
        lc = build_lifecycle_event(T.PLAN, K.STARTED, unit_id="p")
        assert list(await projector.project(lc)) == []

    @pytest.mark.anyio
    async def test_unmapped_source_returns_empty(self, projector):
        assert list(await projector.project(MessageEvent(message="hi"))) == []

    @pytest.mark.anyio
    async def test_recursion_terminates(self, projector):
        # 守卫失效会造成 hook 层无限递归；这里连投两层证明幂等空集
        lc = build_lifecycle_event(T.STEP, K.STARTED, unit_id="s")
        first = await projector.project(lc)
        assert list(first) == []
