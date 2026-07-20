"""C7 PR5 — subagent 分支：spawned/reduce/sibling-cancel × repo 单次批查（spec §4.5）。"""
from typing import Dict

import pytest

from app.domain.models.event import (
    CoordinatorReduceEvent,
    CoordinatorSiblingCancelEvent,
    CoordinatorWorkerSpawnedEvent,
)
from app.domain.models.lifecycle import (
    LifecycleEventKind as K,
    LifecycleState as S,
    LifecycleType as T,
)
from app.domain.models.mailbox_envelope import CostAggregate, ResultReadyOutcome as RO
from app.domain.models.patch_apply_plan import GroupOutcome
from app.domain.services.lifecycle_projector import LifecycleProjector


class _CountingLookup:
    def __init__(self, mapping: Dict[str, str], *, fail: bool = False) -> None:
        self.mapping = mapping
        self.calls = 0
        self.fail = fail

    async def __call__(self, coordinator_run_id: str) -> Dict[str, str]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("db down")
        return self.mapping


def _projector(lookup, *, subagent_on: bool = True) -> LifecycleProjector:
    return LifecycleProjector(
        parent_session_id="parent-1",
        subagent_enabled=lambda: subagent_on,
        child_lookup=lookup,
    )


def _spawned(child_id="child-a", wu="wu-a") -> CoordinatorWorkerSpawnedEvent:
    return CoordinatorWorkerSpawnedEvent(
        objective="o", phase="exploration", allowed_tools=[], write_lease_count=0,
        child_session_id=child_id, work_unit_id=wu,
        coordinator_run_id="run-1", parent_session_id="parent-1",
    )


def _reduce(outcomes: Dict[str, RO]) -> CoordinatorReduceEvent:
    return CoordinatorReduceEvent(
        group_outcome=GroupOutcome.SUCCESS, per_worker_outcomes=outcomes,
        diagnostics_summary="", cost_total=CostAggregate(),
        coordinator_run_id="run-1", parent_session_id="parent-1",
    )


def _sibling_cancel(cancelled: list[str]) -> CoordinatorSiblingCancelEvent:
    return CoordinatorSiblingCancelEvent(
        triggered_by_work_unit_id="wu-a", triggered_by_outcome=RO.FAILED,
        cancelled_work_unit_ids=cancelled, reason="fail_fast",
        coordinator_run_id="run-1", parent_session_id="parent-1",
    )


MAPPING = {"wu-a": "child-a", "wu-b": "child-b", "wu-c": "child-c"}


class TestSpawned:
    @pytest.mark.anyio
    async def test_spawned_maps_started_without_lookup(self):
        lookup = _CountingLookup(MAPPING)
        out = await _projector(lookup).project(_spawned())
        assert [(o.lifecycle_type, o.event, o.state, o.unit_id) for o in out] == [
            (T.SUBAGENT, K.STARTED, S.RUNNING, "child-a")
        ]
        assert out[0].correlation.work_unit_id == "wu-a"
        assert out[0].correlation.coordinator_run_id == "run-1"
        assert out[0].parent_unit_id == "parent-1"
        assert out[0].epoch == 0                      # INV-C7-8：非 task 恒 0
        assert lookup.calls == 0                       # spawned 自带 child id

    @pytest.mark.anyio
    async def test_spawned_missing_child_id_skips(self):
        ev = _spawned()
        ev.child_session_id = None
        out = await _projector(_CountingLookup(MAPPING)).project(ev)
        assert list(out) == []


class TestReduce:
    @pytest.mark.anyio
    async def test_reduce_expands_per_child_with_single_query(self):
        lookup = _CountingLookup(MAPPING)
        out = await _projector(lookup).project(_reduce({
            "wu-a": RO.SUCCESS, "wu-b": RO.FAILED, "wu-c": RO.TIMED_OUT,
        }))
        assert lookup.calls == 1                       # R4#3：N child ≠ N 次查询
        got = {o.unit_id: (o.event, o.reason) for o in out}
        assert got == {
            "child-a": (K.COMPLETED, None),
            "child-b": (K.FAILED, "worker_failed"),
            "child-c": (K.FAILED, "watchdog_timeout"),
        }

    @pytest.mark.anyio
    async def test_needs_authorization_maps_failed_with_original_outcome(self):
        out = await _projector(_CountingLookup(MAPPING)).project(
            _reduce({"wu-a": RO.NEEDS_AUTHORIZATION})
        )
        assert (out[0].event, out[0].reason) == (K.FAILED, "needs_authorization")
        assert out[0].detail.original_outcome == "needs_authorization"

    @pytest.mark.anyio
    async def test_cancelled_outcome_maps_cancelled(self):
        out = await _projector(_CountingLookup(MAPPING)).project(_reduce({"wu-a": RO.CANCELLED}))
        assert [(o.event, o.state) for o in out] == [(K.CANCELLED, S.CANCELLED)]

    @pytest.mark.anyio
    async def test_unknown_wu_skipped_others_survive(self):
        out = await _projector(_CountingLookup({"wu-a": "child-a"})).project(_reduce({
            "wu-a": RO.SUCCESS, "wu-ghost": RO.FAILED,
        }))
        assert [o.unit_id for o in out] == ["child-a"]  # 合法降级：缺映射跳过+log

    @pytest.mark.anyio
    async def test_lookup_failure_degrades_to_empty(self):
        out = await _projector(_CountingLookup(MAPPING, fail=True)).project(_reduce({"wu-a": RO.SUCCESS}))
        assert list(out) == []                          # 不 raise 不硬造

    @pytest.mark.anyio
    async def test_group_outcome_never_maps_per_child(self):
        # R4#6：CONFLICT/INCOMPLETE/MIXED 是 group 级信号——per-child 展开只由
        # per_worker_outcomes 驱动；INCOMPLETE 下无 outcome 的 child 停 started（§1 局限 1）
        ev = _reduce({})
        ev.group_outcome = GroupOutcome.SUCCESS  # 任意 group 值都不产 per-child 事件
        out = await _projector(_CountingLookup(MAPPING)).project(ev)
        assert list(out) == []


class TestSiblingCancel:
    @pytest.mark.anyio
    async def test_sibling_cancel_expands_via_lookup(self):
        lookup = _CountingLookup(MAPPING)
        out = await _projector(lookup).project(_sibling_cancel(["wu-b", "wu-c"]))
        assert lookup.calls == 1
        assert {o.unit_id for o in out} == {"child-b", "child-c"}
        assert all((o.event, o.reason) == (K.CANCELLED, "sibling_cancel") for o in out)


class TestAndGate:
    @pytest.mark.anyio
    async def test_subagent_flag_off_projects_nothing(self):
        # R10#A9 运行期 AND：master 在 hook 层查，subagent 在分支入口查
        lookup = _CountingLookup(MAPPING)
        p = _projector(lookup, subagent_on=False)
        assert list(await p.project(_spawned())) == []
        assert list(await p.project(_reduce({"wu-a": RO.SUCCESS}))) == []
        assert list(await p.project(_sibling_cancel(["wu-b"]))) == []
        assert lookup.calls == 0                        # off 时零 repo 调用（INV-C7-3 扩展）
