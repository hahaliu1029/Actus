"""C7 PR2 — 咽喉 hook：顺序/seq/persist 跟随/flag-off 零进入/递归防护（spec §6/INV-C7-3/6）。"""
import json
from unittest.mock import AsyncMock, patch

import pytest

from app.domain.models.app_config import LifecycleRuntimeConfig
from app.domain.models.event import PlanEvent, PlanEventStatus
from app.domain.models.lifecycle import LifecycleEventKind as K, LifecycleType as T
from app.domain.services.agent_task_runner import AgentTaskRunner
from app.domain.services.lifecycle_emit import build_lifecycle_event
from app.domain.services.lifecycle_projector import LifecycleProjector
from app.domain.models.plan import Plan


class _FakeRedis:
    def __init__(self) -> None:
        self._n = 0

    async def incr(self, key: str) -> int:
        self._n += 1
        return self._n

    async def expire(self, key: str, ttl: int) -> None:
        return None


class _FakeSessionRepo:
    def __init__(self) -> None:
        self.persisted: list = []

    async def add_event(self, session_id: str, event) -> None:
        self.persisted.append(event)


class _FakeUoW:
    def __init__(self) -> None:
        self.session = _FakeSessionRepo()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _OutputStream:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def put(self, event_json: str) -> str:
        # 忠实建模生产端 consumer 的 id 回填（agent_service.py:2392 / redis
        # recovery 同模式）：交付给 client 的每条事件 id 被覆盖成 Redis Stream
        # message-id。存储侧同构回填后，source 的可观测 id 才等于 lifecycle
        # 的 source_event_id，配对审计（INV-C7-6）方能在流上成立。
        stream_id = f"stream-{len(self.events) + 1}"
        payload = json.loads(event_json)
        payload["id"] = stream_id
        self.events.append(json.dumps(payload))
        return stream_id


class _Task:
    def __init__(self) -> None:
        self.output_stream = _OutputStream()


def _make_runner(flag_on: bool) -> AgentTaskRunner:
    runner = AgentTaskRunner.__new__(AgentTaskRunner)  # 跳过 20+ 参 ctor（既有测试惯例）
    runner._session_id = "sess-1"
    runner._uow = _FakeUoW()
    runner._event_seq_client = _FakeRedis()
    runner._event_seq_ttl_seconds = 3600
    runner._lifecycle_runtime = LifecycleRuntimeConfig(lifecycle_events_enabled=flag_on)
    runner._lifecycle_projector = LifecycleProjector(parent_session_id="sess-1")
    runner._lifecycle_terminal_emitted = set()
    return runner


def _plan_event() -> PlanEvent:
    return PlanEvent(plan=Plan(id="plan-1"), status=PlanEventStatus.CREATED)


@pytest.mark.anyio
async def test_flag_on_pairs_source_then_lifecycle_with_greater_seq():
    runner, task = _make_runner(True), _Task()
    await runner._put_and_add_event(task, _plan_event())
    assert len(task.output_stream.events) == 2
    src = json.loads(task.output_stream.events[0])
    lc = json.loads(task.output_stream.events[1])
    assert src["type"] == "plan" and lc["type"] == "lifecycle"
    assert lc["seq"] > src["seq"]                      # INV-C7-6: 同 seq 空间且 lifecycle.seq > source.seq
    assert lc["source_seq"] == src["seq"]
    assert lc["source_event_id"] == "stream-1"          # XADD 覆盖后的最终 id（R10#A5）
    assert lc["lifecycle_type"] == "plan" and lc["event"] == "started"


@pytest.mark.anyio
async def test_persist_follows_source_true():
    runner, task = _make_runner(True), _Task()
    await runner._put_and_add_event(task, _plan_event(), persist=True)
    persisted_types = [e.type for e in runner._uow.session.persisted]
    assert persisted_types == ["plan", "lifecycle"]     # R3#8: lifecycle persist 跟随 source


@pytest.mark.anyio
async def test_persist_follows_source_false():
    runner, task = _make_runner(True), _Task()
    await runner._put_and_add_event(task, _plan_event(), persist=False)
    assert runner._uow.session.persisted == []          # 两者都不落库
    assert len(task.output_stream.events) == 2          # 但都进流


@pytest.mark.anyio
async def test_flag_off_zero_entry_zero_construction():
    # INV-C7-3：flag-off 下 hook 零进入——projector.project 与 build_lifecycle_event 零调用
    runner, task = _make_runner(False), _Task()
    with patch.object(LifecycleProjector, "project", new=AsyncMock()) as spy_project, \
         patch("app.domain.services.lifecycle_projector.build_lifecycle_event") as spy_build:
        await runner._put_and_add_event(task, _plan_event())
    spy_project.assert_not_awaited()
    spy_build.assert_not_called()
    assert len(task.output_stream.events) == 1
    assert json.loads(task.output_stream.events[0])["type"] == "plan"


@pytest.mark.anyio
async def test_lifecycle_event_does_not_reproject():
    # 递归防护双保险：emit 路径传 project_lifecycle=False；且投影器自投影守卫兜底
    runner, task = _make_runner(True), _Task()
    lc = build_lifecycle_event(T.PLAN, K.STARTED, unit_id="p")
    await runner._put_and_add_event(task, lc)           # 默认 project_lifecycle=True 也不得再投
    assert len(task.output_stream.events) == 1


@pytest.mark.anyio
async def test_projection_failure_never_breaks_source_emit():
    runner, task = _make_runner(True), _Task()
    with patch.object(LifecycleProjector, "project", new=AsyncMock(side_effect=RuntimeError("boom"))):
        await runner._put_and_add_event(task, _plan_event())
    assert len(task.output_stream.events) == 1          # source 照常入流，投影失败仅告警


@pytest.mark.anyio
async def test_legacy_runner_without_lifecycle_attrs_is_noop():
    # 既有测试用 __new__ 构造且不设 C7 属性——hook 必须 getattr 防御性 no-op
    runner, task = _make_runner(True), _Task()
    del runner._lifecycle_runtime
    await runner._put_and_add_event(task, _plan_event())
    assert len(task.output_stream.events) == 1
