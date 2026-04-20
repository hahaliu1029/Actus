"""Pre-existing bug anchor: UUID cursor zeroing → full output replay.

Spec §6.3.b: chat() 异常 fallback 产生 UUID-id ErrorEvent, add_event 落 PG
但不改 session.status. 冷启动 fetchSessionById 采用服务端 status="running"
自动 sendChat(..., {event_id: UUID}). agent_service.chat() L634-638 判
_is_valid_redis_stream_id(UUID) == False → latest_event_id=None →
output_stream.get(start_id=None) 从头重读 → 整段 output 重放.

此测试用 REAL assert 锚定 bug:
- 当前 (bug): output_stream.get 被调用时 start_id is None → 重放所有事件 → 断言失败 → XFAIL
- Phase 2 修复后: 传 UUID 应该返回 0 events 或 error (具体策略见 spec §6.3.b) → PASS → strict xfail 报错

Spec: docs/superpowers/specs/2026-04-17-n2-sse-transport-repair-design.md §6.3.b
"""

from __future__ import annotations

import asyncio

import pytest
from app.application.services.agent_service import AgentService
from app.domain.models.session import Session, SessionStatus

from tests.app.application.services.conftest import default_snapshot as _default_snapshot

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---- Dummy task / stream (同 Task 6, 简版) ----


class _NoopSessionRepository:
    async def update_unread_message_count(self, session_id, count): pass
    async def update_latest_message(self, session_id, message, timestamp): pass
    async def add_event(self, session_id, event): pass


class _NoopUoW:
    def __init__(self): self.session = _NoopSessionRepository()
    async def __aenter__(self): return self
    async def __aexit__(self, *args): return None


def _uow_factory(): return _NoopUoW()


class _DummyInputStream:
    async def put(self, event_json): return "evt-user-1"


class _QueuedOutputStream:
    def __init__(self, owner, events):
        self._owner = owner
        self._events = list(events)
        self._cursor = 0
        self.start_id_calls = []

    async def get(self, start_id=None, block_ms=None):
        self.start_id_calls.append(start_id)
        while self._cursor < len(self._events):
            sid, payload = self._events[self._cursor]
            self._cursor += 1
            if start_id is not None and sid <= start_id:
                continue
            return sid, payload
        self._owner.done_flag = True
        return None, None


class _DummyTask:
    def __init__(self, output_events):
        self.done_flag = False
        self.output_stream = _QueuedOutputStream(self, output_events)
        self.input_stream = _DummyInputStream()

    @property
    def done(self): return self.done_flag

    async def invoke(self): return None


class _DummyTaskClass:
    @classmethod
    def get(cls, task_id): return None
    @classmethod
    def create(cls, task_runner): return _DummyTask([])
    @classmethod
    async def destroy(cls): return None


def _build_service():
    return AgentService(
        uow_factory=_uow_factory,
        config_snapshot=_default_snapshot(),
        sandbox_cls=object,
        task_cls=_DummyTaskClass,
        search_engine=object(),
        file_storage=object(),
    )


def _patch(service, task, monkeypatch):
    async def fake_get_accessible_session(*args, **kwargs):
        return Session(id="session-1", user_id="user-1", status=SessionStatus.RUNNING)
    async def fake_check_attachments_access(*a, **k): pass
    async def fake_get_task(_s): return task
    async def fake_safe_update_unread_count(_s): pass
    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_check_attachments_access", fake_check_attachments_access)
    monkeypatch.setattr(service, "_get_task", fake_get_task)
    monkeypatch.setattr(service, "_safe_update_unread_count", fake_safe_update_unread_count)


# ---- xfail test ----


@pytest.mark.xfail(
    reason=(
        "pre-existing UUID cursor zeroing replay (spec §6.3.b): passing a UUID as "
        "latest_event_id is treated as invalid, cursor is zeroed, full output stream "
        "replays from head. N2 scope 外不修. 此 anchor 覆盖的修法面: chat() 返回空 / "
        "返回 error event / raise 异常 (三者都让 received_output_ids 长度 < 3). "
        "若 Phase 2 采取'让 session.status 进入终态, 前端不再自动 reconnect'这类 "
        "frontend-only 修法, backend 仍会 full replay, 此 anchor 会保持 XFAIL — "
        "这是已知局限, 实施者需另起一个 frontend-side anchor"
    ),
    strict=True,
)
async def test_uuid_event_id_as_cursor_does_not_cause_full_replay(monkeypatch) -> None:
    """传 UUID latest_event_id, 断言 output stream 不从头重读 (fixed behavior).

    Collection loop 把 chat() 抛出的任意异常吞成 "0 events received" (走 `< 3` 分支),
    这样 "raise" 修法也会翻 XPASS 触发 strict xfail flag.
    """
    output_events = [
        ("1000-0", '{"event_id": "1000-0", "type": "message"}'),
        ("1000-1", '{"event_id": "1000-1", "type": "message"}'),
        ("1000-2", '{"event_id": "1000-2", "type": "message"}'),
    ]
    task = _DummyTask(output_events=output_events)

    service = _build_service()
    _patch(service, task, monkeypatch)

    # 客户端带着 UUID ErrorEvent id 回来
    chat_gen = service.chat(
        session_id="session-1",
        user_id="user-1",
        message=None,
        attachments=None,
        latest_event_id="550e8400-e29b-41d4-a716-446655440000",
        timestamp=None,
    )

    received_output_ids: list[str] = []
    try:
        while True:
            evt = await asyncio.wait_for(chat_gen.__anext__(), timeout=0.5)
            received_output_ids.append(evt.id)
    except StopAsyncIteration:
        pass
    except Exception:
        # Phase 2 修法若为 "raise" (让 chat() 对 UUID 直接抛异常), 吞掉让 assertion
        # 阶段决定 XFAIL/XPASS — 否则未捕获异常会被 strict xfail 计作 XFAIL 而非 XPASS,
        # 导致此 anchor 在 raise 修法下永远不会翻.
        pass

    # FIXED BEHAVIOR: UUID 应被视为 stale session, 不触发全量 replay
    # (具体策略 Phase 2 决定: 返回空, 或返回 error, 或抛异常)
    # 本断言保守选"不全量重放" — 只要不返回所有 3 条事件即算修复
    # 当前 BUG: UUID → start_id=None → output_stream.get 从头返回所有 3 条 → XFAIL
    # 修复后 (任一策略): received_output_ids 长度 < 3 → PASS → strict xfail 报错
    assert len(received_output_ids) < 3, (
        f"UUID cursor zeroing caused full replay: got all {len(received_output_ids)} "
        f"events {received_output_ids}"
    )
