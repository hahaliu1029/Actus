"""Pre-existing bug anchor: cross-stream cursor race on page-reopen.

Spec §6.3.a: 若 last-seen event_id 来自 input stream, 与 output stream 第一帧同 ms,
agent_service.chat() L634-638 只验格式 不区分 stream key 空间,
直接喂给 output_stream.get(start_id=...), 导致 output 首帧被跳过.

此测试用 REAL assert (不是 NotImplementedError) 锚定 bug:
- 当前 (bug): 断言失败, 被 xfail strict 收入为 XFAIL
- Phase 2 修复后: 断言通过, xfail strict 会报错, 提示摘 marker

Spec: docs/superpowers/specs/2026-04-17-n2-sse-transport-repair-design.md §6.3.a
"""

from __future__ import annotations

import asyncio
from typing import Optional

import pytest
from app.application.services.agent_service import AgentService
from app.domain.models.session import Session, SessionStatus

from tests.app.application.services.conftest import default_snapshot as _default_snapshot

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---- 复用 Task 5 的 dummy task; 实施时可抽 shared conftest ----
# (为保持每个测试文件可独立阅读, 这里短版复制 _DummyTask / _QueuedOutputStream / _patch)


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
    def __init__(self, return_id): self._r = return_id
    async def put(self, event_json): return self._r


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
    def __init__(self, output_events, input_return_id):
        self.done_flag = False
        self.output_stream = _QueuedOutputStream(self, output_events)
        self.input_stream = _DummyInputStream(input_return_id)

    @property
    def done(self): return self.done_flag

    async def invoke(self): return None


class _DummyTaskClass:
    @classmethod
    def get(cls, task_id): return None

    @classmethod
    def create(cls, task_runner): return _DummyTask([], "evt-user-1")

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


# ---- xfail 测试 ----


@pytest.mark.xfail(
    reason=(
        "pre-existing cross-stream cursor race (spec §6.3.a): input-stream id "
        "used as output_stream cursor can skip the first output frame on same-ms "
        "collision. N2 scope 外不修. 此 anchor 只覆盖 **backend-side 修法** — "
        "即 agent_service.chat() 不再把 input-stream id 喂给 output_stream.get() "
        "(例如 _is_valid_redis_stream_id 之上加 stream key 空间判断, 或分离 id 空间). "
        "若 Phase 2 采取 **frontend-side 修法** (getLatestEventId 只取 output-stream "
        "事件, input id 从不回传), backend 行为不变, 此 anchor 会保持 XFAIL — "
        "这是已知局限, 实施 frontend 修法时需另加一个 frontend-side anchor"
    ),
    strict=True,
)
async def test_input_stream_id_as_cursor_skips_output_first_frame(monkeypatch) -> None:
    """构造 input 与 output 首帧同 id 场景, 断言 output 首帧能被返回 (fixed behavior)."""
    # 核心构造: input stream 返回 "1000-0", output stream 第一帧也是 "1000-0"
    # 这模拟 "user 发 message 和 agent 第一条 output 在同一 ms" 的 race 场景
    output_events = [
        ("1000-0", '{"event_id": "1000-0", "type": "message"}'),
        ("1000-1", '{"event_id": "1000-1", "type": "message"}'),
    ]
    task = _DummyTask(output_events=output_events, input_return_id="1000-0")

    service = _build_service()
    _patch(service, task, monkeypatch)

    # 用户带着 "1000-0" (input-stream id) 回来续流
    chat_gen = service.chat(
        session_id="session-1",
        user_id="user-1",
        message=None,
        attachments=None,
        latest_event_id="1000-0",
        timestamp=None,
    )

    received_output_ids: list[str] = []
    try:
        while True:
            evt = await asyncio.wait_for(chat_gen.__anext__(), timeout=0.5)
            received_output_ids.append(evt.id)
    except StopAsyncIteration:
        pass

    # FIXED BEHAVIOR: output stream 应该全部返回 (不被 cross-stream cursor 跳过)
    # 当前 BUG: "1000-0" output 首帧被跳过 (因 start_id="1000-0" <= "1000-0")
    # → 当前: received_output_ids == ["1000-1"] → 此 assert 失败 → XFAIL
    # → 修复后: received_output_ids == ["1000-0", "1000-1"] → PASS → strict XFAIL 报错
    assert received_output_ids == ["1000-0", "1000-1"], (
        f"cross-stream cursor race: expected both output frames, got {received_output_ids}"
    )
