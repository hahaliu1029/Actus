"""N2 §7 回归锁: page-reopen 路径通过 chat() latest_event_id cursor 续读 output stream.

模拟 Web session-store.ts:793-803 刷新进 running session: 取 latestEventId
-> sendChat(sessionId, {event_id}) -> POST /api/sessions/{id}/chat.

用 output-stream id 避开 §6.3.a race (由 test_chat_post_event_id_input_stream_skip_race.py 锚定).

Spec: docs/superpowers/specs/2026-04-17-n2-sse-transport-repair-design.md §5.3 + §7
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


# ---- Dummy Task / Output Stream (Boilerplate B) ----
# (内联此处以保持单文件可读; 若仓库后续抽到 shared conftest 可 DRY 化)


class _NoopSessionRepository:
    def __init__(self) -> None:
        self.add_event_calls: list[tuple[str, object]] = []

    async def update_unread_message_count(self, session_id: str, count: int) -> None:
        return None

    async def update_latest_message(self, session_id: str, message: str, timestamp) -> None:
        return None

    async def add_event(self, session_id: str, event) -> None:
        self.add_event_calls.append((session_id, event))


class _NoopUoW:
    def __init__(self) -> None:
        self.session = _NoopSessionRepository()

    async def __aenter__(self) -> "_NoopUoW":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return None


def _uow_factory() -> _NoopUoW:
    return _NoopUoW()


class _DummyInputStream:
    def __init__(self, return_id: str = "evt-user-1") -> None:
        self._return_id = return_id
        self.events: list[str] = []

    async def put(self, event_json: str) -> str:
        self.events.append(event_json)
        return self._return_id


class _QueuedOutputStream:
    def __init__(self, owner: "_DummyTask", events: list[tuple[str, str]]) -> None:
        self._owner = owner
        self._events = list(events)
        self._cursor: int = 0
        self.start_id_calls: list[Optional[str]] = []
        self.block_ms_calls: list[Optional[int]] = []

    async def get(self, start_id: str = None, block_ms: int = None):
        self.start_id_calls.append(start_id)
        self.block_ms_calls.append(block_ms)
        while self._cursor < len(self._events):
            sid, payload = self._events[self._cursor]
            self._cursor += 1
            if start_id is not None and sid <= start_id:
                continue
            return sid, payload
        self._owner.done_flag = True
        return None, None


class _DummyTask:
    def __init__(
        self,
        output_events: list[tuple[str, str]] | None = None,
        input_return_id: str = "evt-user-1",
    ) -> None:
        self.done_flag = False
        self.output_stream = _QueuedOutputStream(self, output_events or [])
        self.input_stream = _DummyInputStream(input_return_id)

    @property
    def done(self) -> bool:
        return self.done_flag

    async def invoke(self) -> None:
        return None


class _DummyTaskClass:
    @classmethod
    def get(cls, task_id: str):
        return None

    @classmethod
    def create(cls, task_runner):
        return _DummyTask()

    @classmethod
    async def destroy(cls) -> None:
        return None


def _build_service() -> AgentService:
    return AgentService(
        uow_factory=_uow_factory,
        config_snapshot=_default_snapshot(),
        sandbox_cls=object,
        task_cls=_DummyTaskClass,
        search_engine=object(),
        file_storage=object(),
    )


def _patch(service: AgentService, task: _DummyTask, monkeypatch) -> None:
    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return Session(id="session-1", user_id="user-1", status=SessionStatus.RUNNING)

    async def fake_check_attachments_access(*args, **kwargs) -> None:
        return None

    async def fake_get_task(_session: Session):
        return task

    async def fake_safe_update_unread_count(_session_id: str) -> None:
        return None

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_check_attachments_access", fake_check_attachments_access)
    monkeypatch.setattr(service, "_get_task", fake_get_task)
    monkeypatch.setattr(service, "_safe_update_unread_count", fake_safe_update_unread_count)


# ---- Actual test ----


async def test_page_reopen_with_output_stream_id_resumes_cursor(monkeypatch) -> None:
    """传入 output-stream id 作为 latest_event_id, chat() 用它做 start_id 续读."""
    # 预设 output stream 有 3 条事件, id 单调
    output_events = [
        ("1000-0", '{"event_id": "1000-0", "type": "message"}'),
        ("1000-1", '{"event_id": "1000-1", "type": "message"}'),
        ("1000-2", '{"event_id": "1000-2", "type": "message"}'),
    ]
    task = _DummyTask(output_events=output_events)

    service = _build_service()
    _patch(service, task, monkeypatch)

    # 以 "1000-0" 为 cursor 续读
    chat_gen = service.chat(
        session_id="session-1",
        user_id="user-1",
        message=None,
        attachments=None,
        latest_event_id="1000-0",
        timestamp=None,
    )

    received_ids: list[str] = []
    try:
        while True:
            evt = await asyncio.wait_for(chat_gen.__anext__(), timeout=0.5)
            received_ids.append(evt.id)
    except StopAsyncIteration:
        pass

    # 断言: 只收到 "1000-1" 和 "1000-2", "1000-0" 被过滤
    assert received_ids == ["1000-1", "1000-2"], (
        f"expected cursor to skip '1000-0', got {received_ids}"
    )
    # 并且 chat() 把 "1000-0" 真实传给了 output_stream.get
    assert task.output_stream.start_id_calls[0] == "1000-0"
