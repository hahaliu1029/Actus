"""N2 §6.4 显式风险锁: reader-drop, writer-continues gap pullback.

关键场景: reader 断开时 PG 只持久化了前 3 条 (session.events 限于已落库部分),
writer 继续写 5 条到 Redis output stream. get_events_since 必须通过 Redis
补偿路径把那 5 条 gap 拉回来 —— PG 单独不够.

此测试的 fixture 故意让 PG 只有 3 条, Redis 只有 5 条 gap, 迫使 Redis 路径
必须真实执行; 并用 spy 断言 recovery 确实被调用, 其输出真实进入合并.

codex plan-review v2 P1 修正: 避免 "PG 全量覆盖 Redis 路径" 的 fixture 退化.

Spec: docs/superpowers/specs/2026-04-17-n2-sse-transport-repair-design.md §6.4 + §7
"""

from __future__ import annotations

import pytest
from app.application.services.agent_service import AgentService
from app.domain.external.event_recovery import EventRecoveryPort, EventRecoveryResult
from app.domain.models.event import BaseEvent, MessageEvent
from app.domain.models.session import Session, SessionStatus

from tests.app.application.services.conftest import default_snapshot as _default_snapshot

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_event(sid: str, text: str) -> BaseEvent:
    e = MessageEvent(role="assistant", message=text)
    e.id = sid
    return e


class _SpyEventRecovery(EventRecoveryPort):
    """返回预置的 gap events, 并记录调用参数以断言路径被真实走到."""

    def __init__(self, gap_events: list[BaseEvent]) -> None:
        self._gap = gap_events
        self.call_count = 0
        self.last_after_event_id: str | None = "__unset__"

    async def get_recent_events(
        self, task_id: str, after_event_id: str | None
    ) -> EventRecoveryResult:
        self.call_count += 1
        self.last_after_event_id = after_event_id
        # 任何非 None after_event_id 都返回全部 gap (严格模拟 Redis XRANGE
        # after-exclusive: 传入的 id 本来就不在 gap 里, 不需要过滤)
        return EventRecoveryResult(events=list(self._gap), has_more=False)


class _SessionRepoWithSession:
    def __init__(self, session: Session) -> None:
        self._session = session

    async def get_by_id(self, session_id: str) -> Session:
        return self._session


class _NoopUoWReturning:
    def __init__(self, session: Session) -> None:
        self.session = _SessionRepoWithSession(session)

    async def __aenter__(self): return self
    async def __aexit__(self, *args): return None


async def test_reader_drop_writer_continues_gap_pullback_requires_redis_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reader 读 3 帧关闭 → PG 只有 3 帧 → writer 继续写 5 帧到 Redis →
    GET since='1000-2' 必须走 Redis 补偿路径把 5 gap 拉回来.
    """
    # PG = 前 3 条 read 事件 (reader drop 时这些已持久化)
    pg_events = [_make_event(f"1000-{i}", f"read-{i}") for i in range(3)]
    # Redis gap = 后 5 条 writer continues 的事件 (仅在 Redis, 未落 PG)
    gap_events = [_make_event(f"1000-{i}", f"gap-{i - 3}") for i in range(3, 8)]

    session_with_events = Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.RUNNING,
        task_id="task-1",  # 触发 Redis 补偿路径 (agent_service.py:556)
        events=pg_events,   # 只有 PG 的前 3 条
    )

    spy_recovery = _SpyEventRecovery(gap_events)

    service = AgentService(
        uow_factory=lambda: _NoopUoWReturning(session_with_events),
        config_snapshot=_default_snapshot(),
        sandbox_cls=object,
        task_cls=None,
        search_engine=object(),
        file_storage=object(),
        event_recovery=spy_recovery,
    )

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return session_with_events

    monkeypatch.setattr(
        service, "_get_accessible_session", fake_get_accessible_session
    )

    # 客户端最后看到 "1000-2" (PG 里的最后一条 read event)
    result = await service.get_events_since(
        session_id="session-1",
        since_event_id="1000-2",
        user_id="user-1",
    )

    # PG 路径切片: since=1000-2 在 pg_events 位于 idx=2, 切片后 pg_events=[]
    # Redis 补偿路径: 因 pg_events 空, redis_start_id 回落到 since_event_id="1000-2"
    # Recovery 返回 gap 5 条 → filter (pg 空) → redis_only_events = gap 5 条
    # 最终 result.events = pg_events([]) + redis_only_events(5) = 5 条 gap

    returned_ids = {e.id for e in result["events"]}
    expected_gap_ids = {f"1000-{i}" for i in range(3, 8)}

    # 1. 严格集合相等 (gap 完整, 无缺失, 无多余)
    assert returned_ids == expected_gap_ids, (
        f"gap pullback mismatch: got {sorted(returned_ids)}, "
        f"expected {sorted(expected_gap_ids)}"
    )

    # 2. Redis 补偿路径确实被调用 (证明 fixture 没有让 PG 误 mask Redis)
    assert spy_recovery.call_count == 1, (
        f"expected Redis recovery to be called exactly once, "
        f"got {spy_recovery.call_count}"
    )

    # 3. Redis 起点是 since_event_id (因 PG 切片后为空, 走 fallback 分支)
    assert spy_recovery.last_after_event_id == "1000-2", (
        f"expected after_event_id='1000-2' (fallback from empty PG slice), "
        f"got {spy_recovery.last_after_event_id!r}"
    )
