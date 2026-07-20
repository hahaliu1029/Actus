"""C7 PR7 — §12-3/F7：含 lifecycle 的流断线重放无丢失无异常；未知成员单条降级。"""
import json

import pytest

import app.infrastructure.external.event_recovery.redis_event_recovery as recovery_mod
from app.domain.models.event import LifecycleEvent
from app.domain.models.lifecycle import LifecycleEventKind as K, LifecycleType as T
from app.domain.services.lifecycle_emit import build_lifecycle_event


class _FakeQueue:
    entries: list = []

    def __init__(self, stream_name: str) -> None:
        self._name = stream_name

    async def get_range(self, start_id: str = "-", count: int = 10000):
        for message_id, payload in self.entries:
            yield message_id, payload


@pytest.mark.anyio
async def test_mixed_stream_replay_keeps_lifecycle_and_skips_garbage(monkeypatch):
    lc = build_lifecycle_event(T.STEP, K.COMPLETED, unit_id="s1")
    lc.seq = 12
    _FakeQueue.entries = [
        ("1-1", '{"type": "done", "seq": 10}'),
        ("1-2", lc.model_dump_json()),
        ("1-3", '{"type": "totally_unknown_member", "seq": 13}'),   # 老 pod 视角的新事件≈此形态
        ("1-4", '{"type": "message", "role": "assistant", "message": "m", "seq": 14}'),
    ]
    monkeypatch.setattr(recovery_mod, "RedisStreamMessageQueue", _FakeQueue)

    result = await recovery_mod.RedisEventRecovery().get_recent_events("task-1", None)
    types = [e.type for e in result.events]
    assert types == ["done", "lifecycle", "message"]                 # 未知单条 catch-continue（F7）
    restored = result.events[1]
    assert isinstance(restored, LifecycleEvent)
    assert restored.id == "1-2"                                      # Stream ID 回填
    assert restored.seq == 12


@pytest.mark.anyio
async def test_seq_cursor_filters_lifecycle_like_any_event(monkeypatch):
    lc_old = build_lifecycle_event(T.TOOL, K.STARTED, unit_id="tc1"); lc_old.seq = 5
    lc_new = build_lifecycle_event(T.TOOL, K.COMPLETED, unit_id="tc1"); lc_new.seq = 9
    _FakeQueue.entries = [("2-1", lc_old.model_dump_json()), ("2-2", lc_new.model_dump_json())]
    monkeypatch.setattr(recovery_mod, "RedisStreamMessageQueue", _FakeQueue)

    result = await recovery_mod.RedisEventRecovery().get_recent_events("task-1", None, after_seq=5)
    assert [e.seq for e in result.events] == [9]                     # last_seq 水位无缝（F5/F21）
