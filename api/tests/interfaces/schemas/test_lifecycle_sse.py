"""C7 PR1 — LifecycleSSEEvent 点分 wire 名 + EventMapper 显式注册（spec §3.2）。"""
import json

import pytest

from app.domain.models.lifecycle import (
    SUPPORTED_EVENTS,
    LifecycleEventKind as K,
    LifecycleType as T,
)
from app.domain.services.lifecycle_emit import build_lifecycle_event
from app.interfaces.schemas.event import EventMapper, LifecycleSSEEvent

SUPPORTED_PAIRS = [(t, k) for t, ks in SUPPORTED_EVENTS.items() for k in ks]


@pytest.fixture(autouse=True)
def _reset_mapper_cache():
    # 类级缓存在测试间隔离，避免其他测试文件先建表的顺序耦合
    EventMapper._cache_mapping = None
    yield
    EventMapper._cache_mapping = None


class TestDottedWireName:
    @pytest.mark.parametrize("pair", SUPPORTED_PAIRS, ids=lambda p: f"{p[0].value}-{p[1].value}")
    def test_dotted_name_three_segments_match_payload(self, pair):
        t, k = pair
        ev = build_lifecycle_event(t, k, unit_id="u1", epoch=(1 if (t, k) == (T.TASK, K.RETRIED) else 0))
        sse = EventMapper.event_to_sse_event(ev)
        assert isinstance(sse, LifecycleSSEEvent)
        # 分流契约（R1#9）：点分名三段与 payload 判别字段逐一相等
        assert sse.event == f"lifecycle.{t.value}.{k.value}"
        data = json.loads(sse.to_sse_data_json())
        assert data["type"] == "lifecycle"          # domain 判别符不复用 wire 名字段（R2#5 分职）
        assert data["lifecycle_type"] == t.value
        assert data["event"] == k.value
        assert data["unit_id"] == "u1"

    def test_not_common_sse_fallback(self):
        # F6/F22 教训：绝不允许 lifecycle 走 CommonSSEEvent 降级
        from app.interfaces.schemas.event import CommonSSEEvent
        ev = build_lifecycle_event(T.STEP, K.COMPLETED, unit_id="s1")
        sse = EventMapper.event_to_sse_event(ev)
        assert not isinstance(sse, CommonSSEEvent)

    def test_mapping_registered_unconditionally(self):
        mapping = EventMapper._get_event_type_mapping()
        assert "lifecycle" in mapping
        assert mapping["lifecycle"].sse_event_class is LifecycleSSEEvent

    def test_data_carries_seq_and_epoch(self):
        ev = build_lifecycle_event(T.TASK, K.RETRIED, unit_id="s1", epoch=2, reason="retry_from_suspend")
        ev.seq = 99
        sse = EventMapper.event_to_sse_event(ev)
        data = json.loads(sse.to_sse_data_json())
        assert data["seq"] == 99
        assert data["epoch"] == 2
        assert data["reason"] == "retry_from_suspend"
