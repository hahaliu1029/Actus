"""C7 PR1 — LifecycleEvent schema / Event 联合 additive / helper 构造校验 / Redis round-trip。"""
import pytest
from pydantic import TypeAdapter, ValidationError

from app.domain.models.event import Event, LifecycleEvent, PlanEvent
from app.domain.models.lifecycle import (
    LifecycleContractError,
    LifecycleCorrelationV1,
    LifecycleDetailV1,
    LifecycleEventKind as K,
    LifecycleState as S,
    LifecycleType as T,
    STATE_FOR,
    SUPPORTED_EVENTS,
)
from app.domain.models.plan import Plan
from app.domain.services.lifecycle_emit import build_lifecycle_event

EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(Event)

ALL_PAIRS = [(t, k) for t in T for k in K]
SUPPORTED_PAIRS = [(t, k) for t, ks in SUPPORTED_EVENTS.items() for k in ks]
UNSUPPORTED_PAIRS = [p for p in ALL_PAIRS if p[0] not in SUPPORTED_EVENTS or p[1] not in SUPPORTED_EVENTS[p[0]]]


class TestBuildHelper:
    @pytest.mark.parametrize("pair", SUPPORTED_PAIRS, ids=lambda p: f"{p[0].value}-{p[1].value}")
    def test_every_supported_pair_builds_with_derived_state(self, pair):
        t, k = pair
        ev = build_lifecycle_event(t, k, unit_id="u1", epoch=(1 if (t, k) == (T.TASK, K.RETRIED) else 0))
        assert ev.type == "lifecycle"
        assert ev.state == STATE_FOR[(t, k)]  # R10#A4: state 由派生表决定
        assert ev.unit_id == "u1"

    @pytest.mark.parametrize("pair", UNSUPPORTED_PAIRS, ids=lambda p: f"{p[0].value}-{p[1].value}")
    def test_every_unsupported_pair_raises(self, pair):
        t, k = pair
        with pytest.raises(LifecycleContractError):
            build_lifecycle_event(t, k, unit_id="u1")

    def test_unsupported_pair_count_is_nine(self):
        assert len(UNSUPPORTED_PAIRS) == 9  # 30 total - 21 supported

    # INV-C7-8 四象限（R10#B1：正负偏差同判；raise 非 assert）
    def test_epoch_task_zero_ok(self):
        assert build_lifecycle_event(T.TASK, K.STARTED, unit_id="u", epoch=0).epoch == 0

    def test_epoch_task_positive_ok(self):
        assert build_lifecycle_event(T.TASK, K.RETRIED, unit_id="u", epoch=2).epoch == 2

    def test_epoch_nontask_positive_raises(self):
        with pytest.raises(LifecycleContractError):
            build_lifecycle_event(T.TOOL, K.STARTED, unit_id="u", epoch=1)

    def test_epoch_negative_raises(self):
        with pytest.raises(LifecycleContractError):
            build_lifecycle_event(T.TASK, K.STARTED, unit_id="u", epoch=-1)
        with pytest.raises(LifecycleContractError):
            build_lifecycle_event(T.TOOL, K.STARTED, unit_id="u", epoch=-1)

    def test_schema_field_ge_zero_backstop(self):
        # schema 层 Field(ge=0) 兜底（绕过 helper 的反序列化路径也拒负值）
        with pytest.raises(ValidationError):
            LifecycleEvent.model_validate({
                "lifecycle_type": "task", "event": "started", "state": "running",
                "unit_id": "u", "epoch": -1,
            })

    def test_reason_must_be_controlled_code(self):
        with pytest.raises(LifecycleContractError):
            build_lifecycle_event(T.TOOL, K.FAILED, unit_id="u", reason="Exception: boom at /etc/passwd")
        ok = build_lifecycle_event(T.TOOL, K.FAILED, unit_id="u", reason="tool_error")
        assert ok.reason == "tool_error"

    def test_empty_unit_id_raises(self):
        with pytest.raises(LifecycleContractError):
            build_lifecycle_event(T.TOOL, K.STARTED, unit_id="")

    def test_source_fills_provenance_triplet(self):
        src = PlanEvent(plan=Plan(id="p1"))
        src.seq = 7
        src.id = "evt-7"
        ev = build_lifecycle_event(T.PLAN, K.STARTED, unit_id="p1", source=src)
        assert (ev.source_event_type, ev.source_event_id, ev.source_seq) == ("plan", "evt-7", 7)

    def test_no_source_leaves_provenance_none(self):
        ev = build_lifecycle_event(T.TASK, K.STARTED, unit_id="s1")
        assert ev.source_event_type is None and ev.source_event_id is None and ev.source_seq is None


class TestEventUnion:
    def test_union_round_trip(self):
        # F7 兼容：新事件经 TypeAdapter(Event) 全联合反序列化无损往返
        ev = build_lifecycle_event(
            T.SUBAGENT, K.FAILED, unit_id="child-1",
            reason="worker_failed",
            detail=LifecycleDetailV1(original_outcome="failed"),
            parent_unit_id="parent-1",
            correlation=LifecycleCorrelationV1(work_unit_id="wu-1", coordinator_run_id="run-1"),
        )
        ev.seq = 42
        restored = EVENT_ADAPTER.validate_json(ev.model_dump_json())
        assert isinstance(restored, LifecycleEvent)
        assert restored.model_dump() == ev.model_dump()

    def test_union_rejects_unknown_type_string(self):
        # 锁定 F7 依赖的行为：未知 type 走 ValidationError（recovery 层单条 catch-continue）
        with pytest.raises(ValidationError):
            EVENT_ADAPTER.validate_json('{"type": "no_such_event_type_xyz"}')

    def test_existing_members_unaffected(self):
        # additive 断言：旧成员照常反序列化
        restored = EVENT_ADAPTER.validate_json('{"type": "done"}')
        assert restored.type == "done"
