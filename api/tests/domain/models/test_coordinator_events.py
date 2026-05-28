"""C2 PR-8 Task 8.1 — domain model tests for coordinator SSE events.

Covers:
- CoordinatorLineageMixin field defaults (all Optional[str] = None).
- 5 new event classes carry the right Literal discriminator + required fields.
- Roundtrip dump/load (so the discriminated Event union can route them).
- Every coordinator event class inherits both BaseEvent and the lineage mixin.
- The Event discriminated union accepts the 5 new types.
"""

import pytest

from app.domain.models.event import (
    BaseEvent,
    CoordinatorApplyEvent,
    CoordinatorDispatchEvent,
    CoordinatorLineageMixin,
    CoordinatorReduceEvent,
    CoordinatorSiblingCancelEvent,
    CoordinatorWorkerSpawnedEvent,
    Event,
)
from app.domain.models.mailbox_envelope import CostAggregate, ResultReadyOutcome
from app.domain.models.patch_apply_plan import GroupOutcome
from pydantic import TypeAdapter


class TestLineageMixinDefaults:
    def test_all_none_when_root(self):
        m = CoordinatorLineageMixin()
        assert m.root_session_id is None
        assert m.parent_session_id is None
        assert m.child_session_id is None
        assert m.coordinator_run_id is None
        assert m.work_unit_id is None

    def test_partial_lineage(self):
        m = CoordinatorLineageMixin(
            root_session_id="root-1",
            coordinator_run_id="run-1",
        )
        assert m.root_session_id == "root-1"
        assert m.coordinator_run_id == "run-1"
        assert m.parent_session_id is None
        assert m.child_session_id is None
        assert m.work_unit_id is None


class TestCoordinatorDispatchEvent:
    def test_required_fields(self):
        e = CoordinatorDispatchEvent(
            step_id="s1",
            work_unit_count=3,
            work_unit_ids=["wu1", "wu2", "wu3"],
            phases=["exploration", "write", "write"],
            coordinator_run_id="r1",
        )
        assert e.type == "coordinator_dispatch"
        assert e.work_unit_count == 3
        assert e.work_unit_ids == ["wu1", "wu2", "wu3"]
        assert e.phases == ["exploration", "write", "write"]
        assert e.coordinator_run_id == "r1"
        # Lineage mixin defaults intact for fields not provided.
        assert e.root_session_id is None
        assert e.parent_session_id is None
        assert e.child_session_id is None
        assert e.work_unit_id is None


class TestCoordinatorWorkerSpawnedEvent:
    def test_required_fields(self):
        e = CoordinatorWorkerSpawnedEvent(
            objective="explore docs",
            phase="exploration",
            allowed_tools=["file_read", "shell_execute"],
            write_lease_count=0,
            child_session_id="child-1",
            work_unit_id="wu-1",
        )
        assert e.type == "coordinator_worker_spawned"
        assert e.phase == "exploration"
        assert e.write_lease_count == 0
        assert e.child_session_id == "child-1"
        assert e.work_unit_id == "wu-1"


class TestCoordinatorReduceEvent:
    def test_required_fields(self):
        e = CoordinatorReduceEvent(
            group_outcome=GroupOutcome.SUCCESS,
            per_worker_outcomes={"wu1": ResultReadyOutcome.SUCCESS},
            diagnostics_summary="all good",
            cost_total=CostAggregate(),
            coordinator_run_id="run-1",
        )
        assert e.type == "coordinator_reduce"
        assert e.group_outcome == GroupOutcome.SUCCESS
        assert e.per_worker_outcomes == {"wu1": ResultReadyOutcome.SUCCESS}
        # conflict_paths defaults to [].
        assert e.conflict_paths == []
        assert e.cost_total == CostAggregate()

    def test_conflict_paths_populated(self):
        e = CoordinatorReduceEvent(
            group_outcome=GroupOutcome.CONFLICT,
            per_worker_outcomes={
                "wu1": ResultReadyOutcome.SUCCESS,
                "wu2": ResultReadyOutcome.SUCCESS,
            },
            diagnostics_summary="conflict on a.py",
            conflict_paths=["a.py"],
            cost_total=CostAggregate(total_input_tokens=10, total_output_tokens=5),
        )
        assert e.conflict_paths == ["a.py"]
        assert e.cost_total.total_input_tokens == 10


class TestCoordinatorApplyEvent:
    def test_minimal(self):
        e = CoordinatorApplyEvent(apply_status="applied")
        assert e.type == "coordinator_apply"
        assert e.apply_status == "applied"
        # Optional fields default appropriately.
        assert e.file_count == 0
        assert e.total_bytes == 0
        assert e.failed_at_path is None
        assert e.rollback_status is None

    def test_rollback_case(self):
        e = CoordinatorApplyEvent(
            apply_status="rolled_back",
            file_count=3,
            total_bytes=2048,
            failed_at_path="src/x.py",
            rollback_status="restored",
        )
        assert e.apply_status == "rolled_back"
        assert e.failed_at_path == "src/x.py"
        assert e.rollback_status == "restored"


class TestCoordinatorSiblingCancelEvent:
    def test_required_fields(self):
        e = CoordinatorSiblingCancelEvent(
            triggered_by_work_unit_id="wu-1",
            triggered_by_outcome=ResultReadyOutcome.FAILED,
            cancelled_work_unit_ids=["wu-2", "wu-3"],
            reason="fail_fast",
        )
        assert e.type == "coordinator_sibling_cancel"
        assert e.triggered_by_work_unit_id == "wu-1"
        assert e.triggered_by_outcome == ResultReadyOutcome.FAILED
        assert e.cancelled_work_unit_ids == ["wu-2", "wu-3"]
        assert e.reason == "fail_fast"


class TestRoundtripDumpLoad:
    def test_dispatch_roundtrip(self):
        e = CoordinatorDispatchEvent(
            step_id="s1",
            work_unit_count=1,
            work_unit_ids=["wu1"],
            phases=["write"],
        )
        dumped = e.model_dump_json()
        loaded = CoordinatorDispatchEvent.model_validate_json(dumped)
        assert loaded.type == "coordinator_dispatch"
        assert loaded.work_unit_ids == ["wu1"]
        assert loaded.phases == ["write"]

    def test_reduce_roundtrip(self):
        e = CoordinatorReduceEvent(
            group_outcome=GroupOutcome.MIXED,
            per_worker_outcomes={"wu1": ResultReadyOutcome.SUCCESS, "wu2": ResultReadyOutcome.FAILED},
            diagnostics_summary="one failed",
            cost_total=CostAggregate(total_input_tokens=100, tool_call_count=2),
        )
        dumped = e.model_dump_json()
        loaded = CoordinatorReduceEvent.model_validate_json(dumped)
        assert loaded.type == "coordinator_reduce"
        assert loaded.group_outcome == GroupOutcome.MIXED
        assert loaded.cost_total.total_input_tokens == 100
        assert loaded.cost_total.tool_call_count == 2

    def test_sibling_cancel_roundtrip(self):
        e = CoordinatorSiblingCancelEvent(
            triggered_by_work_unit_id="wu-1",
            triggered_by_outcome=ResultReadyOutcome.TIMED_OUT,
            cancelled_work_unit_ids=["wu-2"],
            reason="wallclock",
        )
        dumped = e.model_dump_json()
        loaded = CoordinatorSiblingCancelEvent.model_validate_json(dumped)
        assert loaded.triggered_by_outcome == ResultReadyOutcome.TIMED_OUT
        assert loaded.cancelled_work_unit_ids == ["wu-2"]


class TestAllEventsInheritMixin:
    @pytest.mark.parametrize(
        "cls",
        [
            CoordinatorDispatchEvent,
            CoordinatorWorkerSpawnedEvent,
            CoordinatorReduceEvent,
            CoordinatorApplyEvent,
            CoordinatorSiblingCancelEvent,
        ],
    )
    def test_inherits_lineage_mixin(self, cls):
        assert issubclass(cls, CoordinatorLineageMixin)
        assert issubclass(cls, BaseEvent)


class TestEventUnionDispatch:
    """The Event discriminated union must route each new coordinator type."""

    @pytest.fixture
    def adapter(self) -> TypeAdapter:
        return TypeAdapter(Event)

    def test_dispatch_routes_coordinator_dispatch(self, adapter: TypeAdapter):
        payload = CoordinatorDispatchEvent(
            step_id="s1",
            work_unit_count=2,
            work_unit_ids=["wu1", "wu2"],
            phases=["exploration", "write"],
        ).model_dump()
        routed = adapter.validate_python(payload)
        assert isinstance(routed, CoordinatorDispatchEvent)

    def test_dispatch_routes_worker_spawned(self, adapter: TypeAdapter):
        payload = CoordinatorWorkerSpawnedEvent(
            objective="o",
            phase="write",
            allowed_tools=["file_write"],
            write_lease_count=1,
        ).model_dump()
        routed = adapter.validate_python(payload)
        assert isinstance(routed, CoordinatorWorkerSpawnedEvent)

    def test_dispatch_routes_reduce(self, adapter: TypeAdapter):
        payload = CoordinatorReduceEvent(
            group_outcome=GroupOutcome.SUCCESS,
            per_worker_outcomes={"wu1": ResultReadyOutcome.SUCCESS},
            diagnostics_summary="ok",
            cost_total=CostAggregate(),
        ).model_dump()
        routed = adapter.validate_python(payload)
        assert isinstance(routed, CoordinatorReduceEvent)

    def test_dispatch_routes_apply(self, adapter: TypeAdapter):
        payload = CoordinatorApplyEvent(apply_status="applied").model_dump()
        routed = adapter.validate_python(payload)
        assert isinstance(routed, CoordinatorApplyEvent)

    def test_dispatch_routes_sibling_cancel(self, adapter: TypeAdapter):
        payload = CoordinatorSiblingCancelEvent(
            triggered_by_work_unit_id="wu-1",
            triggered_by_outcome=ResultReadyOutcome.FAILED,
            cancelled_work_unit_ids=["wu-2"],
            reason="fail_fast",
        ).model_dump()
        routed = adapter.validate_python(payload)
        assert isinstance(routed, CoordinatorSiblingCancelEvent)
