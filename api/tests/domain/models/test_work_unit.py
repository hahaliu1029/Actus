import pytest
from pydantic import ValidationError
from app.domain.models.work_unit import (
    ParallelRunSpec, ParallelWorkUnitGroupRequest, PathLease,
    ProposedPath, WorkUnit, WorkUnitRequest,
)

class TestProposedPath:
    def test_valid(self):
        assert ProposedPath(path="/x", op="add").op == "add"
    def test_invalid_op(self):
        with pytest.raises(ValidationError):
            ProposedPath(path="/x", op="invalid")  # type: ignore[arg-type]

class TestPathLease:
    def test_modify_plan_time_allows_none_digest(self):
        assert PathLease(path="/a", op="modify").base_digest is None
    def test_add_must_not_have_digest(self):
        with pytest.raises(ValidationError, match="op=add"):
            PathLease(path="/a", op="add", base_digest="abc")
    def test_add_must_not_have_seed(self):
        with pytest.raises(ValidationError, match="op=add"):
            PathLease(path="/a", op="add", seed_content_ref="ref")

class TestWorkUnitRequest:
    def test_exploration_empty_paths(self):
        wu = WorkUnitRequest(objective="x", phase="exploration", allowed_tools=["file_read"])
        assert wu.proposed_paths == []
    def test_write_must_have_paths(self):
        with pytest.raises(ValidationError, match="phase=write"):
            WorkUnitRequest(objective="x", phase="write", allowed_tools=["file_write"], proposed_paths=[])

class TestWorkUnit:
    def test_write_must_have_lease(self):
        with pytest.raises(ValidationError, match="phase=write"):
            WorkUnit(work_unit_id="x.a1.0", objective="x", phase="write", allowed_tools=["file_write"], write_lease=[])
    def test_exploration_ok_empty_lease(self):
        wu = WorkUnit(work_unit_id="x.a1.0", objective="x", phase="exploration", allowed_tools=["file_read"], write_lease=[])
        assert wu.phase == "exploration"

class TestParallelWorkUnitGroupRequest:
    def test_no_run_id(self):
        req = ParallelWorkUnitGroupRequest(work_units=[
            WorkUnitRequest(objective="x", phase="exploration", allowed_tools=["file_read"]),
        ])
        assert not hasattr(req, "coordinator_run_id")

class TestParallelRunSpec:
    def test_runtime_has_run_id(self):
        spec = ParallelRunSpec(
            coordinator_run_id="s1:abcd1234abcd1234:a1",
            work_units=[WorkUnit(work_unit_id="abcd1234abcd1234.a1.0", objective="x",
                                  phase="exploration", allowed_tools=["file_read"], write_lease=[])],
        )
        assert spec.coordinator_run_id.startswith("s1:")
