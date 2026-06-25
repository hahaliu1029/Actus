"""[S4 §9] role carry-through through the REAL request->runtime builder.

These tests go through ``_build_work_units_from_requests`` (not just the two
models) to lock the spec §9 contract that ``WorkUnitRequest.role`` is carried
verbatim onto the constructed ``WorkUnit``. Without the ``role=getattr(req,
"role", None)`` field in the builder's ``WorkUnit(...)`` call, the planner-
authored role is silently dropped at the first construction site (the built
unit would have role=None even when the request set role="explorer") — these
tests would FAIL.
"""
from app.domain.models.work_unit import WorkUnitRequest
from app.domain.services.graphs.parallel_execution_subgraph import (
    _build_work_units_from_requests,
)


def test_role_carried_from_request_to_unit():
    # A planner-authored role rides through the real builder onto the WorkUnit.
    req = WorkUnitRequest(objective="o", phase="exploration", role="explorer")
    units = _build_work_units_from_requests(
        [req], step_id_hash16="abc123def4567890", attempt_ix=0
    )
    assert len(units) == 1
    assert units[0].role == "explorer"


def test_role_default_none_carried_through_inv0():
    # INV-0: a request with no role yields a unit with role=None (no behavior
    # change when the planner emits no role).
    req = WorkUnitRequest(objective="o", phase="exploration")
    units = _build_work_units_from_requests(
        [req], step_id_hash16="abc123def4567890", attempt_ix=0
    )
    assert len(units) == 1
    assert units[0].role is None
