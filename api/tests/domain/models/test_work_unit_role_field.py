from app.domain.models.work_unit import WorkUnit, WorkUnitRequest


def test_request_role_defaults_none():
    r = WorkUnitRequest(objective="o", phase="exploration")
    assert r.role is None


def test_request_role_roundtrips():
    r = WorkUnitRequest(objective="o", phase="exploration", role="explorer")
    assert r.role == "explorer"


def test_work_unit_role_defaults_none():
    wu = WorkUnit(work_unit_id="w.a0.0", objective="o", phase="exploration")
    assert wu.role is None
