import pytest
from unittest.mock import AsyncMock, MagicMock
from app.application.services.coordinator_rehydrate_service import (
    CoordinatorRehydrateService,
)

pytestmark = pytest.mark.anyio


async def test_call_time_emitter_emits_health_when_singleton_none():
    """[INV-F3.1] Service built emit_event=None; a call-time emitter still
    surfaces a HealthEvent for a rollback_partial audit."""
    audit = MagicMock(status="rollback_partial", id=7)
    audit_repo = MagicMock()
    audit_repo.find_latest_for_run = AsyncMock(return_value=audit)
    svc = CoordinatorRehydrateService(
        session_repository=MagicMock(),
        envelope_store=MagicMock(),
        audit_repository=audit_repo,
        publisher=MagicMock(),
        emit_event=None,  # singleton field None (lifespan default)
    )
    emitted = []
    async def _emit(ev):
        emitted.append(ev)
    info = await svc._check_already_applied("run-1", emit_event=_emit)
    assert info.status == "rollback_partial"
    assert len(emitted) == 1
    assert emitted[0].metrics["code"] == "coordinator_apply_rollback_partial"
    # [finish-core R1-P1] The alert is INFORMATIONAL — it must be DEGRADED, never
    # TERMINATING. A TERMINATING health event sticky-maps the live (continuing)
    # session to ``timed_out`` on the frontend (session-store.ts), which is wrong:
    # main_graph's ALREADY_APPLIED short-circuit only returns operator text.
    from app.domain.models.event import HealthStatus
    assert emitted[0].status == HealthStatus.DEGRADED


async def test_call_time_emitter_emits_crash_mid_apply_when_singleton_none():
    """[finish-core R3] crash_mid_apply branch must also thread the call-time
    emitter (prod builds the service emit_event=None). Guards against dropping
    emit_event= on the in_progress/crash branch."""
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace
    from app.domain.models.event import HealthStatus
    audit = SimpleNamespace(
        id=99, status="in_progress",
        started_at=datetime.now(timezone.utc) - timedelta(seconds=600),
    )
    audit_repo = MagicMock()
    audit_repo.find_latest_for_run = AsyncMock(return_value=audit)
    svc = CoordinatorRehydrateService(
        session_repository=MagicMock(), envelope_store=MagicMock(),
        audit_repository=audit_repo, publisher=MagicMock(), emit_event=None,
    )
    emitted = []
    async def _emit(ev):
        emitted.append(ev)
    info = await svc._check_already_applied("run-1", emit_event=_emit)
    assert info.status == "crash_mid_apply"
    assert len(emitted) == 1
    assert emitted[0].metrics["code"] == "coordinator_apply_crash_mid_apply"
    assert emitted[0].status == HealthStatus.DEGRADED


async def test_call_time_emitter_preferred_and_singleton_not_mutated():
    """[finish-core R3] When BOTH a construction-time singleton and a call-time
    emitter exist, ONLY the call-time one is used, and the singleton field is
    never mutated (prevents per-run queue leaking into later singleton calls)."""
    from app.domain.models.event import HealthStatus
    singleton = AsyncMock()
    audit = MagicMock(status="rollback_partial", id=7)
    audit_repo = MagicMock()
    audit_repo.find_latest_for_run = AsyncMock(return_value=audit)
    svc = CoordinatorRehydrateService(
        session_repository=MagicMock(), envelope_store=MagicMock(),
        audit_repository=audit_repo, publisher=MagicMock(), emit_event=singleton,
    )
    call_time = []
    async def _emit(ev):
        call_time.append(ev)
    await svc._check_already_applied("run-1", emit_event=_emit)
    assert len(call_time) == 1                  # call-time used
    singleton.assert_not_awaited()              # singleton NOT used
    assert svc._emit_event is singleton         # field unchanged (no mutation)


async def test_detect_existing_run_threads_emit_event(monkeypatch):
    """detect_existing_run forwards a call-time emitter into _check_already_applied."""
    svc = CoordinatorRehydrateService(
        session_repository=MagicMock(find_children_by_coordinator_run=AsyncMock(return_value=[])),
        envelope_store=MagicMock(),
        audit_repository=MagicMock(),
        publisher=MagicMock(),
        emit_event=None,
    )
    captured = {}
    async def _check(run_id, emit_event=None):
        captured["emit"] = emit_event
        return None
    monkeypatch.setattr(svc, "_check_already_applied", _check)
    sentinel = lambda ev: None
    svc._sr.find_children_by_coordinator_run = AsyncMock(return_value=[
        MagicMock(id="c1", work_unit_id="wu1", status=MagicMock(value="running")),
    ])
    svc._es.find_terminal_envelopes_by_run = AsyncMock(return_value=[])
    await svc.detect_existing_run(
        coordinator_run_id="run-1", parent_session_id="p", emit_event=sentinel,
    )
    assert captured["emit"] is sentinel
