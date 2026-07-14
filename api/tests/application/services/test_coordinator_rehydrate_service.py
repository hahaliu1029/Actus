"""Unit tests for CoordinatorRehydrateService (PR-7 Task 7.3)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.application.services.coordinator_rehydrate_service import (
    AlreadyAppliedInfo,
    CoordinatorRehydrateService,
)

pytestmark = pytest.mark.anyio


def _make_envelope(
    wu_id,
    *,
    envelope_type="RESULT_READY",
    payload=None,
    child_session_id=None,
    received_at=None,
):
    return SimpleNamespace(
        work_unit_id=wu_id,
        envelope_type=envelope_type,
        payload=payload or {"outcome": "success"},
        child_session_id=child_session_id or f"c_{wu_id}",
        received_at=received_at or datetime.now(timezone.utc),
    )


def _make_child(*, session_id, wu_id, status="running"):
    return SimpleNamespace(id=session_id, work_unit_id=wu_id, status=status)


class _FakeApplyReconcileMarkerStore:
    def __init__(self) -> None:
        self.entries: dict[str, tuple[float, float]] = {}
        self.get_or_create_calls: list[tuple[str, float, int]] = []
        self.clear_calls: list[str] = []

    async def get_or_create(
        self,
        coordinator_run_id: str,
        *,
        now_epoch: float,
        ttl_seconds: int,
    ) -> float:
        self.get_or_create_calls.append(
            (coordinator_run_id, now_epoch, ttl_seconds),
        )
        existing = self.entries.get(coordinator_run_id)
        if existing is not None and existing[1] > now_epoch:
            return existing[0]
        self.entries[coordinator_run_id] = (
            now_epoch, now_epoch + ttl_seconds,
        )
        return now_epoch

    async def clear(self, coordinator_run_id: str) -> None:
        self.clear_calls.append(coordinator_run_id)
        self.entries.pop(coordinator_run_id, None)


async def test_returns_none_when_no_children():
    sr = AsyncMock()
    sr.find_children_by_coordinator_run = AsyncMock(return_value=[])
    es = AsyncMock()
    ar = AsyncMock()
    svc = CoordinatorRehydrateService(
        session_repository=sr, envelope_store=es, audit_repository=ar,
    )
    out = await svc.detect_existing_run(
        coordinator_run_id="r1", parent_session_id="p1",
    )
    assert out is None
    # [codex R3 P2] Explicit kwarg-shape assertion catches future Protocol
    # drift on ``SessionRepository.find_children_by_coordinator_run``
    # (bare AsyncMock can't catch signature changes by itself).
    sr.find_children_by_coordinator_run.assert_awaited_once_with(
        coordinator_run_id="r1", parent_session_id="p1",
    )


async def test_pending_only_when_no_terminals():
    sr = AsyncMock()
    sr.find_children_by_coordinator_run = AsyncMock(return_value=[
        _make_child(session_id="c1", wu_id="wu1", status="running"),
        _make_child(session_id="c2", wu_id="wu2", status="pending"),
    ])
    es = AsyncMock()
    es.find_terminal_envelopes_by_run = AsyncMock(return_value=[])
    ar = AsyncMock()
    ar.find_latest_for_run = AsyncMock(return_value=None)
    svc = CoordinatorRehydrateService(
        session_repository=sr, envelope_store=es, audit_repository=ar,
    )
    res = await svc.detect_existing_run(
        coordinator_run_id="r1", parent_session_id="p1",
    )
    assert res is not None
    assert res.pending == ["wu1", "wu2"]
    assert res.terminal == {}
    assert res.child_session_ids == {"wu1": "c1", "wu2": "c2"}
    assert res.already_applied is None


async def test_terminal_skipped_from_pending_and_preserves_envelope_type():
    sr = AsyncMock()
    sr.find_children_by_coordinator_run = AsyncMock(return_value=[
        _make_child(session_id="c1", wu_id="wu1", status="completed"),
        _make_child(session_id="c2", wu_id="wu2", status="running"),
    ])
    es = AsyncMock()
    es.find_terminal_envelopes_by_run = AsyncMock(return_value=[
        _make_envelope("wu1", envelope_type="RESULT_READY",
                       payload={"outcome": "success"}),
    ])
    ar = AsyncMock()
    ar.find_latest_for_run = AsyncMock(return_value=None)
    svc = CoordinatorRehydrateService(
        session_repository=sr, envelope_store=es, audit_repository=ar,
    )
    res = await svc.detect_existing_run(
        coordinator_run_id="r1", parent_session_id="p1",
    )
    assert res.pending == ["wu2"]
    assert "wu1" in res.terminal
    assert res.terminal["wu1"].envelope_type == "RESULT_READY"


async def test_cancel_ack_terminal_preserved():
    sr = AsyncMock()
    sr.find_children_by_coordinator_run = AsyncMock(return_value=[
        _make_child(session_id="c1", wu_id="wu1", status="cancelled"),
    ])
    es = AsyncMock()
    es.find_terminal_envelopes_by_run = AsyncMock(return_value=[
        _make_envelope("wu1", envelope_type="CANCEL_ACK"),
    ])
    ar = AsyncMock()
    ar.find_latest_for_run = AsyncMock(return_value=None)
    svc = CoordinatorRehydrateService(
        session_repository=sr, envelope_store=es, audit_repository=ar,
    )
    res = await svc.detect_existing_run(
        coordinator_run_id="r1", parent_session_id="p1",
    )
    assert res.terminal["wu1"].envelope_type == "CANCEL_ACK"


async def test_already_applied_success_short_circuit():
    sr = AsyncMock()
    sr.find_children_by_coordinator_run = AsyncMock(return_value=[
        _make_child(session_id="c1", wu_id="wu1", status="completed"),
    ])
    es = AsyncMock()
    es.find_terminal_envelopes_by_run = AsyncMock(return_value=[])
    ar = AsyncMock()
    ar.find_latest_for_run = AsyncMock(return_value=SimpleNamespace(
        id=42, status="success",
        started_at=datetime.now(timezone.utc),
    ))
    svc = CoordinatorRehydrateService(
        session_repository=sr, envelope_store=es, audit_repository=ar,
    )
    res = await svc.detect_existing_run(
        coordinator_run_id="r1", parent_session_id="p1",
    )
    assert res.already_applied == AlreadyAppliedInfo(status="success", audit_id=42)


async def test_already_applied_rollback_partial_emits_health_event():
    emitter = AsyncMock()
    sr = AsyncMock()
    sr.find_children_by_coordinator_run = AsyncMock(return_value=[
        _make_child(session_id="c1", wu_id="wu1", status="completed"),
    ])
    es = AsyncMock()
    es.find_terminal_envelopes_by_run = AsyncMock(return_value=[])
    ar = AsyncMock()
    ar.find_latest_for_run = AsyncMock(return_value=SimpleNamespace(
        id=7, status="rollback_partial",
        started_at=datetime.now(timezone.utc),
    ))
    svc = CoordinatorRehydrateService(
        session_repository=sr, envelope_store=es, audit_repository=ar,
        emit_event=emitter,
    )
    res = await svc.detect_existing_run(
        coordinator_run_id="r1", parent_session_id="p1",
    )
    assert res.already_applied.status == "rollback_partial"
    assert emitter.await_count == 1
    emitted = emitter.await_args.args[0]
    from app.domain.models.event import HealthEvent, HealthStatus
    assert isinstance(emitted, HealthEvent)
    # [finish-core R1-P1] DEGRADED (informational), NOT TERMINATING — a rehydrate
    # recovery alert must not sticky-map the live session to timed_out on the
    # frontend (the session continues; main_graph only returns operator text).
    assert emitted.status == HealthStatus.DEGRADED
    assert emitted.metrics["code"] == "coordinator_apply_rollback_partial"


async def test_already_applied_crash_mid_apply_only_after_lock_missing_grace():
    emitter = AsyncMock()
    now = datetime(2026, 7, 15, 8, 0, tzinfo=timezone.utc)
    markers = _FakeApplyReconcileMarkerStore()
    markers.entries["r1"] = (
        now.timestamp() - 60,
        now.timestamp() + 86_340,
    )
    sr = AsyncMock()
    sr.find_children_by_coordinator_run = AsyncMock(return_value=[
        _make_child(session_id="c1", wu_id="wu1", status="running"),
    ])
    es = AsyncMock()
    es.find_terminal_envelopes_by_run = AsyncMock(return_value=[])
    ar = AsyncMock()
    ar.find_latest_for_run = AsyncMock(return_value=SimpleNamespace(
        id=99, status="in_progress",
        started_at=now - timedelta(seconds=600),
    ))
    svc = CoordinatorRehydrateService(
        session_repository=sr, envelope_store=es, audit_repository=ar,
        emit_event=emitter,
        apply_lock_probe=AsyncMock(return_value=False),
        reconcile_marker_store=markers,
        clock=lambda: now,
    )
    res = await svc.detect_existing_run(
        coordinator_run_id="r1", parent_session_id="p1",
    )
    assert res.already_applied.status == "crash_mid_apply"
    assert emitter.await_count == 1
    assert emitter.await_args.args[0].metrics["code"] == "coordinator_apply_crash_mid_apply"


async def test_in_progress_recent_no_alert():
    emitter = AsyncMock()
    sr = AsyncMock()
    sr.find_children_by_coordinator_run = AsyncMock(return_value=[
        _make_child(session_id="c1", wu_id="wu1", status="running"),
    ])
    es = AsyncMock()
    es.find_terminal_envelopes_by_run = AsyncMock(return_value=[])
    ar = AsyncMock()
    ar.find_latest_for_run = AsyncMock(return_value=SimpleNamespace(
        id=11, status="in_progress",
        started_at=datetime.now(timezone.utc) - timedelta(seconds=30),
    ))
    svc = CoordinatorRehydrateService(
        session_repository=sr, envelope_store=es, audit_repository=ar,
        emit_event=emitter,
    )
    res = await svc.detect_existing_run(
        coordinator_run_id="r1", parent_session_id="p1",
    )
    assert res.already_applied.status == "in_progress_recent"
    assert emitter.await_count == 0  # no alert


async def test_emit_event_failure_does_not_break_rehydrate():
    """[best-effort] emit_event raises but rehydrate result still returns."""
    emitter = AsyncMock(side_effect=RuntimeError("event sink down"))
    sr = AsyncMock()
    sr.find_children_by_coordinator_run = AsyncMock(return_value=[
        _make_child(session_id="c1", wu_id="wu1", status="completed"),
    ])
    es = AsyncMock()
    es.find_terminal_envelopes_by_run = AsyncMock(return_value=[])
    ar = AsyncMock()
    ar.find_latest_for_run = AsyncMock(return_value=SimpleNamespace(
        id=7, status="rollback_partial",
        started_at=datetime.now(timezone.utc),
    ))
    svc = CoordinatorRehydrateService(
        session_repository=sr, envelope_store=es, audit_repository=ar,
        emit_event=emitter,
    )
    res = await svc.detect_existing_run(
        coordinator_run_id="r1", parent_session_id="p1",
    )
    assert res.already_applied.status == "rollback_partial"


async def test_naive_started_at_normalized_to_utc():
    """Defensive: some ORM paths may yield naive datetime — service must not crash."""
    emitter = AsyncMock()
    sr = AsyncMock()
    sr.find_children_by_coordinator_run = AsyncMock(return_value=[
        _make_child(session_id="c1", wu_id="wu1", status="running"),
    ])
    es = AsyncMock()
    es.find_terminal_envelopes_by_run = AsyncMock(return_value=[])
    ar = AsyncMock()
    now = datetime(2026, 7, 15, 8, 0, tzinfo=timezone.utc)
    naive_old = (now - timedelta(seconds=600)).replace(tzinfo=None)
    markers = _FakeApplyReconcileMarkerStore()
    markers.entries["r1"] = (
        now.timestamp() - 60,
        now.timestamp() + 86_340,
    )
    ar.find_latest_for_run = AsyncMock(return_value=SimpleNamespace(
        id=33, status="in_progress",
        started_at=naive_old,
    ))
    svc = CoordinatorRehydrateService(
        session_repository=sr, envelope_store=es, audit_repository=ar,
        emit_event=emitter,
        apply_lock_probe=AsyncMock(return_value=False),
        reconcile_marker_store=markers,
        clock=lambda: now,
    )
    res = await svc.detect_existing_run(
        coordinator_run_id="r1", parent_session_id="p1",
    )
    assert res.already_applied.status == "crash_mid_apply"


async def test_old_in_progress_with_fresh_apply_owner_is_recent_and_clears_marker():
    now = datetime(2026, 7, 15, 8, 0, tzinfo=timezone.utc)
    markers = _FakeApplyReconcileMarkerStore()
    markers.entries["run-live"] = (
        now.timestamp() - 900,
        now.timestamp() + 85_500,
    )
    audit_repo = AsyncMock()
    audit_repo.find_latest_for_run = AsyncMock(return_value=SimpleNamespace(
        id=70,
        status="in_progress",
        started_at=now - timedelta(hours=3),
    ))
    lock_probe = AsyncMock(return_value=True)
    emitter = AsyncMock()
    svc = CoordinatorRehydrateService(
        session_repository=AsyncMock(),
        envelope_store=AsyncMock(),
        audit_repository=audit_repo,
        emit_event=emitter,
        apply_lock_probe=lock_probe,
        reconcile_marker_store=markers,
        clock=lambda: now,
    )

    result = await svc._check_already_applied("run-live")

    assert result == AlreadyAppliedInfo(
        status="in_progress_recent", audit_id=70,
    )
    lock_probe.assert_awaited_once_with("run-live")
    assert markers.clear_calls == ["run-live"]
    assert "run-live" not in markers.entries
    emitter.assert_not_awaited()


async def test_missing_apply_lock_first_observation_only_creates_ttl_marker():
    now = datetime(2026, 7, 15, 8, 0, tzinfo=timezone.utc)
    markers = _FakeApplyReconcileMarkerStore()
    audit_repo = AsyncMock()
    audit_repo.find_latest_for_run = AsyncMock(return_value=SimpleNamespace(
        id=71,
        status="in_progress",
        started_at=now - timedelta(hours=3),
    ))
    emitter = AsyncMock()
    svc = CoordinatorRehydrateService(
        session_repository=AsyncMock(),
        envelope_store=AsyncMock(),
        audit_repository=audit_repo,
        emit_event=emitter,
        apply_lock_probe=AsyncMock(return_value=False),
        reconcile_marker_store=markers,
        clock=lambda: now,
    )

    result = await svc._check_already_applied("run-first")

    assert result == AlreadyAppliedInfo(
        status="in_progress_recent", audit_id=71,
    )
    assert markers.get_or_create_calls == [
        ("run-first", now.timestamp(), 86_400),
    ]
    emitter.assert_not_awaited()


async def test_missing_apply_lock_within_grace_remains_recent():
    now = datetime(2026, 7, 15, 8, 0, tzinfo=timezone.utc)
    markers = _FakeApplyReconcileMarkerStore()
    markers.entries["run-grace"] = (
        now.timestamp() - 29,
        now.timestamp() + 86_371,
    )
    audit_repo = AsyncMock()
    audit_repo.find_latest_for_run = AsyncMock(return_value=SimpleNamespace(
        id=72,
        status="in_progress",
        started_at=now - timedelta(hours=3),
    ))
    emitter = AsyncMock()
    svc = CoordinatorRehydrateService(
        session_repository=AsyncMock(),
        envelope_store=AsyncMock(),
        audit_repository=audit_repo,
        emit_event=emitter,
        apply_lock_probe=AsyncMock(return_value=False),
        reconcile_marker_store=markers,
        clock=lambda: now,
    )

    result = await svc._check_already_applied("run-grace")

    assert result == AlreadyAppliedInfo(
        status="in_progress_recent", audit_id=72,
    )
    emitter.assert_not_awaited()


async def test_missing_apply_lock_after_minutes_still_crashes_before_marker_ttl():
    now = datetime(2026, 7, 15, 8, 5, tzinfo=timezone.utc)
    markers = _FakeApplyReconcileMarkerStore()
    markers.entries["run-stale"] = (
        now.timestamp() - 300,
        now.timestamp() + 86_100,
    )
    audit_repo = AsyncMock()
    audit_repo.find_latest_for_run = AsyncMock(return_value=SimpleNamespace(
        id=73,
        status="in_progress",
        started_at=now - timedelta(hours=3),
    ))
    emitter = AsyncMock()
    svc = CoordinatorRehydrateService(
        session_repository=AsyncMock(),
        envelope_store=AsyncMock(),
        audit_repository=audit_repo,
        emit_event=emitter,
        apply_lock_probe=AsyncMock(return_value=False),
        reconcile_marker_store=markers,
        clock=lambda: now,
    )

    result = await svc._check_already_applied("run-stale")

    assert result == AlreadyAppliedInfo(
        status="crash_mid_apply", audit_id=73,
    )
    assert emitter.await_args.args[0].metrics["code"] == (
        "coordinator_apply_crash_mid_apply"
    )


async def test_reconcile_markers_are_isolated_by_coordinator_run_id():
    now = datetime(2026, 7, 15, 8, 5, tzinfo=timezone.utc)
    markers = _FakeApplyReconcileMarkerStore()
    markers.entries["run-old"] = (
        now.timestamp() - 300,
        now.timestamp() + 86_100,
    )
    audit_repo = AsyncMock()
    audit_repo.find_latest_for_run = AsyncMock(return_value=SimpleNamespace(
        id=74,
        status="in_progress",
        started_at=now - timedelta(hours=3),
    ))
    svc = CoordinatorRehydrateService(
        session_repository=AsyncMock(),
        envelope_store=AsyncMock(),
        audit_repository=audit_repo,
        apply_lock_probe=AsyncMock(return_value=False),
        reconcile_marker_store=markers,
        clock=lambda: now,
    )

    old = await svc._check_already_applied("run-old")
    fresh = await svc._check_already_applied("run-new")

    assert old.status == "crash_mid_apply"
    assert fresh.status == "in_progress_recent"
    assert "run-old" in markers.entries
    assert markers.entries["run-new"][0] == now.timestamp()


async def test_success_and_rollback_partial_do_not_probe_apply_lock():
    probe = AsyncMock(side_effect=AssertionError("terminal audit must not probe"))
    markers = _FakeApplyReconcileMarkerStore()
    audit_repo = AsyncMock()
    svc = CoordinatorRehydrateService(
        session_repository=AsyncMock(),
        envelope_store=AsyncMock(),
        audit_repository=audit_repo,
        apply_lock_probe=probe,
        reconcile_marker_store=markers,
    )
    audit_repo.find_latest_for_run = AsyncMock(return_value=SimpleNamespace(
        id=75, status="success", started_at=datetime.now(timezone.utc),
    ))
    assert (await svc._check_already_applied("run-success")).status == "success"
    audit_repo.find_latest_for_run = AsyncMock(return_value=SimpleNamespace(
        id=76, status="rollback_partial", started_at=datetime.now(timezone.utc),
    ))
    assert (await svc._check_already_applied("run-rollback")).status == (
        "rollback_partial"
    )
    probe.assert_not_awaited()
    assert markers.get_or_create_calls == []


async def test_real_session_status_enum_routed_to_pending():
    """[codex R2 P1] Regression: live ``Session.status`` is a
    ``SessionStatus`` enum whose ``.value`` is LOWERCASE ('pending' /
    'running'). The pending-routing predicate must match via .value
    extraction (status.value if hasattr else str(status)). If a future
    refactor reintroduces uppercase comparison, real children would
    silently be filtered OUT of the Send fan-out -- this test catches it.
    """
    from app.domain.models.session import SessionStatus
    sr = AsyncMock()
    sr.find_children_by_coordinator_run = AsyncMock(return_value=[
        SimpleNamespace(
            id="c1", work_unit_id="wu1",
            status=SessionStatus.RUNNING,  # real enum, NOT string
        ),
        SimpleNamespace(
            id="c2", work_unit_id="wu2",
            status=SessionStatus.PENDING,  # real enum, NOT string
        ),
    ])
    es = AsyncMock()
    es.find_terminal_envelopes_by_run = AsyncMock(return_value=[])
    ar = AsyncMock()
    ar.find_latest_for_run = AsyncMock(return_value=None)
    svc = CoordinatorRehydrateService(
        session_repository=sr, envelope_store=es, audit_repository=ar,
    )
    res = await svc.detect_existing_run(
        coordinator_run_id="r1", parent_session_id="p1",
    )
    assert res is not None
    # Both children must land in pending; no silent filter-out.
    assert res.pending == ["wu1", "wu2"]
