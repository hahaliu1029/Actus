"""Unit tests for CoordinatorRehydrateService (PR-7 Task 7.3)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.coordinator_rehydrate_service import (
    AlreadyAppliedInfo,
    ApplyLeaseObservation,
    CoordinatorRehydrateService,
)

pytestmark = pytest.mark.anyio


async def _run_under_crash_fence(_run_id, operation, **_kwargs):
    return await operation()


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


@pytest.mark.parametrize(
    "overrides",
    [
        {"apply_reconcile_grace_seconds": True},
        {"apply_reconcile_grace_seconds": 0},
        {"apply_reconcile_grace_seconds": -1},
        {"apply_reconcile_grace_seconds": 1.5},
        {"apply_reconcile_grace_seconds": "30"},
        {"apply_reconcile_marker_ttl_seconds": True},
        {"apply_reconcile_marker_ttl_seconds": 30},
        {"apply_reconcile_marker_ttl_seconds": 29},
        {"apply_reconcile_marker_ttl_seconds": 86_400.5},
        {"apply_reconcile_marker_ttl_seconds": float("inf")},
    ],
)
def test_rehydrate_reconcile_windows_require_ordered_positive_integers(
    overrides,
) -> None:
    with pytest.raises(ValueError, match="apply_reconcile"):
        CoordinatorRehydrateService(
            session_repository=AsyncMock(),
            envelope_store=AsyncMock(),
            audit_repository=AsyncMock(),
            **overrides,
        )


async def test_returns_none_when_no_children():
    sr = AsyncMock()
    sr.find_children_by_coordinator_run = AsyncMock(return_value=[])
    es = AsyncMock()
    ar = AsyncMock()
    ar.update_terminal = AsyncMock(return_value=True)
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
    sr = AsyncMock()
    sr.find_children_by_coordinator_run = AsyncMock(return_value=[
        _make_child(session_id="c1", wu_id="wu1", status="running"),
    ])
    es = AsyncMock()
    es.find_terminal_envelopes_by_run = AsyncMock(return_value=[])
    ar = AsyncMock()
    ar.update_terminal = AsyncMock(return_value=True)
    ar.find_latest_for_run = AsyncMock(return_value=SimpleNamespace(
        id=99, status="in_progress",
        started_at=now - timedelta(seconds=600),
    ))
    svc = CoordinatorRehydrateService(
        session_repository=sr, envelope_store=es, audit_repository=ar,
        emit_event=emitter,
        apply_lease_observer=AsyncMock(return_value=ApplyLeaseObservation(
            owner_is_live=False, missing_for_seconds=60,
        )),
        apply_crash_fence=_run_under_crash_fence,
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
    ar.update_terminal = AsyncMock(return_value=True)
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
    ar.update_terminal = AsyncMock(return_value=True)
    now = datetime(2026, 7, 15, 8, 0, tzinfo=timezone.utc)
    naive_old = (now - timedelta(seconds=600)).replace(tzinfo=None)
    ar.find_latest_for_run = AsyncMock(return_value=SimpleNamespace(
        id=33, status="in_progress",
        started_at=naive_old,
    ))
    svc = CoordinatorRehydrateService(
        session_repository=sr, envelope_store=es, audit_repository=ar,
        emit_event=emitter,
        apply_lease_observer=AsyncMock(return_value=ApplyLeaseObservation(
            owner_is_live=False, missing_for_seconds=60,
        )),
        apply_crash_fence=_run_under_crash_fence,
        clock=lambda: now,
    )
    res = await svc.detect_existing_run(
        coordinator_run_id="r1", parent_session_id="p1",
    )
    assert res.already_applied.status == "crash_mid_apply"


async def test_old_in_progress_with_fresh_apply_owner_is_recent():
    now = datetime(2026, 7, 15, 8, 0, tzinfo=timezone.utc)
    audit_repo = AsyncMock()
    audit_repo.update_terminal = AsyncMock(return_value=True)
    audit_repo.find_latest_for_run = AsyncMock(return_value=SimpleNamespace(
        id=70,
        status="in_progress",
        started_at=now - timedelta(hours=3),
    ))
    observer = AsyncMock(return_value=ApplyLeaseObservation(
        owner_is_live=True, missing_for_seconds=0,
    ))
    emitter = AsyncMock()
    svc = CoordinatorRehydrateService(
        session_repository=AsyncMock(),
        envelope_store=AsyncMock(),
        audit_repository=audit_repo,
        emit_event=emitter,
        apply_lease_observer=observer,
        clock=lambda: now,
    )

    result = await svc._check_already_applied("run-live")

    assert result == AlreadyAppliedInfo(
        status="in_progress_recent", audit_id=70,
    )
    observer.assert_awaited_once_with(
        "run-live", marker_ttl_seconds=86_400,
    )
    emitter.assert_not_awaited()


async def test_atomic_apply_lease_observation_ignores_cross_pod_clock_skew():
    now = datetime(2050, 1, 1, tzinfo=timezone.utc)
    audit_repo = AsyncMock()
    audit_repo.find_latest_for_run = AsyncMock(return_value=SimpleNamespace(
        id=701,
        status="in_progress",
        started_at=now - timedelta(hours=3),
    ))
    observer = AsyncMock(return_value=ApplyLeaseObservation(
        owner_is_live=False,
        missing_for_seconds=29.0,
    ))
    svc = CoordinatorRehydrateService(
        session_repository=AsyncMock(),
        envelope_store=AsyncMock(),
        audit_repository=audit_repo,
        apply_lease_observer=observer,
        # The pod wall clock is intentionally unrelated to the Redis lease
        # clock. It may describe audit age, but never marker elapsed time.
        clock=lambda: now,
    )

    result = await svc._check_already_applied("run-clock-skew")

    assert result == AlreadyAppliedInfo(
        status="in_progress_recent", audit_id=701,
    )
    observer.assert_awaited_once_with(
        "run-clock-skew", marker_ttl_seconds=86_400,
    )


async def test_atomic_live_then_missing_observation_restarts_full_grace():
    now = datetime(2026, 7, 15, 8, 0, tzinfo=timezone.utc)
    audit_repo = AsyncMock()
    audit_repo.find_latest_for_run = AsyncMock(return_value=SimpleNamespace(
        id=702,
        status="in_progress",
        started_at=now - timedelta(hours=3),
    ))
    observer = AsyncMock(side_effect=[
        ApplyLeaseObservation(owner_is_live=True, missing_for_seconds=0.0),
        ApplyLeaseObservation(owner_is_live=False, missing_for_seconds=0.0),
    ])
    svc = CoordinatorRehydrateService(
        session_repository=AsyncMock(),
        envelope_store=AsyncMock(),
        audit_repository=audit_repo,
        apply_lease_observer=observer,
        clock=lambda: now,
    )

    live = await svc._check_already_applied("run-cycle")
    first_missing = await svc._check_already_applied("run-cycle")

    assert live.status == "in_progress_recent"
    assert first_missing.status == "in_progress_recent"
    assert observer.await_count == 2


async def test_missing_apply_lock_first_observation_only_creates_ttl_marker():
    now = datetime(2026, 7, 15, 8, 0, tzinfo=timezone.utc)
    observer = AsyncMock(return_value=ApplyLeaseObservation(
        owner_is_live=False, missing_for_seconds=0,
    ))
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
        apply_lease_observer=observer,
        clock=lambda: now,
    )

    result = await svc._check_already_applied("run-first")

    assert result == AlreadyAppliedInfo(
        status="in_progress_recent", audit_id=71,
    )
    observer.assert_awaited_once_with(
        "run-first", marker_ttl_seconds=86_400,
    )
    emitter.assert_not_awaited()


async def test_missing_apply_lock_within_grace_remains_recent():
    now = datetime(2026, 7, 15, 8, 0, tzinfo=timezone.utc)
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
        apply_lease_observer=AsyncMock(return_value=ApplyLeaseObservation(
            owner_is_live=False, missing_for_seconds=29,
        )),
        clock=lambda: now,
    )

    result = await svc._check_already_applied("run-grace")

    assert result == AlreadyAppliedInfo(
        status="in_progress_recent", audit_id=72,
    )
    emitter.assert_not_awaited()


async def test_missing_apply_lock_after_minutes_still_crashes_before_marker_ttl():
    now = datetime(2026, 7, 15, 8, 5, tzinfo=timezone.utc)
    audit_repo = AsyncMock()
    audit_repo.update_terminal = AsyncMock(return_value=True)
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
        apply_lease_observer=AsyncMock(return_value=ApplyLeaseObservation(
            owner_is_live=False, missing_for_seconds=300,
        )),
        apply_crash_fence=_run_under_crash_fence,
        clock=lambda: now,
    )

    result = await svc._check_already_applied("run-stale")

    assert result == AlreadyAppliedInfo(
        status="crash_mid_apply", audit_id=73,
    )
    audit_repo.update_terminal.assert_awaited_once_with(
        73,
        status="crash_mid_apply",
        failed_reason=(
            "apply owner continuously missing beyond reconciliation grace"
        ),
    )
    assert emitter.await_args.args[0].metrics["code"] == (
        "coordinator_apply_crash_mid_apply"
    )


async def test_crash_classification_fails_safe_when_fence_is_busy():
    now = datetime(2026, 7, 15, 8, 5, tzinfo=timezone.utc)
    audit_repo = AsyncMock()
    audit_repo.find_latest_for_run = AsyncMock(return_value=SimpleNamespace(
        id=730,
        status="in_progress",
        started_at=now - timedelta(hours=3),
    ))
    audit_repo.update_terminal = AsyncMock(return_value=True)
    fence = AsyncMock(return_value=None)
    emitter = AsyncMock()
    svc = CoordinatorRehydrateService(
        session_repository=AsyncMock(),
        envelope_store=AsyncMock(),
        audit_repository=audit_repo,
        emit_event=emitter,
        apply_lease_observer=AsyncMock(return_value=ApplyLeaseObservation(
            owner_is_live=False, missing_for_seconds=300,
        )),
        apply_crash_fence=fence,
        clock=lambda: now,
    )

    result = await svc._check_already_applied("run-fence-busy")

    assert result == AlreadyAppliedInfo(
        status="in_progress_recent", audit_id=730,
    )
    audit_repo.update_terminal.assert_not_awaited()
    emitter.assert_not_awaited()


async def test_crash_cas_miss_rereads_concurrent_success():
    now = datetime(2026, 7, 15, 8, 5, tzinfo=timezone.utc)
    stale = SimpleNamespace(
        id=732,
        status="in_progress",
        started_at=now - timedelta(hours=3),
    )
    success = SimpleNamespace(
        id=732,
        status="success",
        started_at=stale.started_at,
    )
    audit_repo = AsyncMock()
    audit_repo.find_latest_for_run = AsyncMock(
        side_effect=[stale, success, success],
    )
    audit_repo.update_terminal = AsyncMock(return_value=False)
    emitter = AsyncMock()
    svc = CoordinatorRehydrateService(
        session_repository=AsyncMock(),
        envelope_store=AsyncMock(),
        audit_repository=audit_repo,
        emit_event=emitter,
        apply_lease_observer=AsyncMock(return_value=ApplyLeaseObservation(
            owner_is_live=False, missing_for_seconds=300,
        )),
        apply_crash_fence=_run_under_crash_fence,
        clock=lambda: now,
    )

    result = await svc._check_already_applied("run-raced-success")

    assert result == AlreadyAppliedInfo(status="success", audit_id=732)
    assert audit_repo.find_latest_for_run.await_count == 3
    emitter.assert_not_awaited()


async def test_crash_fence_rechecks_latest_attempt_before_cas():
    now = datetime(2026, 7, 15, 8, 5, tzinfo=timezone.utc)
    old = SimpleNamespace(
        id=733,
        status="in_progress",
        started_at=now - timedelta(hours=3),
    )
    replacement = SimpleNamespace(
        id=734,
        status="in_progress",
        started_at=now,
    )
    audit_repo = AsyncMock()
    audit_repo.find_latest_for_run = AsyncMock(
        side_effect=[old, replacement, replacement],
    )
    audit_repo.update_terminal = AsyncMock(return_value=True)
    svc = CoordinatorRehydrateService(
        session_repository=AsyncMock(),
        envelope_store=AsyncMock(),
        audit_repository=audit_repo,
        apply_lease_observer=AsyncMock(return_value=ApplyLeaseObservation(
            owner_is_live=False, missing_for_seconds=300,
        )),
        apply_crash_fence=_run_under_crash_fence,
        clock=lambda: now,
    )

    result = await svc._check_already_applied("run-replaced-attempt")

    assert result == AlreadyAppliedInfo(
        status="in_progress_recent", audit_id=734,
    )
    audit_repo.update_terminal.assert_not_awaited()


async def test_persisted_crash_classification_cannot_regress_after_marker_expiry():
    now = datetime(2026, 7, 15, 8, 5, tzinfo=timezone.utc)
    audit = SimpleNamespace(
        id=731,
        status="in_progress",
        started_at=now - timedelta(hours=25),
    )
    audit_repo = AsyncMock()
    audit_repo.find_latest_for_run = AsyncMock(return_value=audit)

    async def persist_terminal(_audit_id: int, **kwargs) -> bool:
        audit.status = kwargs["status"]
        return True

    audit_repo.update_terminal = AsyncMock(side_effect=persist_terminal)
    observer = AsyncMock(return_value=ApplyLeaseObservation(
        owner_is_live=False,
        missing_for_seconds=31,
    ))
    svc = CoordinatorRehydrateService(
        session_repository=AsyncMock(),
        envelope_store=AsyncMock(),
        audit_repository=audit_repo,
        apply_lease_observer=observer,
        apply_crash_fence=_run_under_crash_fence,
        clock=lambda: now,
    )

    first = await svc._check_already_applied("run-expired-marker")
    observer.reset_mock()
    observer.return_value = ApplyLeaseObservation(
        owner_is_live=False,
        missing_for_seconds=0,
    )
    second = await svc._check_already_applied("run-expired-marker")

    assert first.status == "crash_mid_apply"
    assert second.status == "crash_mid_apply"
    observer.assert_not_awaited()


@pytest.mark.parametrize("bad_first_missing", [float("nan"), float("inf"), float("-inf")])
async def test_malformed_reconcile_marker_fails_closed_as_in_progress(
    bad_first_missing: float,
) -> None:
    now = datetime(2026, 7, 15, 8, 5, tzinfo=timezone.utc)
    audit_repo = AsyncMock()
    audit_repo.update_terminal = AsyncMock(return_value=True)
    audit_repo.find_latest_for_run = AsyncMock(return_value=SimpleNamespace(
        id=730,
        status="in_progress",
        started_at=now - timedelta(hours=3),
    ))
    emitter = AsyncMock()
    svc = CoordinatorRehydrateService(
        session_repository=AsyncMock(),
        envelope_store=AsyncMock(),
        audit_repository=audit_repo,
        emit_event=emitter,
        apply_lease_observer=AsyncMock(return_value=ApplyLeaseObservation(
            owner_is_live=False, missing_for_seconds=bad_first_missing,
        )),
        clock=lambda: now,
    )

    result = await svc._check_already_applied("run-corrupt-marker")

    assert result.status == "in_progress_recent"
    emitter.assert_not_awaited()


async def test_reconcile_markers_are_isolated_by_coordinator_run_id():
    now = datetime(2026, 7, 15, 8, 5, tzinfo=timezone.utc)
    observer = AsyncMock(side_effect=[
        ApplyLeaseObservation(owner_is_live=False, missing_for_seconds=300),
        ApplyLeaseObservation(owner_is_live=False, missing_for_seconds=0),
    ])
    audit_repo = AsyncMock()
    audit_repo.update_terminal = AsyncMock(return_value=True)
    audit_repo.find_latest_for_run = AsyncMock(return_value=SimpleNamespace(
        id=74,
        status="in_progress",
        started_at=now - timedelta(hours=3),
    ))
    svc = CoordinatorRehydrateService(
        session_repository=AsyncMock(),
        envelope_store=AsyncMock(),
        audit_repository=audit_repo,
        apply_lease_observer=observer,
        apply_crash_fence=_run_under_crash_fence,
        clock=lambda: now,
    )

    old = await svc._check_already_applied("run-old")
    fresh = await svc._check_already_applied("run-new")

    assert old.status == "crash_mid_apply"
    assert fresh.status == "in_progress_recent"
    assert [call.args[0] for call in observer.await_args_list] == [
        "run-old", "run-new",
    ]


async def test_success_and_rollback_partial_do_not_probe_apply_lock():
    observer = AsyncMock(
        side_effect=AssertionError("terminal audit must not observe lease"),
    )
    audit_repo = AsyncMock()
    svc = CoordinatorRehydrateService(
        session_repository=AsyncMock(),
        envelope_store=AsyncMock(),
        audit_repository=audit_repo,
        apply_lease_observer=observer,
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
    observer.assert_not_awaited()


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
