"""C2b child-row reaper — integration tests (CI only; needs Postgres).

Run: cd api && SQLALCHEMY_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/manus_test \
     uv run pytest tests/integration/test_child_row_reaper.py -v

Covers the real SQL query ``find_running_mailbox_children`` and one end-to-end
sweep (row-lagged child with a persisted RESULT_READY → terminalized COMPLETED).
The orchestration branches are unit-tested in
tests/app/application/services/test_child_terminal_reconciler.py.
"""
from __future__ import annotations

import uuid as _uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from app.application.services.child_terminal_reconciler import (
    sweep_running_mailbox_children,
)
from app.domain.models.session import SessionStatus
from app.infrastructure.models.session import SessionModel
from app.infrastructure.models.user import UserModel
from app.infrastructure.repositories.db_session_repository import DBSessionRepository

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


async def _mk_user(db_session) -> str:
    uid = str(_uuid.uuid4())
    db_session.add(UserModel(id=uid, username=f"reaper_{uid[:8]}", password_hash="x"))
    await db_session.flush()
    return uid


async def _mk_root(db_session, uid: str) -> str:
    rid = f"root-{_uuid.uuid4().hex[:12]}"
    db_session.add(SessionModel(
        id=rid, user_id=uid, status="running", title="coord root",
        worker_type="root",
    ))
    await db_session.flush()
    return rid


def _child(*, sid, uid, parent, status="running", plane="mailbox",
           mode="foreground", run_id=None, wu_id=None, preset="coordinator_step"):
    is_terminal = status in {"completed", "timed_out"}
    return SessionModel(
        id=sid, user_id=uid, parent_session_id=parent, status=status,
        worker_type="subagent", subagent_control_plane=plane,
        execution_mode=mode, coordinator_run_id=run_id, work_unit_id=wu_id,
        tool_filter_preset=preset, title="child",
        execution_phase="terminated" if is_terminal else "running",
        terminal_reason="natural" if is_terminal else None,
        background_reason="explicit" if mode == "background" else None,
        expires_at=(
            datetime.now(timezone.utc) + timedelta(hours=1)
            if mode == "background"
            else None
        ),
    )


async def test_find_running_mailbox_children_selects_only_running_foreground(db_session):
    uid = await _mk_user(db_session)
    root = await _mk_root(db_session, uid)
    run_id = f"{root}:abc:a0"

    included = f"child-inc-{_uuid.uuid4().hex[:8]}"
    nul = f"nul-{_uuid.uuid4().hex[:8]}"
    bg = f"bg-{_uuid.uuid4().hex[:8]}"
    leg = f"leg-{_uuid.uuid4().hex[:8]}"
    done = f"done-{_uuid.uuid4().hex[:8]}"
    wait = f"wait-{_uuid.uuid4().hex[:8]}"
    tkp = f"tkp-{_uuid.uuid4().hex[:8]}"
    tko = f"tko-{_uuid.uuid4().hex[:8]}"
    db_session.add_all([
        _child(sid=included, uid=uid, parent=root, run_id=run_id, wu_id="wu.a0.0"),
        # INCLUDED: NULL-lineage running foreground mailbox subagent — the
        # query has NO lineage filter (ChildLineageRow fields are nullable;
        # the per-child match skips it, NOT the query). Uses the
        # non-coordinator preset to mirror reality.
        _child(sid=nul, uid=uid, parent=root, run_id=None, wu_id=None,
               preset="subagent_research"),
        # excluded: background turn (owned by reconcile_running_background_at_boot)
        _child(sid=bg, uid=uid, parent=root, mode="background", run_id=run_id, wu_id="wu.a0.1"),
        # excluded: non-mailbox (legacy plane)
        _child(sid=leg, uid=uid, parent=root, plane="legacy", run_id=run_id, wu_id="wu.a0.2"),
        # excluded: already terminal
        _child(sid=done, uid=uid, parent=root, status="completed", run_id=run_id, wu_id="wu.a0.3"),
        # excluded: live human-interaction states (not 'running')
        _child(sid=wait, uid=uid, parent=root, status="waiting", run_id=run_id, wu_id="wu.a0.4"),
        _child(sid=tkp, uid=uid, parent=root, status="takeover_pending", run_id=run_id, wu_id="wu.a0.5"),
        _child(sid=tko, uid=uid, parent=root, status="takeover", run_id=run_id, wu_id="wu.a0.6"),
    ])
    await db_session.flush()

    repo = DBSessionRepository(db_session=db_session)
    rows = await repo.find_running_mailbox_children()
    selected = {r.session_id for r in rows}

    assert included in selected
    assert sum(1 for r in rows if r.session_id == included) == 1
    inc_row = next(r for r in rows if r.session_id == included)
    assert inc_row.coordinator_run_id == run_id
    assert inc_row.work_unit_id == "wu.a0.0"
    # nullable-lineage contract: the NULL-lineage child IS returned, with Nones
    assert nul in selected
    nul_row = next(r for r in rows if r.session_id == nul)
    assert nul_row.coordinator_run_id is None
    assert nul_row.work_unit_id is None
    # every negative row + the root (worker_type='root') must be excluded
    for excluded in (bg, leg, done, wait, tkp, tko, root):
        assert excluded not in selected


async def test_sweep_terminalizes_row_lagged_child(async_session_factory, uow_factory):
    """End-to-end: a RUNNING coordinator child with a persisted RESULT_READY
    success envelope is terminalized to COMPLETED by the sweep.

    Uses ``async_session_factory`` for ALL DB access (NOT the ``db_session``
    fixture — that one yields inside an ``async with session.begin()`` block, so
    committing on it raises 'Can't operate on closed transaction'). The setup
    rows must be COMMITTED because ``terminalize_row`` opens its own
    ``uow_factory`` session (also bound to the test engine)."""
    from app.application.composition.graph_assembly import build_session_state_machine
    from app.infrastructure.repositories.db_coordinator_result_envelope_store_repository import (
        DbCoordinatorResultEnvelopeStoreRepository,
    )

    uid = str(_uuid.uuid4())
    root = f"root-{_uuid.uuid4().hex[:12]}"
    run_id = f"{root}:def:a0"
    wu_id = "wu.def.a0.0"
    child_id = f"child-e2e-{_uuid.uuid4().hex[:8]}"

    # Commit setup via a standalone (non-begin-wrapped) session.
    async with async_session_factory() as setup:
        setup.add(UserModel(id=uid, username=f"reaper_{uid[:8]}", password_hash="x"))
        await setup.flush()
        setup.add(SessionModel(id=root, user_id=uid, status="running",
                               title="coord root", worker_type="root"))
        await setup.flush()  # parent before child (self-FK)
        setup.add(_child(sid=child_id, uid=uid, parent=root, run_id=run_id, wu_id=wu_id))
        await setup.commit()

    try:
        store = DbCoordinatorResultEnvelopeStoreRepository(session_factory=async_session_factory)
        await store.persist_terminal(
            coordinator_run_id=run_id, work_unit_id=wu_id,
            child_session_id=child_id, envelope_type="RESULT_READY",
            payload={"outcome": "success"},
        )

        ssm = build_session_state_machine(uow_factory=uow_factory)
        async with async_session_factory() as read:
            repo = DBSessionRepository(db_session=read)
            stats = await sweep_running_mailbox_children(
                session_repo=repo, envelope_store=store,
                state_machine=ssm, uow_factory=uow_factory,
            )
        assert stats.terminalized == 1

        # Re-read on a fresh session to confirm the committed terminal write.
        async with async_session_factory() as verify:
            row = await DBSessionRepository(db_session=verify).get_by_id(child_id)
            assert row is not None
            assert row.status == SessionStatus.COMPLETED

        # Reopen-safety (spec §8, R1-P1 regression): a reaped child must
        # classify TERMINAL on rehydrate via its persisted envelope — never
        # the parallel_execution_subgraph.py:1075 limbo (limbo = expected
        # − terminal − pending − missing; wu in `terminal` ⇒ not limbo).
        # Direct service call — chat()-reopen exercises the same read path.
        from app.application.services.coordinator_rehydrate_service import (
            CoordinatorRehydrateService,
        )

        class _NoAuditRepo:
            async def find_latest_for_run(self, run_id):
                return None  # no apply-audit row for this run

        async with async_session_factory() as rh:
            svc = CoordinatorRehydrateService(
                session_repository=DBSessionRepository(db_session=rh),
                envelope_store=store,
                audit_repository=_NoAuditRepo(),
            )
            result = await svc.detect_existing_run(
                coordinator_run_id=run_id, parent_session_id=root,
            )
        assert result is not None
        assert wu_id in result.terminal      # classified via the envelope
        assert wu_id not in result.pending   # not re-dispatched, not limbo
    finally:
        # Child BEFORE parent — parent_session_id FK is ON DELETE RESTRICT.
        async with async_session_factory() as cleanup:
            await cleanup.execute(
                text("DELETE FROM coordinator_result_envelope_store WHERE coordinator_run_id = :r"),
                {"r": run_id},
            )
            await cleanup.execute(text("DELETE FROM sessions WHERE id = :c"), {"c": child_id})
            await cleanup.execute(text("DELETE FROM sessions WHERE id = :p"), {"p": root})
            await cleanup.execute(text("DELETE FROM users WHERE id = :u"), {"u": uid})
            await cleanup.commit()


async def test_sweep_r18_skips_no_envelope_child_preserving_respawn_trigger(
    async_session_factory, uow_factory
):
    """R18 system property (spec §8): root R with row-lagged child A (persisted
    envelope) AND child B whose envelope was never persisted (PEL-only at boot)
    → the sweep terminalizes A but does NOT touch B, so R still qualifies for
    ``reconcile_orphans``' supervisor re-spawn trigger
    (``find_running_mailbox_plane_root_ids`` keys on non-terminal child rows) —
    the re-spawned supervisor can then drain B's PEL envelope."""
    from app.application.composition.graph_assembly import build_session_state_machine
    from app.infrastructure.repositories.db_coordinator_result_envelope_store_repository import (
        DbCoordinatorResultEnvelopeStoreRepository,
    )

    uid = str(_uuid.uuid4())
    root = f"root-{_uuid.uuid4().hex[:12]}"
    run_id = f"{root}:r18:a0"
    wu_a, wu_b = "wu.r18.a0.0", "wu.r18.a0.1"
    child_a = f"child-a-{_uuid.uuid4().hex[:8]}"
    child_b = f"child-b-{_uuid.uuid4().hex[:8]}"

    async with async_session_factory() as setup:
        setup.add(UserModel(id=uid, username=f"reaper_{uid[:8]}", password_hash="x"))
        await setup.flush()
        setup.add(SessionModel(id=root, user_id=uid, status="running",
                               title="coord root", worker_type="root"))
        await setup.flush()
        setup.add(_child(sid=child_a, uid=uid, parent=root, run_id=run_id, wu_id=wu_a))
        setup.add(_child(sid=child_b, uid=uid, parent=root, run_id=run_id, wu_id=wu_b))
        await setup.commit()

    try:
        store = DbCoordinatorResultEnvelopeStoreRepository(session_factory=async_session_factory)
        # Only A's envelope is persisted; B's is "still in the PEL" (absent).
        await store.persist_terminal(
            coordinator_run_id=run_id, work_unit_id=wu_a,
            child_session_id=child_a, envelope_type="RESULT_READY",
            payload={"outcome": "success"},
        )

        ssm = build_session_state_machine(uow_factory=uow_factory)
        async with async_session_factory() as read:
            repo = DBSessionRepository(db_session=read)
            stats = await sweep_running_mailbox_children(
                session_repo=repo, envelope_store=store,
                state_machine=ssm, uow_factory=uow_factory,
            )
        assert stats.terminalized == 1  # A only
        assert stats.skipped >= 1       # B skipped (no envelope)

        async with async_session_factory() as verify:
            vrepo = DBSessionRepository(db_session=verify)
            row_a = await vrepo.get_by_id(child_a)
            row_b = await vrepo.get_by_id(child_b)
            assert row_a is not None and row_a.status == SessionStatus.COMPLETED
            assert row_b is not None and row_b.status == SessionStatus.RUNNING
            # THE R18 property: B's non-terminal row keeps R in the re-spawn set.
            respawn_roots = await vrepo.find_running_mailbox_plane_root_ids()
            assert root in respawn_roots

        # Eventual consistency (spec §4.2/§5.3): the re-spawned supervisor's
        # drain later persists B's envelope; the NEXT sweep matches it.
        # Simulate the drain's persist, then run the sweep again.
        await store.persist_terminal(
            coordinator_run_id=run_id, work_unit_id=wu_b,
            child_session_id=child_b, envelope_type="CANCEL_ACK",
            payload={"final_state": "force_terminated"},
        )
        async with async_session_factory() as read2:
            stats2 = await sweep_running_mailbox_children(
                session_repo=DBSessionRepository(db_session=read2),
                envelope_store=store,
                state_machine=ssm, uow_factory=uow_factory,
            )
        assert stats2.terminalized == 1  # B, this time
        async with async_session_factory() as verify2:
            row_b2 = await DBSessionRepository(db_session=verify2).get_by_id(child_b)
            assert row_b2 is not None
            # force_terminated derives the TIMED_OUT/watchdog_timeout pair
            assert row_b2.status == SessionStatus.TIMED_OUT
    finally:
        async with async_session_factory() as cleanup:
            await cleanup.execute(
                text("DELETE FROM coordinator_result_envelope_store WHERE coordinator_run_id = :r"),
                {"r": run_id},
            )
            await cleanup.execute(
                text("DELETE FROM sessions WHERE id IN (:a, :b)"),
                {"a": child_a, "b": child_b},
            )
            await cleanup.execute(text("DELETE FROM sessions WHERE id = :p"), {"p": root})
            await cleanup.execute(text("DELETE FROM users WHERE id = :u"), {"u": uid})
            await cleanup.commit()
