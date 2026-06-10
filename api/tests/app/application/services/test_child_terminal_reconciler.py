"""Unit tests for the C2b child-row reaper (child_terminal_reconciler).

Spec: docs/superpowers/specs/2026-06-09-c2b-child-row-reaper-design.md §8.
Pure / fake-driven — no DB. The SQL query ``find_running_mailbox_children`` is
exercised separately in tests/integration/test_child_row_reaper.py (CI only).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.application.services.child_terminal_reconciler import (
    row_terminal_from_envelope,
)
from app.domain.models.session import SessionStatus

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.parametrize(
    "envelope_type, payload, expected",
    [
        ("RESULT_READY", {"outcome": "timed_out"}, (SessionStatus.TIMED_OUT, "watchdog_timeout")),
        ("RESULT_READY", {"outcome": "success"}, (SessionStatus.COMPLETED, "natural")),
        ("RESULT_READY", {"outcome": "failed"}, (SessionStatus.COMPLETED, "natural")),
        ("RESULT_READY", {"outcome": "cancelled"}, (SessionStatus.COMPLETED, "natural")),
        ("RESULT_READY", {"outcome": "needs_authorization"}, (SessionStatus.COMPLETED, "natural")),
        ("CANCEL_ACK", {"final_state": "force_terminated"}, (SessionStatus.TIMED_OUT, "watchdog_timeout")),
        ("CANCEL_ACK", {"final_state": "cancelled"}, (SessionStatus.COMPLETED, "natural")),
        ("CANCEL_ACK", {"final_state": "completed"}, (SessionStatus.COMPLETED, "natural")),
        # totality: missing / null / unknown fields default to COMPLETED/natural
        ("RESULT_READY", {}, (SessionStatus.COMPLETED, "natural")),
        ("RESULT_READY", None, (SessionStatus.COMPLETED, "natural")),
        ("CANCEL_ACK", {"final_state": None}, (SessionStatus.COMPLETED, "natural")),
        ("SOME_FUTURE_TYPE", {"outcome": "timed_out"}, (SessionStatus.COMPLETED, "natural")),
    ],
)
def test_row_terminal_from_envelope(envelope_type, payload, expected):
    assert row_terminal_from_envelope(envelope_type, payload) == expected


from app.application.services.child_terminal_reconciler import terminalize_row


# ─── shared fakes ──────────────────────────────────────────────────────── #

class _FakeStateMachine:
    """Records terminate() calls; returns a configurable CAS bool or raises."""

    def __init__(self, returns: bool = True) -> None:
        self.returns = returns
        self.raise_exc: BaseException | None = None
        self.calls: list[tuple] = []

    async def terminate(self, session_id, to, terminal_reason, *, session_repo):
        self.calls.append((session_id, to, terminal_reason, session_repo))
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.returns


class _FakeDbSession:
    def __init__(self) -> None:
        self.commits = 0

    async def commit(self) -> None:
        self.commits += 1


class _FakeUoW:
    def __init__(self) -> None:
        self.session = object()  # ssm.terminate's session_repo arg
        self.db_session = _FakeDbSession()
        self.entered = False

    async def __aenter__(self) -> "_FakeUoW":
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None  # propagate any exception (matches DBUnitOfWork success path)


def _uow_factory_fake(uow: "_FakeUoW"):
    def _factory() -> "_FakeUoW":
        return uow
    return _factory


# ─── terminalize_row ───────────────────────────────────────────────────── #

async def test_terminalize_row_cas_won():
    sm = _FakeStateMachine(returns=True)
    uow = _FakeUoW()
    ok = await terminalize_row(
        "child-1", SessionStatus.COMPLETED, "natural",
        state_machine=sm, uow_factory=_uow_factory_fake(uow),
    )
    assert ok is True
    assert sm.calls == [("child-1", SessionStatus.COMPLETED, "natural", uow.session)]
    assert uow.db_session.commits == 1


async def test_terminalize_row_cas_lost():
    sm = _FakeStateMachine(returns=False)
    uow = _FakeUoW()
    ok = await terminalize_row(
        "child-1", SessionStatus.TIMED_OUT, "watchdog_timeout",
        state_machine=sm, uow_factory=_uow_factory_fake(uow),
    )
    assert ok is False
    assert len(sm.calls) == 1
    assert uow.db_session.commits == 1


async def test_terminalize_row_missing_state_machine_noop():
    uow = _FakeUoW()
    ok = await terminalize_row(
        "child-1", SessionStatus.COMPLETED, "natural",
        state_machine=None, uow_factory=_uow_factory_fake(uow),
    )
    assert ok is False
    assert uow.entered is False  # no UoW opened


async def test_terminalize_row_missing_uow_factory_noop():
    sm = _FakeStateMachine(returns=True)
    ok = await terminalize_row(
        "child-1", SessionStatus.COMPLETED, "natural",
        state_machine=sm, uow_factory=None,
    )
    assert ok is False
    assert sm.calls == []


async def test_terminalize_row_db_error_propagates():
    sm = _FakeStateMachine()
    sm.raise_exc = ValueError("db hiccup")
    uow = _FakeUoW()
    with pytest.raises(ValueError):
        await terminalize_row(
            "child-1", SessionStatus.COMPLETED, "natural",
            state_machine=sm, uow_factory=_uow_factory_fake(uow),
        )


async def test_terminalize_row_cancelled_propagates():
    sm = _FakeStateMachine()
    sm.raise_exc = asyncio.CancelledError()
    uow = _FakeUoW()
    with pytest.raises(asyncio.CancelledError):
        await terminalize_row(
            "child-1", SessionStatus.COMPLETED, "natural",
            state_machine=sm, uow_factory=_uow_factory_fake(uow),
        )


from app.application.services.child_terminal_reconciler import (
    match_terminal_envelope_to_row,
)


class _FakeEnvelope:
    def __init__(self, *, work_unit_id, child_session_id, envelope_type, payload):
        self.work_unit_id = work_unit_id
        self.child_session_id = child_session_id
        self.envelope_type = envelope_type
        self.payload = payload


class _FakeEnvelopeStore:
    def __init__(self, envelopes_by_run: dict[str, list]):
        self._by_run = envelopes_by_run
        self.calls: list[str] = []

    async def find_terminal_envelopes_by_run(self, run_id):
        self.calls.append(run_id)
        return list(self._by_run.get(run_id, []))


class _FakeSessionRepo:
    """get_by_id returns a stub row (or raises if the stored value is an exc)."""

    def __init__(self, rows_by_id: dict, children: list | None = None):
        self._rows = rows_by_id
        self._children = children or []

    async def get_by_id(self, session_id):
        val = self._rows.get(session_id)
        if isinstance(val, BaseException):
            raise val
        return val

    async def find_running_mailbox_children(self):
        return list(self._children)


def _row(*, status=SessionStatus.RUNNING, execution_mode="foreground", **extra):
    """Minimal stand-in for the domain Session read contract the reaper uses
    (``.status``, ``.execution_mode``). Extra attrs (e.g. ``updated_at``) prove
    the reaper ignores fields it must not depend on."""
    return SimpleNamespace(status=status, execution_mode=execution_mode, **extra)


async def test_match_terminalizes_to_completed():
    sm = _FakeStateMachine(returns=True)
    uow = _FakeUoW()
    repo = _FakeSessionRepo({"c1": _row()})
    store = _FakeEnvelopeStore({
        "run-1": [_FakeEnvelope(work_unit_id="wu-1", child_session_id="c1",
                                envelope_type="RESULT_READY", payload={"outcome": "success"})],
    })
    did = await match_terminal_envelope_to_row(
        "c1", "run-1", "wu-1",
        state_machine=sm, uow_factory=_uow_factory_fake(uow),
        envelope_store=store, session_repo=repo,
    )
    assert did is True
    assert sm.calls[0][1] == SessionStatus.COMPLETED
    assert sm.calls[0][2] == "natural"


async def test_match_terminalizes_to_timed_out():
    sm = _FakeStateMachine(returns=True)
    repo = _FakeSessionRepo({"c1": _row()})
    store = _FakeEnvelopeStore({
        "run-1": [_FakeEnvelope(work_unit_id="wu-1", child_session_id="c1",
                                envelope_type="RESULT_READY", payload={"outcome": "timed_out"})],
    })
    did = await match_terminal_envelope_to_row(
        "c1", "run-1", "wu-1",
        state_machine=sm, uow_factory=_uow_factory_fake(_FakeUoW()),
        envelope_store=store, session_repo=repo,
    )
    assert did is True
    assert sm.calls[0][1] == SessionStatus.TIMED_OUT
    assert sm.calls[0][2] == "watchdog_timeout"


async def test_match_no_matching_envelope_skips():
    sm = _FakeStateMachine()
    repo = _FakeSessionRepo({"c1": _row()})
    store = _FakeEnvelopeStore({"run-1": []})  # nothing persisted yet
    did = await match_terminal_envelope_to_row(
        "c1", "run-1", "wu-1",
        state_machine=sm, uow_factory=_uow_factory_fake(_FakeUoW()),
        envelope_store=store, session_repo=repo,
    )
    assert did is False
    assert sm.calls == []  # NO backfill, NO row write


async def test_match_already_terminal_noop_without_store_lookup():
    sm = _FakeStateMachine()
    repo = _FakeSessionRepo({"c1": _row(status=SessionStatus.COMPLETED)})
    store = _FakeEnvelopeStore({})
    did = await match_terminal_envelope_to_row(
        "c1", "run-1", "wu-1",
        state_machine=sm, uow_factory=_uow_factory_fake(_FakeUoW()),
        envelope_store=store, session_repo=repo,
    )
    assert did is False
    assert store.calls == []  # short-circuits before querying the store
    assert sm.calls == []


async def test_match_row_missing_noop():
    sm = _FakeStateMachine()
    repo = _FakeSessionRepo({"c1": None})
    did = await match_terminal_envelope_to_row(
        "c1", "run-1", "wu-1",
        state_machine=sm, uow_factory=_uow_factory_fake(_FakeUoW()),
        envelope_store=_FakeEnvelopeStore({}), session_repo=repo,
    )
    assert did is False
    assert sm.calls == []


async def test_match_background_execution_mode_skips():
    sm = _FakeStateMachine()
    store = _FakeEnvelopeStore({})
    repo = _FakeSessionRepo({"c1": _row(execution_mode="background")})
    did = await match_terminal_envelope_to_row(
        "c1", "run-1", "wu-1",
        state_machine=sm, uow_factory=_uow_factory_fake(_FakeUoW()),
        envelope_store=store, session_repo=repo,
    )
    assert did is False
    assert store.calls == []  # background turn owned by reconcile_running_background_at_boot
    assert sm.calls == []


async def test_match_null_lineage_skips():
    sm = _FakeStateMachine()
    store = _FakeEnvelopeStore({})
    repo = _FakeSessionRepo({"c1": _row()})
    did = await match_terminal_envelope_to_row(
        "c1", None, None,  # non-coordinator mailbox subagent
        state_machine=sm, uow_factory=_uow_factory_fake(_FakeUoW()),
        envelope_store=store, session_repo=repo,
    )
    assert did is False
    assert store.calls == []
    assert sm.calls == []


async def test_match_identity_mismatch_skips():
    sm = _FakeStateMachine()
    repo = _FakeSessionRepo({"c1": _row()})
    store = _FakeEnvelopeStore({
        # right wu, WRONG child_session_id → must not match (defensive identity check)
        "run-1": [_FakeEnvelope(work_unit_id="wu-1", child_session_id="OTHER",
                                envelope_type="RESULT_READY", payload={"outcome": "success"})],
    })
    did = await match_terminal_envelope_to_row(
        "c1", "run-1", "wu-1",
        state_machine=sm, uow_factory=_uow_factory_fake(_FakeUoW()),
        envelope_store=store, session_repo=repo,
    )
    assert did is False
    assert sm.calls == []


async def test_match_work_unit_mismatch_skips():
    sm = _FakeStateMachine()
    repo = _FakeSessionRepo({"c1": _row()})
    store = _FakeEnvelopeStore({
        "run-1": [_FakeEnvelope(work_unit_id="OTHER_WU", child_session_id="c1",
                                envelope_type="RESULT_READY", payload={"outcome": "success"})],
    })
    did = await match_terminal_envelope_to_row(
        "c1", "run-1", "wu-1",
        state_machine=sm, uow_factory=_uow_factory_fake(_FakeUoW()),
        envelope_store=store, session_repo=repo,
    )
    assert did is False
    assert sm.calls == []


async def test_match_no_timestamp_guard_regression():
    """R22 lock-in: a genuine zombie whose row carries a LATER ``updated_at``
    than the envelope's ``received_at`` (sandbox-destroy bumped it post-envelope)
    is STILL matched + terminalized. The reaper must not reintroduce any
    timestamp/epoch guard."""
    sm = _FakeStateMachine(returns=True)
    repo = _FakeSessionRepo({"c1": _row(updated_at="2999-01-01T00:00:00")})
    store = _FakeEnvelopeStore({
        "run-1": [_FakeEnvelope(work_unit_id="wu-1", child_session_id="c1",
                                envelope_type="RESULT_READY", payload={"outcome": "success"})],
    })
    did = await match_terminal_envelope_to_row(
        "c1", "run-1", "wu-1",
        state_machine=sm, uow_factory=_uow_factory_fake(_FakeUoW()),
        envelope_store=store, session_repo=repo,
    )
    assert did is True  # matched despite the late updated_at — no timestamp guard


from app.application.services.child_terminal_reconciler import (
    SweepStats,
    sweep_running_mailbox_children,
)
from app.domain.repositories.session_repository import ChildLineageRow


async def test_sweep_counts_terminalize_and_skips():
    sm = _FakeStateMachine(returns=True)
    children = [
        ChildLineageRow("c1", "run-1", "wu-1"),  # has envelope → terminalize
        ChildLineageRow("c2", "run-1", "wu-2"),  # no envelope → skip
        ChildLineageRow("c3", "run-1", "wu-3"),  # already terminal → skip
    ]
    rows = {
        "c1": _row(),
        "c2": _row(),
        "c3": _row(status=SessionStatus.TIMED_OUT),
    }
    repo = _FakeSessionRepo(rows, children=children)
    store = _FakeEnvelopeStore({
        "run-1": [_FakeEnvelope(work_unit_id="wu-1", child_session_id="c1",
                                envelope_type="RESULT_READY", payload={"outcome": "success"})],
    })
    stats = await sweep_running_mailbox_children(
        session_repo=repo, envelope_store=store,
        state_machine=sm, uow_factory=_uow_factory_fake(_FakeUoW()),
    )
    assert isinstance(stats, SweepStats)
    assert (stats.scanned, stats.terminalized, stats.skipped, stats.errored) == (3, 1, 2, 0)


async def test_sweep_inner_try_isolates_bad_child():
    sm = _FakeStateMachine(returns=True)
    children = [
        ChildLineageRow("bad", "run-1", "wu-x"),  # get_by_id raises
        ChildLineageRow("c1", "run-1", "wu-1"),   # still processed → terminalize
    ]
    rows = {"bad": ValueError("boom"), "c1": _row()}
    repo = _FakeSessionRepo(rows, children=children)
    store = _FakeEnvelopeStore({
        "run-1": [_FakeEnvelope(work_unit_id="wu-1", child_session_id="c1",
                                envelope_type="RESULT_READY", payload={"outcome": "success"})],
    })
    stats = await sweep_running_mailbox_children(
        session_repo=repo, envelope_store=store,
        state_machine=sm, uow_factory=_uow_factory_fake(_FakeUoW()),
    )
    assert (stats.scanned, stats.terminalized, stats.errored) == (2, 1, 1)


async def test_sweep_cancelled_propagates():
    children = [ChildLineageRow("cancel", "run-1", "wu-x")]
    repo = _FakeSessionRepo({"cancel": asyncio.CancelledError()}, children=children)
    with pytest.raises(asyncio.CancelledError):
        await sweep_running_mailbox_children(
            session_repo=repo, envelope_store=_FakeEnvelopeStore({}),
            state_machine=_FakeStateMachine(), uow_factory=_uow_factory_fake(_FakeUoW()),
        )


async def test_sweep_query_failure_propagates_to_caller():
    """find_running_mailbox_children failure propagates so main.py's OUTER
    best-effort try (not the inner per-child try) swallows it."""
    class _BoomRepo:
        async def find_running_mailbox_children(self):
            raise RuntimeError("query down")

    with pytest.raises(RuntimeError):
        await sweep_running_mailbox_children(
            session_repo=_BoomRepo(), envelope_store=_FakeEnvelopeStore({}),
            state_machine=_FakeStateMachine(), uow_factory=_uow_factory_fake(_FakeUoW()),
        )
