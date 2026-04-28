"""B4 Issue 1D: write_session_degraded_marker contract.

Spec: docs/superpowers/specs/2026-04-27-b4-1d-finishing-drain-design.md §3.2

Verifies the session-level degraded sentinel writer:
1. Builds a CostRecord with stable uuid5 run_id, zero-tokens, UNKNOWN status.
2. Idempotent under same (session_id, reason).
3. Different reasons produce distinct rows.
4. Persister failure → returns False, no recursive marker write.
5. Soft-bound timeout abandons the inner task to _PENDING_MARKER_TASKS.
"""

from __future__ import annotations

import asyncio
import uuid
from decimal import Decimal
from typing import Awaitable, Callable, List

import pytest

from app.domain.models.cost_record import CostRecord, CostStatus
from app.domain.services.cost_callback_handler import (
    _DRAIN_DEGRADED_NAMESPACE,
    _MARKER_WRITE_TIMEOUT_SECONDS,
    _PENDING_MARKER_TASKS,
    CostCallbackHandler,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_capture_persister() -> tuple[
    Callable[[CostRecord], Awaitable[None]], List[CostRecord]
]:
    captured: List[CostRecord] = []

    async def persist(record: CostRecord) -> None:
        captured.append(record)

    return persist, captured


class TestBuildSessionDegradedRecord:
    def test_record_shape(self) -> None:
        persist, _captured = _make_capture_persister()
        handler = CostCallbackHandler(
            session_id="sess-X", user_id="user-Y", persister=persist
        )

        record = handler._build_session_degraded_record(reason="drain_timeout")

        # uuid5 stable key: same inputs → same uuid.
        expected_run_id = str(uuid.uuid5(
            _DRAIN_DEGRADED_NAMESPACE, "sess-X:terminal-drain:drain_timeout"
        ))
        assert record.run_id == expected_run_id
        assert record.session_id == "sess-X"
        assert record.user_id == "user-Y"
        assert record.node_name == "persist_degraded"
        assert record.cost_status == CostStatus.UNKNOWN
        assert record.total_usd == Decimal(0)
        assert record.input_tokens == 0
        assert record.output_tokens == 0
        assert record.cache_read_tokens == 0
        assert record.cache_write_tokens == 0
        assert record.reasoning_tokens == 0
        # model / provider must NOT be empty strings — aggregation rolls
        # every row into ``by_model[r.model]`` / ``by_provider[r.provider]``
        # so an empty key would leak as ``{"": "0"}`` into GET /cost.
        assert record.model == "session_degraded_marker"
        assert record.provider == "internal"
        assert record.model != "" and record.provider != "", (
            "Sentinel marker labels must be non-empty to avoid polluting "
            "by_model / by_provider breakdown with empty keys."
        )
        # created_at must be tz-aware (CostRecord __post_init__ enforces).
        assert record.created_at.tzinfo is not None

    def test_record_uuid_deterministic_across_calls(self) -> None:
        persist, _captured = _make_capture_persister()
        handler = CostCallbackHandler(
            session_id="sess-Z", user_id="u", persister=persist
        )

        a = handler._build_session_degraded_record(reason="drain_timeout")
        b = handler._build_session_degraded_record(reason="drain_timeout")
        assert a.run_id == b.run_id, (
            "Same (session_id, reason) must produce identical run_id so "
            "ON CONFLICT DO NOTHING dedups idempotent writes."
        )

    def test_record_uuid_distinct_per_reason(self) -> None:
        persist, _captured = _make_capture_persister()
        handler = CostCallbackHandler(
            session_id="sess-Z", user_id="u", persister=persist
        )

        a = handler._build_session_degraded_record(reason="drain_timeout")
        b = handler._build_session_degraded_record(reason="drain_persist_failures")
        assert a.run_id != b.run_id


class TestWriteMarker:
    async def test_happy_path_returns_true_and_persists(self) -> None:
        persist, captured = _make_capture_persister()
        handler = CostCallbackHandler(
            session_id="s", user_id="u", persister=persist
        )

        ok = await handler.write_session_degraded_marker(reason="drain_timeout")

        assert ok is True
        assert len(captured) == 1
        rec = captured[0]
        assert rec.node_name == "persist_degraded"
        assert rec.cost_status == CostStatus.UNKNOWN
        assert rec.total_usd == Decimal(0)

    async def test_idempotent_run_id_under_same_reason(self) -> None:
        """Two writes with same (session_id, reason) → same run_id.

        Prod DB has ``ON CONFLICT DO NOTHING`` on run_id, so the second
        insert is a no-op at the DB layer. This test verifies only the
        ``run_id`` determinism — the DB-level unique-index dedup itself is
        NOT exercised here (or in Task 21, which uses a list-backed
        persister that bypasses the repository's ``pg_insert(...)
        .on_conflict_do_nothing(...)`` path). DB-level conflict behavior
        is the responsibility of the existing ``DbCostRecordRepository``
        ORM tests; if those don't already pin it, add a separate
        repository-level test as a follow-up — out of scope for this plan.
        """
        persist, captured = _make_capture_persister()
        handler = CostCallbackHandler(
            session_id="s", user_id="u", persister=persist
        )

        ok1 = await handler.write_session_degraded_marker(reason="drain_timeout")
        ok2 = await handler.write_session_degraded_marker(reason="drain_timeout")

        assert ok1 is True and ok2 is True
        assert len(captured) == 2
        assert captured[0].run_id == captured[1].run_id, (
            "Idempotent writes must share run_id so DB ON CONFLICT dedups."
        )

    async def test_distinct_run_id_across_reasons(self) -> None:
        persist, captured = _make_capture_persister()
        handler = CostCallbackHandler(
            session_id="s", user_id="u", persister=persist
        )

        await handler.write_session_degraded_marker(reason="drain_timeout")
        await handler.write_session_degraded_marker(reason="drain_persist_failures")
        await handler.write_session_degraded_marker(reason="drain_exception")

        assert len({rec.run_id for rec in captured}) == 3, (
            "Different reasons must produce distinct run_id so each is "
            "operator-triagable; aggregation collapses them anyway."
        )
        # All three rows share node_name → aggregation override fires.
        assert all(rec.node_name == "persist_degraded" for rec in captured)

    async def test_persister_raises_returns_false_no_recursion(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Spec §3.2: persister raise → returns False, NO marker-of-marker.

        We assert exactly ONE persister invocation; if the writer recursed,
        the count would be >=2.
        """
        call_count = 0

        async def failing_persist(record: CostRecord) -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("simulated persister failure")

        handler = CostCallbackHandler(
            session_id="s", user_id="u", persister=failing_persist
        )

        with caplog.at_level("WARNING"):
            ok = await handler.write_session_degraded_marker(reason="drain_timeout")

        assert ok is False
        assert call_count == 1, (
            f"Expected exactly one persister call (no recursive marker "
            f"write); saw {call_count}"
        )
        # Exactly one warning logged for the failed marker write.
        marker_warnings = [
            r for r in caplog.records
            if "marker write failed" in r.getMessage()
        ]
        assert len(marker_warnings) == 1

    async def test_soft_bound_timeout_parks_task(self) -> None:
        """Spec §3.2: persister hangs past _MARKER_WRITE_TIMEOUT_SECONDS →
        returns False AND parks the abandoned task in _PENDING_MARKER_TASKS
        with a done callback installed.

        Bound is "time to RETURN False", not "hard upper bound on persister
        wall time" — the abandoned inner task continues in the background.
        """
        block_event = asyncio.Event()
        persist_calls: List[CostRecord] = []

        async def hanging_persist(record: CostRecord) -> None:
            persist_calls.append(record)
            await block_event.wait()

        handler = CostCallbackHandler(
            session_id="s", user_id="u", persister=hanging_persist
        )

        # Snapshot registry size before so other tests' parked tasks
        # don't leak into our assertion.
        registry_before = len(_PENDING_MARKER_TASKS)

        ok = await handler.write_session_degraded_marker(reason="drain_timeout")

        assert ok is False
        assert len(persist_calls) == 1  # persister was invoked once
        assert len(_PENDING_MARKER_TASKS) == registry_before + 1, (
            "Abandoned task must be parked in _PENDING_MARKER_TASKS as a "
            "GC anchor + observability handle."
        )

        # Cleanup: release the persister so the parked task settles, then
        # let the done callback fire to remove it from the registry.
        block_event.set()
        # Give the event loop a tick to run the persister + done callback.
        for _ in range(20):
            await asyncio.sleep(0.01)
            if len(_PENDING_MARKER_TASKS) == registry_before:
                break
        assert len(_PENDING_MARKER_TASKS) == registry_before, (
            "Done callback must remove the task from registry on completion."
        )

    async def test_cancelled_persister_returns_false_not_raises(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Spec §3.2 + cancellation guard: when the marker task ends up
        cancelled (persister itself raised ``asyncio.CancelledError``, or
        an underlying DB driver propagated cancellation), the writer must
        return ``False`` and NOT propagate ``CancelledError`` to the caller.

        Without the ``if task.cancelled():`` guard, ``task.exception()``
        would re-raise ``CancelledError`` per asyncio semantics — that
        would crash out of ``write_session_degraded_marker``, violate the
        documented "False on any other error" contract, AND skip the
        subsequent status write in ``_set_terminal_status._terminal_op``.
        """

        async def cancelling_persist(record: CostRecord) -> None:
            # Self-raised CancelledError puts the task in CANCELLED state
            # (super().cancel() inside Task.__step). task.cancelled() is
            # then True, and task.exception() would re-raise it.
            raise asyncio.CancelledError("simulated marker cancel")

        handler = CostCallbackHandler(
            session_id="s", user_id="u", persister=cancelling_persist
        )

        with caplog.at_level("WARNING"):
            ok = await handler.write_session_degraded_marker(
                reason="drain_timeout"
            )

        assert ok is False, (
            "Cancelled marker task must return False — NOT propagate "
            "CancelledError to the caller."
        )
        cancel_warnings = [
            r for r in caplog.records
            if "marker write task cancelled" in r.getMessage()
        ]
        assert len(cancel_warnings) == 1, (
            f"Expected exactly one 'marker write task cancelled' warning, "
            f"saw {len(cancel_warnings)}"
        )
