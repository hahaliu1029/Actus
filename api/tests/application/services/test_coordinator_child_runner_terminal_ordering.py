"""C2 PR-9b-B Task B10 — INV-B6 terminal-ordering regression LOCK.

Spec ref: §INV-B6 (spec line 516) — the child terminal envelope
(``RESULT_READY``) MUST be published AFTER the cost flush / degraded-marker
write. A future PR-10 refactor must not silently invert that order (e.g. by
moving the publish ahead of the drain to shave latency), which would let the
parent observe a completion whose cost rollup is still in flight / never
persisted.

Where the ordering ACTUALLY lives (re-derived in Task B10 Step 0 — the plan
snippet's claim that it lives in ``CoordinatorChildRunner._finalize_success``
paired with a cost flush is WRONG: that finalizer publishes RESULT_READY but
never flushes cost):

    AgentTaskRunner._set_terminal_status._terminal_op  (shielded body)
      1. cost_handler.flush_pending(timeout=3.0)             ← cost_flush
      2. cost_handler.write_session_degraded_marker(reason)  ← degraded_marker
         (conditional: drain timeout OR persist failures)
      3. fresh-UoW status write + commit
      4. _maybe_stop_mailbox_supervisor()
      5. _maybe_stop_child_publisher(status, reason)
            └─ publisher.publish(RESULT_READY envelope)      ← result_ready_publish

Both the flush/marker AND the publish run inside the SAME ``asyncio.shield``-ed
``_terminal_op`` — so their relative order is a hard, observable contract, not
an emergent timing artifact.

This test drives the real ``_set_terminal_status`` body (no production code
change) and records the call order of cost flush vs. RESULT_READY publish via a
shared list. It mirrors the harness in
``tests/domain/services/test_agent_task_runner_terminal_shield.py``
(``object.__new__`` bypass + per-attribute assignment) and the publisher-gate
attribute set in
``tests/domain/services/test_agent_task_runner_terminal_publisher_flag.py``.

The success-path test is the load-bearing INV-B6 lock. The degraded-marker
variant is marked ``xfail`` per the plan (the marker-before-publish ordering is
deferred to the PR-10 refactor; locking it now would either be a tautology over
the same ``_terminal_op`` body or require asserting against code that isn't
restructured yet).
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.domain.models.mailbox_envelope import MailboxEnvelopeType
from app.domain.models.session import SessionStatus
from app.domain.services.agent_task_runner import (
    _PENDING_TERMINAL_TASKS,
    AgentTaskRunner,
)
from app.domain.services.cost_callback_handler import (
    CostCallbackHandler,
    FlushResult,
)


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------------
# Harness — mirror test_agent_task_runner_terminal_shield._make_runner_for_terminal
# plus the publisher-gate attribute set so _maybe_stop_child_publisher actually
# reaches publisher.publish (otherwise the publish is silently skipped and the
# ordering assertion would be vacuous).
# ---------------------------------------------------------------------------

def _make_uow_factory() -> Any:
    """Yields a MagicMock-backed UoW supporting ``async with`` + commit."""
    yielded_uows: List[Any] = []

    @asynccontextmanager
    async def _ctx():
        uow = MagicMock(name=f"fresh-uow-{len(yielded_uows)}")
        uow.session = MagicMock()
        uow.session.update_to_terminal = AsyncMock(return_value=None)
        uow.db_session = MagicMock()
        uow.db_session.commit = AsyncMock(return_value=None)
        yielded_uows.append(uow)
        yield uow

    factory = MagicMock(side_effect=_ctx)
    factory.yielded_uows = yielded_uows  # type: ignore[attr-defined]
    return factory


def _make_recording_publisher(order: List[str]) -> AsyncMock:
    """Publisher whose ``publish`` appends ``"result_ready_publish"`` to
    ``order`` ONLY for a RESULT_READY envelope (a CANCEL_ACK or any other type
    is intentionally not recorded — INV-B6 is about the RESULT_READY terminal)."""
    publisher = AsyncMock()

    async def _publish(envelope: Any) -> None:
        if getattr(envelope, "type", None) == MailboxEnvelopeType.RESULT_READY:
            order.append("result_ready_publish")

    publisher.publish = AsyncMock(side_effect=_publish)
    return publisher


def _make_recording_cost_handler(
    order: List[str],
    *,
    flush_result: FlushResult,
    marker_returns: bool = True,
) -> MagicMock:
    """Cost handler whose ``flush_pending`` records ``"cost_flush"`` and whose
    ``write_session_degraded_marker`` records ``"degraded_marker"``."""
    cost_handler = MagicMock(spec=CostCallbackHandler)

    async def _flush(timeout: float) -> FlushResult:
        del timeout
        order.append("cost_flush")
        return flush_result

    async def _marker(reason: str) -> bool:
        del reason
        order.append("degraded_marker")
        return marker_returns

    cost_handler.flush_pending = AsyncMock(side_effect=_flush)
    cost_handler.write_session_degraded_marker = AsyncMock(side_effect=_marker)
    return cost_handler


def _make_runner_for_terminal_ordering(
    cost_handler: Any,
    uow_factory: Any,
    publisher: AsyncMock,
    *,
    session_id: str = "child-B10",
) -> AgentTaskRunner:
    """Construct an AgentTaskRunner via ``object.__new__`` (bypassing the ~25
    real collaborators), assign the attributes the terminal path reads, and
    stub the async sub-helpers (``_is_mailbox_plane_child`` /
    ``_cleanup_heartbeat_task`` / ``_is_root_session``) so the RESULT_READY
    publish path is reached without DB/heartbeat plumbing."""
    runner = object.__new__(AgentTaskRunner)
    # --- _terminal_op core (mirrors terminal_shield harness) ---
    runner._session_id = session_id
    runner._cost_callback_handler = cost_handler
    runner._uow_factory = uow_factory
    runner._on_session_complete = None
    # --- _maybe_stop_child_publisher gate (mirrors terminal_publisher_flag) ---
    runner._terminal_envelope_publisher_disabled = False
    runner._mailbox_publisher = publisher
    runner._supervisor_registry = MagicMock()
    runner._spawn_correlation_id = "spawn:child-B10"
    runner._runner_exception_terminal = False
    cached_session = MagicMock()
    cached_session.parent_session_id = "parent-B10"
    runner._cached_session_for_publisher = cached_session
    # async helpers stubbed so the publish path is reached deterministically.
    runner._is_mailbox_plane_child = AsyncMock(return_value=True)
    runner._cleanup_heartbeat_task = AsyncMock()
    # _maybe_stop_mailbox_supervisor reads _supervisor_registry + _is_root_session.
    runner._is_root_session = AsyncMock(return_value=False)
    return runner


# ---------------------------------------------------------------------------
# INV-B6 load-bearing lock — success path: flush BEFORE RESULT_READY publish.
# ---------------------------------------------------------------------------

async def test_cost_flush_precedes_result_ready_publish() -> None:
    """[INV-B6 spec:516] On a clean (drained) terminal, the cost flush MUST
    run BEFORE the RESULT_READY terminal envelope is published.

    Drives the real ``AgentTaskRunner._set_terminal_status._terminal_op`` with
    ``status=COMPLETED`` + ``terminal_reason=None`` → SUCCESS outcome, so
    ``_maybe_stop_child_publisher`` actually publishes a RESULT_READY envelope.
    A future refactor that moves the publish ahead of the drain inverts
    ``cost_idx < publish_idx`` and trips this lock.
    """
    order: List[str] = []
    cost_handler = _make_recording_cost_handler(
        order,
        flush_result=FlushResult(drained=True, pending_count=0, persist_failures=0),
    )
    publisher = _make_recording_publisher(order)
    factory = _make_uow_factory()
    runner = _make_runner_for_terminal_ordering(cost_handler, factory, publisher)

    # consume_supervisor_terminate_marker is consulted inside
    # _maybe_stop_child_publisher; force False so the RESULT_READY branch
    # (not the supervisor-drove-terminal early return) is taken.
    with patch(
        "app.domain.services.supervisor_terminate_marker."
        "consume_supervisor_terminate_marker",
        return_value=False,
    ):
        await runner._set_terminal_status(SessionStatus.COMPLETED)

    # Both events were observed exactly once (the test is not vacuous).
    cost_handler.flush_pending.assert_awaited_once_with(timeout=3.0)
    publisher.publish.assert_awaited_once()
    # Clean drain → no degraded marker on this path.
    cost_handler.write_session_degraded_marker.assert_not_awaited()

    cost_idx = order.index("cost_flush") if "cost_flush" in order else -1
    publish_idx = (
        order.index("result_ready_publish")
        if "result_ready_publish" in order
        else -1
    )
    assert cost_idx >= 0, (
        f"cost_flush was never observed — harness no longer drives the drain "
        f"(order={order})"
    )
    assert publish_idx >= 0, (
        f"result_ready_publish was never observed — _maybe_stop_child_publisher "
        f"did not reach publisher.publish for a RESULT_READY envelope "
        f"(order={order})"
    )
    assert cost_idx < publish_idx, (
        f"INV-B6 VIOLATED: cost flush must precede RESULT_READY publish, but "
        f"observed order={order} (cost_idx={cost_idx}, publish_idx={publish_idx}). "
        f"A refactor moved the terminal envelope publish ahead of the cost "
        f"drain — the parent can now observe a completion whose cost rollup is "
        f"still in flight."
    )


# ---------------------------------------------------------------------------
# INV-B6 degraded-marker variant — XFAIL per plan (PR-10 refactor).
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason=(
        "INV-B6 degraded-marker ordering (degraded_marker BEFORE RESULT_READY) "
        "is deferred to the PR-10 refactor that restructures the terminal "
        "drain/publish boundary. The success-path test above "
        "(test_cost_flush_precedes_result_ready_publish) is the load-bearing "
        "INV-B6 lock; a dedicated degraded-marker-vs-publish lock is added when "
        "PR-10 lands the restructured finalize path."
    ),
    strict=True,
)
async def test_degraded_marker_also_precedes_result_ready() -> None:
    """[INV-B6 degraded variant — deferred to PR-10] On a drain timeout (or
    persist failures) the degraded-session marker MUST be written BEFORE the
    RESULT_READY publish. Pinning this as a SEPARATE lock from the success-path
    test is deferred to the PR-10 refactor (see xfail reason)."""
    raise NotImplementedError(
        "degraded-marker-before-RESULT_READY lock deferred to PR-10 refactor"
    )


# ---------------------------------------------------------------------------
# Registry hygiene — keep the global terminal-task registry clean so this file
# does not leak background tasks into sibling tests in the same worker.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
async def _drain_pending_terminal_tasks():
    yield
    # Best-effort: let any background terminal task settle so the global
    # _PENDING_TERMINAL_TASKS set does not carry state across tests.
    for _ in range(50):
        if not _PENDING_TERMINAL_TASKS:
            break
        await asyncio.sleep(0.01)
