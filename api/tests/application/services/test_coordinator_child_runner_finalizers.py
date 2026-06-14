"""C2 PR-4 Task 4.7 — CoordinatorChildRunner finalizer matrix tests.

Spec ref: §8.3 (worker contract) + §8.5 (terminal matrix) + §14.3.1
(stop_reason routing) + r13 (phase-aware exploration_proposal).

The runner has 7 finalizer paths. Each MUST:
1. Build the correct payload (RESULT_READY vs CANCEL_ACK, outcome value,
   nested patch_manifest or needs_authorization_details).
2. Publish via ``envelope_factory.make_*`` + ``publisher.publish(envelope)``.
3. Run the cancel listener.shutdown() in ``finally`` for every exit path.

Finalizer matrix:
    natural done + phase=write       → _finalize_success            (PatchManifest)
    natural done + phase=exploration → _finalize_exploration_proposal (proposed_write_plan)
    exception                        → _finalize_failed
    asyncio.TimeoutError             → _finalize_timed_out
    ChildScopeViolation              → _finalize_needs_authorization_from_scope
    StopReason.TOKEN_BUDGET (any)    → _finalize_needs_authorization_budget
    StopReason.WALLCLOCK_BUDGET      → _finalize_needs_authorization_budget
    StopReason.PARENT_CANCEL         → _finalize_cancelled (CANCEL_ACK)
    stop_reason None defensive       → _finalize_cancelled (CANCEL_ACK) + warn
"""
from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.coordinator_child_runner import (
    CoordinatorChildRunner,
    StopReason,
)
from app.domain.models.mailbox_envelope import (
    MailboxEnvelopeType, ResultReadyOutcome,
)
from app.domain.models.work_unit import PathLease, WorkUnit
from app.domain.services.graphs.react_graph import CancelledByEventError
from app.domain.services.permission.child_scope_gate import ScopeDecision
from app.domain.services.permission.child_scope_violation import (
    ChildScopeViolation,
)


pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mk_work_unit(phase: str = "write") -> WorkUnit:
    if phase == "write":
        return WorkUnit(
            work_unit_id="wu1", objective="x", phase="write",
            allowed_tools=["file_write"],
            write_lease=[PathLease(
                path="/x", op="modify", base_digest="abc",
                seed_content_ref="minio://seed",
            )],
        )
    return WorkUnit(
        work_unit_id="wu1", objective="x", phase="exploration",
        allowed_tools=["file_read"], write_lease=[],
    )


def _mk_runner(
    *,
    inner_runner: MagicMock | None = None,
    publisher: AsyncMock | None = None,
    envelope_factory: MagicMock | None = None,
    parent_sandbox: MagicMock | None = None,
    artifact_storage: MagicMock | None = None,
    mailbox_subscriber: MagicMock | None = None,
) -> tuple[CoordinatorChildRunner, asyncio.Event, AsyncMock, MagicMock]:
    cancel_event = asyncio.Event()
    if publisher is None:
        publisher = AsyncMock()
        publisher.publish = AsyncMock()
    if envelope_factory is None:
        envelope_factory = MagicMock()
        envelope_factory.make_result_ready = MagicMock(
            return_value=MagicMock(type=MailboxEnvelopeType.RESULT_READY),
        )
        envelope_factory.make_cancel_ack = MagicMock(
            return_value=MagicMock(type=MailboxEnvelopeType.CANCEL_ACK),
        )
    runner = CoordinatorChildRunner(
        cancel_event=cancel_event,
        inner_runner=inner_runner or MagicMock(),
        publisher=publisher,
        parent_sandbox=parent_sandbox or MagicMock(),
        artifact_storage=artifact_storage or MagicMock(),
        envelope_factory=envelope_factory,
        parent_session_id="p1",
        coordinator_run_id="r1",
        mailbox_subscriber=mailbox_subscriber or MagicMock(),
    )
    return runner, cancel_event, publisher, envelope_factory


# ---------------------------------------------------------------------------
# Listener wiring — race-free start-before-invoke
# ---------------------------------------------------------------------------

async def test_run_work_unit_rejects_inner_runner_missing_protocol(monkeypatch) -> None:
    """[r2 P0] Pre-graph guard — when inner_runner lacks invoke_until_done
    (e.g. someone wires a raw AgentTaskRunner pre-PR-5 adapter), the runner
    MUST raise TypeError BEFORE listener wiring + envelope publish, so the
    debug trail points at the missing adapter, not at AttributeError mid-graph."""
    bad_inner = object()  # no invoke_until_done method
    runner, ce, _, _ = _mk_runner(inner_runner=bad_inner)
    with pytest.raises(TypeError, match="invoke_until_done"):
        await runner.run_work_unit(
            coordinator_run_id="r1", work_unit=_mk_work_unit("write"),
            child_session_id="c1", spawn_manifest=MagicMock(), cancel_event=ce,
            root_session_id="root1",
        )


async def test_run_work_unit_awaits_listener_ready_before_invoke(monkeypatch) -> None:
    order: list[str] = []
    listener = MagicMock()
    listener.ready_event = asyncio.Event()

    async def fake_start() -> None:
        order.append("start")
        listener.ready_event.set()

    async def fake_shutdown(timeout: float = 1.0) -> None:
        order.append("shutdown")

    listener.start = AsyncMock(side_effect=fake_start)
    listener.shutdown = AsyncMock(side_effect=fake_shutdown)

    inner_runner = MagicMock()

    async def fake_invoke(user_message: str):
        order.append("invoke")
        return MagicMock(name="done_event")

    inner_runner.invoke_until_done = AsyncMock(side_effect=fake_invoke)

    runner, ce, _, _ = _mk_runner(inner_runner=inner_runner)
    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner."
        "CoordinatorChildCancelListener",
        lambda **_kw: listener,
    )
    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner."
        "PromptAssembler.build_minimal_for_coordinator_child",
        staticmethod(lambda **_kw: "prompt"),
    )
    await runner.run_work_unit(
        coordinator_run_id="r1", work_unit=_mk_work_unit("exploration"),
        child_session_id="c1", spawn_manifest=MagicMock(), cancel_event=ce,
        root_session_id="root1",
    )
    assert order[0] == "start"
    assert order.index("invoke") > order.index("start")
    assert "shutdown" in order


async def test_listener_shutdown_runs_on_exception(monkeypatch) -> None:
    listener = MagicMock()
    listener.ready_event = asyncio.Event()
    listener.start = AsyncMock(side_effect=lambda: listener.ready_event.set())
    listener.shutdown = AsyncMock()
    inner_runner = MagicMock()
    inner_runner.invoke_until_done = AsyncMock(side_effect=RuntimeError("boom"))
    runner, ce, _, _ = _mk_runner(inner_runner=inner_runner)
    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner."
        "CoordinatorChildCancelListener",
        lambda **_kw: listener,
    )
    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner."
        "PromptAssembler.build_minimal_for_coordinator_child",
        staticmethod(lambda **_kw: "prompt"),
    )
    await runner.run_work_unit(
        coordinator_run_id="r1", work_unit=_mk_work_unit("write"),
        child_session_id="c1", spawn_manifest=MagicMock(), cancel_event=ce,
        root_session_id="root1",
    )
    listener.shutdown.assert_awaited_once()


# ---------------------------------------------------------------------------
# 7 finalizer matrix — happy path + 6 exception paths
# ---------------------------------------------------------------------------

async def _drive(
    *,
    inner_runner_side_effect=None,
    inner_runner_return=None,
    phase: str = "write",
    pre_stop_reason: StopReason | None = None,
    cancel_set: bool = False,
    monkeypatch=None,
) -> tuple[CoordinatorChildRunner, AsyncMock, MagicMock]:
    inner_runner = MagicMock()
    if inner_runner_side_effect is not None:
        inner_runner.invoke_until_done = AsyncMock(
            side_effect=inner_runner_side_effect,
        )
    else:
        inner_runner.invoke_until_done = AsyncMock(
            return_value=inner_runner_return or MagicMock(name="done_event"),
        )

    listener = MagicMock()
    listener.ready_event = asyncio.Event()
    listener.start = AsyncMock(side_effect=lambda: listener.ready_event.set())
    listener.shutdown = AsyncMock()
    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner."
        "CoordinatorChildCancelListener",
        lambda **_kw: listener,
    )
    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner."
        "PromptAssembler.build_minimal_for_coordinator_child",
        staticmethod(lambda **_kw: "prompt"),
    )

    runner, ce, publisher, envf = _mk_runner(inner_runner=inner_runner)
    if pre_stop_reason is not None:
        runner.request_stop(pre_stop_reason)
    if cancel_set and not ce.is_set():
        ce.set()
    await runner.run_work_unit(
        coordinator_run_id="r1", work_unit=_mk_work_unit(phase),
        child_session_id="c1", spawn_manifest=MagicMock(), cancel_event=ce,
        root_session_id="root1",
    )
    return runner, publisher, envf


async def test_finalize_success_write_phase(monkeypatch) -> None:
    _, publisher, envf = await _drive(phase="write", monkeypatch=monkeypatch)
    publisher.publish.assert_awaited_once()
    envf.make_result_ready.assert_called_once()
    payload = envf.make_result_ready.call_args.kwargs["payload"]
    assert payload.outcome == ResultReadyOutcome.SUCCESS
    assert payload.patch_manifest is not None
    assert payload.patch_manifest.patch_id == "r1:wu1:p"
    assert payload.needs_authorization_details is None


async def test_finalize_exploration_proposal(monkeypatch) -> None:
    _, publisher, envf = await _drive(phase="exploration", monkeypatch=monkeypatch)
    publisher.publish.assert_awaited_once()
    payload = envf.make_result_ready.call_args.kwargs["payload"]
    assert payload.outcome == ResultReadyOutcome.NEEDS_AUTHORIZATION
    assert payload.needs_authorization_details is not None
    assert payload.needs_authorization_details.reason == "exploration_proposal"
    assert payload.needs_authorization_details.proposed_write_plan is not None


async def test_finalize_failed_on_unhandled_exception(monkeypatch) -> None:
    _, publisher, envf = await _drive(
        inner_runner_side_effect=RuntimeError("kaboom"),
        monkeypatch=monkeypatch,
    )
    publisher.publish.assert_awaited_once()
    payload = envf.make_result_ready.call_args.kwargs["payload"]
    assert payload.outcome == ResultReadyOutcome.FAILED
    assert "kaboom" in payload.summary


async def test_finalize_timed_out_on_asyncio_timeout(monkeypatch) -> None:
    _, publisher, envf = await _drive(
        inner_runner_side_effect=asyncio.TimeoutError(),
        monkeypatch=monkeypatch,
    )
    publisher.publish.assert_awaited_once()
    payload = envf.make_result_ready.call_args.kwargs["payload"]
    assert payload.outcome == ResultReadyOutcome.TIMED_OUT


async def test_finalize_needs_authorization_from_scope(monkeypatch) -> None:
    exc = ChildScopeViolation(
        decision=ScopeDecision.OUT_OF_PATH_LEASE,
        tool_name="file_write",
        target_path="/forbidden",
    )
    _, publisher, envf = await _drive(
        inner_runner_side_effect=exc, monkeypatch=monkeypatch,
    )
    publisher.publish.assert_awaited_once()
    payload = envf.make_result_ready.call_args.kwargs["payload"]
    assert payload.outcome == ResultReadyOutcome.NEEDS_AUTHORIZATION
    assert payload.needs_authorization_details is not None
    assert payload.needs_authorization_details.reason == "out_of_path_lease"
    assert payload.needs_authorization_details.requested_tool == "file_write"
    assert payload.needs_authorization_details.requested_paths == ("/forbidden",)


@pytest.mark.parametrize("scope_dec,expected_reason", [
    (ScopeDecision.OUT_OF_TOOL_ALLOWLIST, "out_of_tool_allowlist"),
    (ScopeDecision.HARD_BLOCKED, "hard_blocked"),
    (ScopeDecision.OP_MISMATCH, "op_mismatch"),
    (ScopeDecision.OUT_OF_PATH_LEASE, "out_of_path_lease"),
    (ScopeDecision.LEASE_EXPIRED, "lease_expired"),
    (ScopeDecision.REVISION_DRIFT, "revision_drift"),
    (ScopeDecision.BUDGET_EXHAUSTED, "budget_exhausted"),
])
async def test_scope_decision_to_reason_mapping_complete(
    scope_dec, expected_reason, monkeypatch,
) -> None:
    """Pin the full ScopeDecision → NeedsAuthorizationDetails.reason map.
    Any new ScopeDecision value MUST extend the map or this test breaks."""
    exc = ChildScopeViolation(
        decision=scope_dec, tool_name="x", target_path="/x",
    )
    _, publisher, envf = await _drive(
        inner_runner_side_effect=exc, monkeypatch=monkeypatch,
    )
    payload = envf.make_result_ready.call_args.kwargs["payload"]
    assert payload.needs_authorization_details.reason == expected_reason


async def test_finalize_cancelled_on_parent_cancel(monkeypatch) -> None:
    """StopReason.PARENT_CANCEL + CancelledByEventError → CANCEL_ACK(cancelled),
    NOT RESULT_READY (spec §8.5 r5 P0-2)."""
    _, publisher, envf = await _drive(
        inner_runner_side_effect=CancelledByEventError("react_loop_entry"),
        pre_stop_reason=StopReason.PARENT_CANCEL,
        monkeypatch=monkeypatch,
    )
    publisher.publish.assert_awaited_once()
    envf.make_cancel_ack.assert_called_once()
    envf.make_result_ready.assert_not_called()
    payload = envf.make_cancel_ack.call_args.kwargs["payload"]
    assert payload.final_state == "cancelled"


async def test_finalize_budget_exhausted_token(monkeypatch) -> None:
    _, publisher, envf = await _drive(
        inner_runner_side_effect=CancelledByEventError("react_loop_entry"),
        pre_stop_reason=StopReason.TOKEN_BUDGET,
        monkeypatch=monkeypatch,
    )
    publisher.publish.assert_awaited_once()
    envf.make_result_ready.assert_called_once()
    payload = envf.make_result_ready.call_args.kwargs["payload"]
    assert payload.outcome == ResultReadyOutcome.NEEDS_AUTHORIZATION
    assert payload.needs_authorization_details.reason == "budget_exhausted"
    assert payload.needs_authorization_details.observed_evidence == (
        "stop_reason=token_budget"
    )  # budget=None harness → exact legacy format (spec §5-1, R6-A1)


async def test_finalize_budget_exhausted_wallclock(monkeypatch) -> None:
    _, publisher, envf = await _drive(
        inner_runner_side_effect=CancelledByEventError("llm_node_entry"),
        pre_stop_reason=StopReason.WALLCLOCK_BUDGET,
        monkeypatch=monkeypatch,
    )
    payload = envf.make_result_ready.call_args.kwargs["payload"]
    assert payload.needs_authorization_details.reason == "budget_exhausted"
    assert payload.needs_authorization_details.observed_evidence == (
        "stop_reason=wallclock_budget"
    )  # (spec §5-2, R6-A2)


async def _drive_finish_line_trip(
    *, reason: StopReason, phase: str = "write", monkeypatch,
) -> tuple[CoordinatorChildRunner, AsyncMock, MagicMock, AsyncMock]:
    """[impl-audit R3#2] Drive a finish-line trip that lands INSIDE
    invoke_until_done (NOT pre-invoke). The inner runner records the stop reason
    mid-invoke and then returns a done_event WITHOUT raising — exactly the
    adapter-returns-done window the exit guard at coordinator_child_runner.py:393
    closes. The plain _drive(pre_stop_reason=...) sets the reason BEFORE
    run_work_unit, which short-circuits at the PRE-invoke guard (:281-296) and
    never reaches :393 — so a finish-line test built on it is vacuous (passes
    even with the :393 guard deleted). This helper forces the run through the
    inner invoke so deleting :393 actually breaks the test."""
    inner_runner = MagicMock()

    listener = MagicMock()
    listener.ready_event = asyncio.Event()
    listener.start = AsyncMock(side_effect=lambda: listener.ready_event.set())
    listener.shutdown = AsyncMock()
    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner."
        "CoordinatorChildCancelListener",
        lambda **_kw: listener,
    )
    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner."
        "PromptAssembler.build_minimal_for_coordinator_child",
        staticmethod(lambda **_kw: "prompt"),
    )

    runner, ce, publisher, envf = _mk_runner(inner_runner=inner_runner)

    async def _invoke_then_trip(**_kw):
        # stop_reason is None at the pre-invoke guard (we did NOT pre-set it),
        # so the run reaches the inner invoke; record the trip now so the EXIT
        # guard (:393) — not the pre-invoke guard (:283) — is what routes.
        runner.request_stop(reason)
        return MagicMock(name="done_event")

    inner_runner.invoke_until_done = AsyncMock(side_effect=_invoke_then_trip)

    await runner.run_work_unit(
        coordinator_run_id="r1", work_unit=_mk_work_unit(phase),
        child_session_id="c1", spawn_manifest=MagicMock(), cancel_event=ce,
        root_session_id="root1",
    )
    return runner, publisher, envf, inner_runner


async def test_finish_line_trip_with_done_returned_routes_budget(monkeypatch) -> None:
    """[INV-B1 — impl-audit R2#1, pinned R3#2] The trip lands while the adapter
    is already blocked inside output_stream.get() awaiting the final DoneEvent:
    request_stop() runs mid-invoke, then DoneEvent enqueues, and get() returns
    it — the adapter's loop-top cancel check already passed, so invoke_until_done
    returns done WITHOUT raising. The runner's exit-point guard (:393) must still
    route to the budget finalizer; natural success must never win over a recorded
    trip. Trips MID-INVOKE so deleting :393 → _finalize_success → SUCCESS makes
    this fail."""
    _, publisher, envf, inner = await _drive_finish_line_trip(
        reason=StopReason.TOKEN_BUDGET, monkeypatch=monkeypatch,
    )
    inner.invoke_until_done.assert_awaited_once()  # proves :283 passed → :393 ran
    publisher.publish.assert_awaited_once()
    envf.make_result_ready.assert_called_once()
    payload = envf.make_result_ready.call_args.kwargs["payload"]
    assert payload.outcome == ResultReadyOutcome.NEEDS_AUTHORIZATION
    assert payload.needs_authorization_details.reason == "budget_exhausted"
    assert payload.needs_authorization_details.observed_evidence == (
        "stop_reason=token_budget"
    )  # budget=None harness → exact legacy format


async def test_finish_line_parent_cancel_with_done_returned_routes_cancelled(
    monkeypatch,
) -> None:
    """[INV-B1 generalization — impl-audit R2#1, pinned R3#2] Same mid-invoke
    finish-line interleaving with PARENT_CANCEL: the exit guard routes through
    _finalize_by_stop_reason → cancel-ack, not success (guard must not hardcode
    budget reasons). Deleting :393 → _finalize_success → SUCCESS, so
    make_result_ready.assert_not_called() would fail.

    Scope note (impl-audit R3#1): this pins a trip RECORDED BY the runner's
    synchronous exit point. A PARENT_CANCEL arriving LATER, during
    _finalize_success's own awaits, is the spec §6-L9 RESULT_READY-vs-
    CANCEL_REQUEST benign interleaving (supervisor observes terminal
    RESULT_READY; redundant force-terminate is benign) — out of scope here."""
    _, publisher, envf, inner = await _drive_finish_line_trip(
        reason=StopReason.PARENT_CANCEL, monkeypatch=monkeypatch,
    )
    inner.invoke_until_done.assert_awaited_once()
    publisher.publish.assert_awaited_once()
    envf.make_cancel_ack.assert_called_once()
    envf.make_result_ready.assert_not_called()


async def test_finalize_none_stop_reason_defensive_fallback(monkeypatch, caplog) -> None:
    """[spec §14.3.1 defensive] CancelledByEventError raised but stop_reason
    None → _finalize_cancelled + WARN log so an unattributed cancel surfaces
    in audit, not silently."""
    caplog.set_level(
        logging.WARNING,
        logger="app.application.services.coordinator_child_runner",
    )
    _, publisher, envf = await _drive(
        inner_runner_side_effect=CancelledByEventError("tool_node_entry"),
        monkeypatch=monkeypatch,
    )
    envf.make_cancel_ack.assert_called_once()
    assert any(
        "stop_reason" in r.message.lower()
        and ("none" in r.message.lower() or "fallback" in r.message.lower())
        for r in caplog.records
    ), f"expected warn about None stop_reason; got {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# Cross-finalizer invariants
# ---------------------------------------------------------------------------

async def test_publisher_called_exactly_once_per_run(monkeypatch) -> None:
    """Every finalizer path must publish EXACTLY one envelope. Double-publish
    would re-fire downstream consumers; zero-publish strands the parent waiter."""
    _, publisher, _ = await _drive(monkeypatch=monkeypatch)
    publisher.publish.assert_awaited_once()


# ---------------------------------------------------------------------------
# [r6 P2] In-flight cancel: request_stop fires DURING inner_runner.invoke_until_done
# (NOT before). This exercises the actual `except CancelledByEventError` catch path
# at coordinator_child_runner.py:229. The earlier matrix tests pre-set request_stop
# before run_work_unit entry, which makes the pre-invoke short-circuit at
# coordinator_child_runner.py:148 fire INSTEAD of the in-flight branch.
# ---------------------------------------------------------------------------


async def _drive_inflight_cancel(
    *,
    stop_reason: StopReason,
    monkeypatch,
) -> tuple[CoordinatorChildRunner, AsyncMock, MagicMock]:
    """Construct an inner_runner whose invoke_until_done calls request_stop
    THEN raises CancelledByEventError — modeling a real react_graph
    checkpoint trip after the runner already entered the inner loop."""
    inner_runner = MagicMock()

    async def fake_invoke(user_message: str):
        # Simulate: the parent's CANCEL_REQUEST landed mid-run, the listener
        # called runner.request_stop, and a react_graph checkpoint raised.
        runner_ref.request_stop(stop_reason)
        raise CancelledByEventError("react_loop_entry")

    inner_runner.invoke_until_done = AsyncMock(side_effect=fake_invoke)

    listener = MagicMock()
    listener.ready_event = asyncio.Event()
    listener.start = AsyncMock(side_effect=lambda: listener.ready_event.set())
    listener.shutdown = AsyncMock()
    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner."
        "CoordinatorChildCancelListener",
        lambda **_kw: listener,
    )
    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner."
        "PromptAssembler.build_minimal_for_coordinator_child",
        staticmethod(lambda **_kw: "prompt"),
    )

    runner_ref, ce, publisher, envf = _mk_runner(inner_runner=inner_runner)
    await runner_ref.run_work_unit(
        coordinator_run_id="r1", work_unit=_mk_work_unit("write"),
        child_session_id="c1", spawn_manifest=MagicMock(), cancel_event=ce,
        root_session_id="root1",
    )
    # Verify inner runner was actually invoked (i.e. we did NOT short-circuit
    # through the pre-invoke branch — that's the whole point of this test).
    inner_runner.invoke_until_done.assert_awaited_once()
    return runner_ref, publisher, envf


async def test_inflight_parent_cancel_via_react_checkpoint(monkeypatch) -> None:
    """[r6 P2] In-flight PARENT_CANCEL → CancelledByEventError caught at
    coordinator_child_runner.py:229 → _finalize_by_stop_reason → CANCEL_ACK.
    Covers the path the earlier pre-start test was bypassing."""
    _, publisher, envf = await _drive_inflight_cancel(
        stop_reason=StopReason.PARENT_CANCEL, monkeypatch=monkeypatch,
    )
    publisher.publish.assert_awaited_once()
    envf.make_cancel_ack.assert_called_once()
    envf.make_result_ready.assert_not_called()


async def test_inflight_token_budget_via_react_checkpoint(monkeypatch) -> None:
    """[r6 P2] In-flight TOKEN_BUDGET → CancelledByEventError →
    _finalize_needs_authorization_budget. Real production path, not the
    pre-start short-circuit."""
    _, publisher, envf = await _drive_inflight_cancel(
        stop_reason=StopReason.TOKEN_BUDGET, monkeypatch=monkeypatch,
    )
    publisher.publish.assert_awaited_once()
    envf.make_result_ready.assert_called_once()
    payload = envf.make_result_ready.call_args.kwargs["payload"]
    assert payload.outcome == ResultReadyOutcome.NEEDS_AUTHORIZATION
    assert payload.needs_authorization_details.reason == "budget_exhausted"
    assert payload.needs_authorization_details.observed_evidence == (
        "stop_reason=token_budget"
    )


async def test_inflight_wallclock_budget_via_react_checkpoint(monkeypatch) -> None:
    """[r6 P2] In-flight WALLCLOCK_BUDGET → CancelledByEventError →
    _finalize_needs_authorization_budget with wallclock_budget evidence."""
    _, publisher, envf = await _drive_inflight_cancel(
        stop_reason=StopReason.WALLCLOCK_BUDGET, monkeypatch=monkeypatch,
    )
    payload = envf.make_result_ready.call_args.kwargs["payload"]
    assert payload.needs_authorization_details.reason == "budget_exhausted"
    assert payload.needs_authorization_details.observed_evidence == (
        "stop_reason=wallclock_budget"
    )


async def test_envelope_factory_called_with_correlation_id(monkeypatch) -> None:
    """Spec wire contract: correlation_id == coordinator_run_id so the parent
    waiter can join SPAWN_REQUEST → RESULT_READY across the audit trail."""
    _, _, envf = await _drive(monkeypatch=monkeypatch)
    call_kwargs = envf.make_result_ready.call_args.kwargs
    assert call_kwargs["correlation_id"] == "r1"
    assert call_kwargs["parent_session_id"] == "p1"
    assert call_kwargs["child_session_id"] == "c1"


# ---------------------------------------------------------------------------
# [C2b budget] INV-B6 — budget=None legacy parity (spec §5-7, four clauses)
# ---------------------------------------------------------------------------


async def test_budget_none_legacy_parity(monkeypatch) -> None:
    """INV-B6 four-clause contract when budget=None (every existing harness
    construction in this file):
      (i)  watchdog is NEVER started;
      (ii) budget callback is neither required nor read;
      (iii) finalizer outcome routing is unchanged (pinned by the whole
           pre-existing matrix in this file — this test re-asserts one path);
      (iv) _build_budget_evidence() keeps the old exact format.
    """
    wd_spy = MagicMock()
    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner."
        "start_wallclock_watchdog",
        wd_spy,
    )
    # (i)+(iii): natural success path, budget=None → no watchdog, SUCCESS.
    _, publisher, envf = await _drive(phase="write", monkeypatch=monkeypatch)
    wd_spy.assert_not_called()
    assert (
        envf.make_result_ready.call_args.kwargs["payload"].outcome
        == ResultReadyOutcome.SUCCESS
    )

    # (ii)+(iv): budget trip with budget=None → old exact evidence format,
    # no callback read (none attached — would AttributeError if read).
    runner2, _, envf2 = await _drive(
        inner_runner_side_effect=CancelledByEventError("react_loop_entry"),
        pre_stop_reason=StopReason.TOKEN_BUDGET,
        monkeypatch=monkeypatch,
    )
    assert runner2._build_budget_evidence() == "stop_reason=token_budget"
    payload = envf2.make_result_ready.call_args.kwargs["payload"]
    assert payload.needs_authorization_details.observed_evidence == (
        "stop_reason=token_budget"
    )


# ---------------------------------------------------------------------------
# [C2b budget D5] Evidence three-state contract (spec §5-6, R7#1 + R6#7)
# ---------------------------------------------------------------------------


class _FakeBudgetCallback:
    """Stands in for BudgetEnforcementCallback: only the cumulative_usd
    read-only property matters to the evidence builder."""

    def __init__(self, cumulative: float) -> None:
        self._c = cumulative

    @property
    def cumulative_usd(self) -> float:
        return self._c


def _mk_budget_for_evidence():
    from app.domain.services.permission.child_permission_context import (
        ChildBudget,
    )

    # Distinct cap vs observed values so a swapped-field mutation fails
    # loudly (R6#7: 拒绝 observed/cap 互换).
    return ChildBudget(
        max_tool_calls=25, max_token_cost_usd=0.5, max_wallclock_seconds=300,
    )


async def test_budget_evidence_enriched_full_state(monkeypatch) -> None:
    """State (i): budget + callback attached → full fields, values from the
    RIGHT sources (observed == callback.cumulative_usd, cap == budget cap)."""
    inner = MagicMock()

    async def trip_then_raise(user_message: str):
        runner_ref.request_stop(StopReason.TOKEN_BUDGET)
        raise CancelledByEventError("react_loop_entry")

    inner.invoke_until_done = AsyncMock(side_effect=trip_then_raise)

    listener = MagicMock()
    listener.ready_event = asyncio.Event()
    listener.start = AsyncMock(side_effect=lambda: listener.ready_event.set())
    listener.shutdown = AsyncMock()
    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner."
        "CoordinatorChildCancelListener",
        lambda **_kw: listener,
    )
    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner."
        "PromptAssembler.build_minimal_for_coordinator_child",
        staticmethod(lambda **_kw: "prompt"),
    )

    cancel_event = asyncio.Event()
    publisher = AsyncMock()
    publisher.publish = AsyncMock()
    envf = MagicMock()
    envf.make_result_ready = MagicMock(
        return_value=MagicMock(type=MailboxEnvelopeType.RESULT_READY),
    )
    envf.make_cancel_ack = MagicMock(
        return_value=MagicMock(type=MailboxEnvelopeType.CANCEL_ACK),
    )
    runner_ref = CoordinatorChildRunner(
        cancel_event=cancel_event,
        inner_runner=inner,
        publisher=publisher,
        envelope_factory=envf,
        parent_session_id="p1",
        coordinator_run_id="r1",
        mailbox_subscriber=MagicMock(),
        budget=_mk_budget_for_evidence(),
    )
    runner_ref.attach_budget_callback(_FakeBudgetCallback(0.123456))

    await runner_ref.run_work_unit(
        coordinator_run_id="r1", work_unit=_mk_work_unit("write"),
        child_session_id="c1", spawn_manifest=MagicMock(),
        cancel_event=cancel_event, root_session_id="root1",
    )

    evidence = (
        envf.make_result_ready.call_args.kwargs["payload"]
        .needs_authorization_details.observed_evidence
    )
    assert "stop_reason=token_budget" in evidence
    assert "token_cost_usd_observed=0.123456" in evidence
    assert "token_cap_usd=0.500000" in evidence
    assert "wallclock_cap_seconds=300" in evidence
    # elapsed: inner invoke ran → stamp present and sane.
    assert "wallclock_elapsed_seconds=" in evidence
    elapsed = float(evidence.split("wallclock_elapsed_seconds=")[1].split()[0])
    assert 0.0 <= elapsed < 60.0


async def test_budget_evidence_wallclock_only_state(monkeypatch) -> None:
    """State (ii): budget set, callback None (unpriced fail-soft) → caps AND
    elapsed still emitted; token_cost_usd_observed OMITTED (R7#1).

    [codex plan-R2#1] elapsed asserted EXPLICITLY with a live start stamp —
    kills the mutant that emits wallclock_elapsed_seconds only inside the
    `if self._budget_callback is not None:` branch. Driven at the evidence-
    builder unit level (the inflight wallclock path with a real stamp is
    covered end-to-end by test_child_wallclock_cap_via_async_watchdog in
    the watchdog-wiring file)."""
    import time as _time

    runner, publisher, envf = await _drive(
        inner_runner_side_effect=CancelledByEventError("react_loop_entry"),
        pre_stop_reason=StopReason.WALLCLOCK_BUDGET,
        monkeypatch=monkeypatch,
    )
    # _drive's harness has no budget kwarg — drive the evidence builder
    # directly on a budget-bearing, callback-less runner with an explicit
    # inner-invoke start stamp (the wallclock-only shape):
    runner._budget = _mk_budget_for_evidence()
    runner._budget_callback = None
    runner._inner_invoke_started_monotonic = _time.monotonic() - 1.0
    evidence = runner._build_budget_evidence()
    assert "stop_reason=wallclock_budget" in evidence
    assert "token_cost_usd_observed" not in evidence
    assert "token_cap_usd=0.500000" in evidence
    assert "wallclock_cap_seconds=300" in evidence
    assert "wallclock_elapsed_seconds=" in evidence, (
        "state (ii) must STILL emit elapsed — only observed is omitted"
    )
    elapsed = float(evidence.split("wallclock_elapsed_seconds=")[1].split()[0])
    assert 0.9 <= elapsed < 60.0


async def test_budget_evidence_success_path_absent(monkeypatch) -> None:
    """Success path produces NO needs_authorization_details (existing :252
    assertion re-pinned here for the D5 slice)."""
    _, publisher, envf = await _drive(phase="write", monkeypatch=monkeypatch)
    payload = envf.make_result_ready.call_args.kwargs["payload"]
    assert payload.needs_authorization_details is None


# ---------------------------------------------------------------------------
# [C2b budget D10/INV-B9] budget_exhaustion metric — best-effort, finalizer-only
# ---------------------------------------------------------------------------


def _mk_metrics() -> MagicMock:
    metrics = MagicMock(name="coordinator_metrics")
    metrics.budget_exhaustion = MagicMock()
    metrics.budget_exhaustion.add = MagicMock()
    return metrics


async def _drive_with_metrics(
    *, monkeypatch, metrics, side_effect, pre_stop_reason=None,
):
    """_drive variant whose runner carries coordinator_metrics."""
    inner_runner = MagicMock()
    inner_runner.invoke_until_done = AsyncMock(side_effect=side_effect)

    listener = MagicMock()
    listener.ready_event = asyncio.Event()
    listener.start = AsyncMock(side_effect=lambda: listener.ready_event.set())
    listener.shutdown = AsyncMock()
    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner."
        "CoordinatorChildCancelListener",
        lambda **_kw: listener,
    )
    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner."
        "PromptAssembler.build_minimal_for_coordinator_child",
        staticmethod(lambda **_kw: "prompt"),
    )
    cancel_event = asyncio.Event()
    publisher = AsyncMock()
    publisher.publish = AsyncMock()
    envf = MagicMock()
    envf.make_result_ready = MagicMock(
        return_value=MagicMock(type=MailboxEnvelopeType.RESULT_READY),
    )
    envf.make_cancel_ack = MagicMock(
        return_value=MagicMock(type=MailboxEnvelopeType.CANCEL_ACK),
    )
    runner = CoordinatorChildRunner(
        cancel_event=cancel_event,
        inner_runner=inner_runner,
        publisher=publisher,
        envelope_factory=envf,
        parent_session_id="p1",
        coordinator_run_id="r1",
        mailbox_subscriber=MagicMock(),
        coordinator_metrics=metrics,
    )
    if pre_stop_reason is not None:
        runner.request_stop(pre_stop_reason)
    await runner.run_work_unit(
        coordinator_run_id="r1", work_unit=_mk_work_unit("write"),
        child_session_id="c1", spawn_manifest=MagicMock(),
        cancel_event=cancel_event, root_session_id="root1",
    )
    return runner, publisher, envf


async def test_budget_trip_emits_exhaustion_metric(monkeypatch) -> None:
    """[spec §5-15] Budget finalizer → budget_exhaustion.add(1, attrs) with
    the three pinned attributes."""
    metrics = _mk_metrics()
    _, publisher, _ = await _drive_with_metrics(
        monkeypatch=monkeypatch, metrics=metrics,
        side_effect=CancelledByEventError("react_loop_entry"),
        pre_stop_reason=StopReason.TOKEN_BUDGET,
    )
    publisher.publish.assert_awaited_once()
    metrics.budget_exhaustion.add.assert_called_once_with(
        1,
        attributes={
            "stop_reason": "token_budget",
            "coordinator_run_id": "r1",
            "work_unit_id": "wu1",
        },
    )


async def test_metric_failure_does_not_block_envelope(monkeypatch) -> None:
    """[INV-B9] Telemetry raising must neither block nor mask the terminal
    envelope publish."""
    metrics = _mk_metrics()
    metrics.budget_exhaustion.add = MagicMock(side_effect=RuntimeError("otel down"))
    _, publisher, envf = await _drive_with_metrics(
        monkeypatch=monkeypatch, metrics=metrics,
        side_effect=CancelledByEventError("react_loop_entry"),
        pre_stop_reason=StopReason.WALLCLOCK_BUDGET,
    )
    publisher.publish.assert_awaited_once()
    payload = envf.make_result_ready.call_args.kwargs["payload"]
    assert payload.outcome == ResultReadyOutcome.NEEDS_AUTHORIZATION


@pytest.mark.parametrize("path", ["success", "cancelled", "failed"])
async def test_non_budget_finalizers_do_not_emit(monkeypatch, path) -> None:
    """[spec §5-15 + R10#1 反向边界] success / cancelled / FAILED finalizers
    never touch the exhaustion counter."""
    metrics = _mk_metrics()
    side_effects = {
        "success": None,
        "cancelled": CancelledByEventError("react_loop_entry"),
        "failed": RuntimeError("kaboom"),
    }
    pre = StopReason.PARENT_CANCEL if path == "cancelled" else None
    effect = side_effects[path]
    await _drive_with_metrics(
        monkeypatch=monkeypatch, metrics=metrics,
        side_effect=effect if effect is not None else None,
        pre_stop_reason=pre,
    )
    metrics.budget_exhaustion.add.assert_not_called()


async def test_publish_failure_propagates_despite_metrics(monkeypatch) -> None:
    """[R10#1] _publish_result_ready raising must PROPAGATE — kills the
    'wrap publish inside the telemetry try/except' mutation. Driven at the
    finalizer level directly (unit)."""
    metrics = _mk_metrics()
    publisher = AsyncMock()
    publisher.publish = AsyncMock(side_effect=RuntimeError("redis down"))
    envf = MagicMock()
    envf.make_result_ready = MagicMock(
        return_value=MagicMock(type=MailboxEnvelopeType.RESULT_READY),
    )
    runner = CoordinatorChildRunner(
        cancel_event=asyncio.Event(),
        publisher=publisher,
        envelope_factory=envf,
        parent_session_id="p1",
        coordinator_run_id="r1",
        mailbox_subscriber=MagicMock(),
        coordinator_metrics=metrics,
    )
    runner.request_stop(StopReason.TOKEN_BUDGET)
    with pytest.raises(RuntimeError, match="redis down"):
        await runner._finalize_needs_authorization_budget(
            "r1", _mk_work_unit("write"), "c1",
        )
    metrics.budget_exhaustion.add.assert_called_once()  # emit happened first
