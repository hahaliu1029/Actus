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
    assert "token_budget" in (payload.needs_authorization_details.observed_evidence or "")


async def test_finalize_budget_exhausted_wallclock(monkeypatch) -> None:
    _, publisher, envf = await _drive(
        inner_runner_side_effect=CancelledByEventError("llm_node_entry"),
        pre_stop_reason=StopReason.WALLCLOCK_BUDGET,
        monkeypatch=monkeypatch,
    )
    payload = envf.make_result_ready.call_args.kwargs["payload"]
    assert payload.needs_authorization_details.reason == "budget_exhausted"
    assert "wallclock_budget" in (payload.needs_authorization_details.observed_evidence or "")


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
    assert "token_budget" in (payload.needs_authorization_details.observed_evidence or "")


async def test_inflight_wallclock_budget_via_react_checkpoint(monkeypatch) -> None:
    """[r6 P2] In-flight WALLCLOCK_BUDGET → CancelledByEventError →
    _finalize_needs_authorization_budget with wallclock_budget evidence."""
    _, publisher, envf = await _drive_inflight_cancel(
        stop_reason=StopReason.WALLCLOCK_BUDGET, monkeypatch=monkeypatch,
    )
    payload = envf.make_result_ready.call_args.kwargs["payload"]
    assert payload.needs_authorization_details.reason == "budget_exhausted"
    assert "wallclock_budget" in (payload.needs_authorization_details.observed_evidence or "")


async def test_envelope_factory_called_with_correlation_id(monkeypatch) -> None:
    """Spec wire contract: correlation_id == coordinator_run_id so the parent
    waiter can join SPAWN_REQUEST → RESULT_READY across the audit trail."""
    _, _, envf = await _drive(monkeypatch=monkeypatch)
    call_kwargs = envf.make_result_ready.call_args.kwargs
    assert call_kwargs["correlation_id"] == "r1"
    assert call_kwargs["parent_session_id"] == "p1"
    assert call_kwargs["child_session_id"] == "c1"
