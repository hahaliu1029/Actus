"""[C2b budget §5-3/§5-4] Wallclock watchdog WIRING tests — runner-level.

Distinct from the finalizer-matrix tests (which pre-set stop_reason and never
need a real budget): these construct the runner WITH a ChildBudget and assert
the watchdog actually starts, actually trips on a slow inner invoke, and is
cancelled on EVERY inner-invoke exit path before any finalizer runs (INV-B2,
the D2 inner-finally mechanism).
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
from app.domain.models.mailbox_envelope import MailboxEnvelopeType, ResultReadyOutcome
from app.domain.models.work_unit import WorkUnit
from app.domain.services.graphs.react_graph import CancelledByEventError
from app.domain.services.permission.child_permission_context import ChildBudget


pytestmark = pytest.mark.anyio


def _mk_work_unit() -> WorkUnit:
    return WorkUnit(
        work_unit_id="wu1", objective="x", phase="exploration",
        allowed_tools=["file_read"], write_lease=[],
    )


def _mk_budget(*, wallclock: int = 300, token: float = 0.5) -> ChildBudget:
    return ChildBudget(
        max_tool_calls=25, max_token_cost_usd=token,
        max_wallclock_seconds=wallclock,
    )


def _patch_listener_and_prompt(monkeypatch) -> None:
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


def _mk_runner(
    *, inner_runner: MagicMock, budget: ChildBudget | None,
) -> tuple[CoordinatorChildRunner, asyncio.Event, AsyncMock, MagicMock]:
    cancel_event = asyncio.Event()
    publisher = AsyncMock()
    publisher.publish = AsyncMock()
    envelope_factory = MagicMock()
    envelope_factory.make_result_ready = MagicMock(
        return_value=MagicMock(type=MailboxEnvelopeType.RESULT_READY),
    )
    envelope_factory.make_cancel_ack = MagicMock(
        return_value=MagicMock(type=MailboxEnvelopeType.CANCEL_ACK),
    )
    runner = CoordinatorChildRunner(
        cancel_event=cancel_event,
        inner_runner=inner_runner,
        publisher=publisher,
        parent_sandbox=MagicMock(),
        artifact_storage=MagicMock(),
        envelope_factory=envelope_factory,
        parent_session_id="p1",
        coordinator_run_id="r1",
        mailbox_subscriber=MagicMock(),
        budget=budget,
    )
    return runner, cancel_event, publisher, envelope_factory


async def _run(runner: CoordinatorChildRunner, cancel_event: asyncio.Event):
    return await runner.run_work_unit(
        coordinator_run_id="r1", work_unit=_mk_work_unit(),
        child_session_id="c1", spawn_manifest=MagicMock(),
        cancel_event=cancel_event, root_session_id="root1",
    )


async def test_child_wallclock_cap_via_async_watchdog(monkeypatch) -> None:
    """[spec §5-3] REAL asyncio timing: sub-second cap + slow inner that
    raises CancelledByEventError at its next 'checkpoint' once the watchdog
    has tripped → budget finalizer → NEEDS_AUTHORIZATION(budget_exhausted)
    with stop_reason=wallclock_budget."""
    _patch_listener_and_prompt(monkeypatch)

    runner_holder: dict = {}

    async def slow_invoke_with_checkpoints(user_message: str):
        # Models react_graph cancel checkpoints: poll the runner's event.
        ce = runner_holder["cancel_event"]
        for _ in range(200):  # up to ~2s — watchdog (0.05s) trips long before
            await asyncio.sleep(0.01)
            if ce.is_set():
                raise CancelledByEventError("checkpoint")
        return MagicMock(name="done_event")

    inner = MagicMock()
    inner.invoke_until_done = AsyncMock(side_effect=slow_invoke_with_checkpoints)
    budget = ChildBudget(
        max_tool_calls=25, max_token_cost_usd=0.5,
        max_wallclock_seconds=1,  # int field; the test monkey-shrinks below
    )
    runner, ce, publisher, envf = _mk_runner(inner_runner=inner, budget=budget)
    runner_holder["cancel_event"] = ce
    # Shrink the cap below int granularity for test speed: patch the started
    # watchdog's sleep via a tiny budget object stand-in is NOT possible on a
    # frozen dataclass — instead patch start_wallclock_watchdog to halve the
    # cap, preserving the production call shape (runner= + cap from budget).
    from app.application.services import coordinator_child_wallclock_watchdog as wd_mod

    real_start = wd_mod.start_wallclock_watchdog
    captured_kwargs: dict = {}

    def fast_start(**kwargs):
        captured_kwargs.update(kwargs)
        kwargs = dict(kwargs)
        kwargs["max_wallclock_seconds"] = 0.05
        return real_start(**kwargs)

    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner."
        "start_wallclock_watchdog",
        fast_start,
    )

    await _run(runner, ce)

    # The runner passed ITSELF + the budget's cap to the watchdog.
    assert captured_kwargs["runner"] is runner
    assert captured_kwargs["max_wallclock_seconds"] == 1
    # Trip routed to the budget finalizer with the wallclock label.
    assert runner._stop_reason == StopReason.WALLCLOCK_BUDGET
    terminal_publishes = [
        call for call in publisher.publish.await_args_list
        if getattr(call.args[0], "type", None) == MailboxEnvelopeType.RESULT_READY
    ]
    assert len(terminal_publishes) == 1
    payload = envf.make_result_ready.call_args.kwargs["payload"]
    assert payload.outcome == ResultReadyOutcome.NEEDS_AUTHORIZATION
    assert payload.needs_authorization_details.reason == "budget_exhausted"
    assert "stop_reason=wallclock_budget" in (
        payload.needs_authorization_details.observed_evidence or ""
    )


@pytest.mark.parametrize(
    "exit_mode",
    ["success", "cancelled_by_event", "timeout", "generic_exception"],
)
async def test_watchdog_cancelled_on_all_exit_paths(monkeypatch, exit_mode) -> None:
    """[spec §5-4 / INV-B2, R6#3 参数化四出口] On EVERY inner-invoke exit the
    watchdog task is cancelled BEFORE the finalizer publishes — the D2
    inner-finally runs ahead of the except arms by Python semantics. Kills
    the 'only cancel on success' mutation."""
    _patch_listener_and_prompt(monkeypatch)

    order: list[str] = []

    class _WdTask:
        def cancel(self) -> None:
            order.append("wd_cancel")

    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner."
        "start_wallclock_watchdog",
        lambda **_kw: _WdTask(),
    )

    inner = MagicMock()
    runner, ce, publisher, envf = _mk_runner(
        inner_runner=inner, budget=_mk_budget(wallclock=300),
    )
    if exit_mode == "success":
        inner.invoke_until_done = AsyncMock(return_value=MagicMock(name="done"))
    elif exit_mode == "cancelled_by_event":
        # In-flight trip: request_stop INSIDE the invoke (a pre-invoke
        # request_stop would short-circuit run_work_unit before the watchdog
        # block — the wrong path for this assertion).
        async def trip_then_raise(user_message: str):
            runner.request_stop(StopReason.TOKEN_BUDGET)
            raise CancelledByEventError("checkpoint")

        inner.invoke_until_done = AsyncMock(side_effect=trip_then_raise)
    elif exit_mode == "timeout":
        inner.invoke_until_done = AsyncMock(side_effect=asyncio.TimeoutError())
    else:  # generic_exception
        inner.invoke_until_done = AsyncMock(side_effect=RuntimeError("boom"))

    async def record_publish(envelope) -> None:
        order.append("publish")

    publisher.publish = AsyncMock(side_effect=record_publish)

    await _run(runner, ce)

    assert "wd_cancel" in order, f"watchdog never cancelled on {exit_mode}"
    assert "publish" in order, f"no terminal envelope on {exit_mode}"
    assert order.index("wd_cancel") < order.index("publish"), (
        f"INV-B2 violated on {exit_mode}: finalizer published before the "
        f"watchdog was cancelled (order={order})"
    )


async def test_zero_wallclock_silently_skips_watchdog(
    monkeypatch, caplog,
) -> None:
    """0 means unlimited: skip the watchdog without warning."""
    _patch_listener_and_prompt(monkeypatch)

    wd_spy = MagicMock()
    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner."
        "start_wallclock_watchdog",
        wd_spy,
    )
    inner = MagicMock()
    inner.invoke_until_done = AsyncMock(return_value=MagicMock(name="done"))
    runner, ce, publisher, _ = _mk_runner(
        inner_runner=inner, budget=_mk_budget(wallclock=0),
    )
    with caplog.at_level(
        logging.WARNING,
        logger="app.application.services.coordinator_child_runner",
    ):
        await _run(runner, ce)

    wd_spy.assert_not_called()
    publisher.publish.assert_awaited_once()  # run completed + published
    assert not [
        r
        for r in caplog.records
        if "wallclock" in r.getMessage().lower()
        and "disabled" in r.getMessage().lower()
    ], f"zero must be silent; got {[r.getMessage() for r in caplog.records]}"


async def test_negative_wallclock_skips_watchdog_with_warning(
    monkeypatch, caplog,
) -> None:
    """Negative direct-construction values stay disabled and observable."""
    _patch_listener_and_prompt(monkeypatch)

    wd_spy = MagicMock()
    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner."
        "start_wallclock_watchdog",
        wd_spy,
    )
    inner = MagicMock()
    inner.invoke_until_done = AsyncMock(return_value=MagicMock(name="done"))
    runner, ce, publisher, _ = _mk_runner(
        inner_runner=inner, budget=_mk_budget(wallclock=-1),
    )
    with caplog.at_level(
        logging.WARNING,
        logger="app.application.services.coordinator_child_runner",
    ):
        await _run(runner, ce)

    wd_spy.assert_not_called()
    publisher.publish.assert_awaited_once()
    assert any(
        "wallclock" in r.getMessage().lower() and "disabled" in r.getMessage().lower()
        for r in caplog.records
    ), f"expected DISABLED warning; got {[r.getMessage() for r in caplog.records]}"
