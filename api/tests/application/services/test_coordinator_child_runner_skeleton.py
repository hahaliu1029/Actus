"""C2 PR-3 §8.3 — CoordinatorChildRunner skeleton tests.

Verifies the request_stop sole-entry semantics + StopReason carrier; PR-4
will expand with the full work-unit loop + finalizers.
"""
from __future__ import annotations

import asyncio
import pytest

from app.application.services.coordinator_child_runner import (
    CoordinatorChildRunner, StopReason,
)


class TestStopReasonValues:
    def test_values(self) -> None:
        assert StopReason.PARENT_CANCEL.value == "parent_cancel"
        assert StopReason.TOKEN_BUDGET.value == "token_budget"
        assert StopReason.WALLCLOCK_BUDGET.value == "wallclock_budget"

    def test_value_set_exact(self) -> None:
        assert {r.value for r in StopReason} == {
            "parent_cancel", "token_budget", "wallclock_budget",
        }


class TestRequestStopSemantics:
    def test_first_reason_wins_subsequent_ignored(self) -> None:
        ce = asyncio.Event()
        r = CoordinatorChildRunner(cancel_event=ce)
        r.request_stop(StopReason.TOKEN_BUDGET)
        assert r._stop_reason == StopReason.TOKEN_BUDGET
        assert ce.is_set()
        r.request_stop(StopReason.PARENT_CANCEL)
        # First setter wins: TOKEN_BUDGET preserved.
        assert r._stop_reason == StopReason.TOKEN_BUDGET

    def test_cancel_event_set_idempotent(self) -> None:
        ce = asyncio.Event()
        r = CoordinatorChildRunner(cancel_event=ce)
        assert not ce.is_set()
        r.request_stop(StopReason.PARENT_CANCEL)
        assert ce.is_set()
        r.request_stop(StopReason.PARENT_CANCEL)
        assert ce.is_set()

    def test_each_stop_reason_settable(self) -> None:
        for reason in StopReason:
            ce = asyncio.Event()
            r = CoordinatorChildRunner(cancel_event=ce)
            r.request_stop(reason)
            assert r._stop_reason == reason

    def test_first_wins_both_orders(self) -> None:
        """[spec §5-10b / INV-B4, R6#4] BOTH orders explicitly: budget-then-
        parent AND parent-then-budget — the second request_stop never
        overwrites _stop_reason, and the event stays set."""
        for first, second in [
            (StopReason.TOKEN_BUDGET, StopReason.PARENT_CANCEL),
            (StopReason.PARENT_CANCEL, StopReason.TOKEN_BUDGET),
        ]:
            ce = asyncio.Event()
            r = CoordinatorChildRunner(cancel_event=ce)
            r.request_stop(first)
            r.request_stop(second)
            assert r._stop_reason == first, f"{second} overwrote {first}"
            assert ce.is_set()


class TestSkeletonCtorAcceptsForwardDeps:
    def test_accepts_envelope_factory_and_subscriber(self) -> None:
        ce = asyncio.Event()
        r = CoordinatorChildRunner(
            cancel_event=ce,
            envelope_factory=object(),
            parent_session_id="p1",
            coordinator_run_id="r1",
            mailbox_subscriber=object(),
        )
        assert r._parent_session_id == "p1"
        assert r._coordinator_run_id == "r1"
        assert r._envelope_factory is not None
        assert r._mailbox_subscriber is not None

    # [C2 finish-core F1.4 / §5.1.2] child_sandbox Port — the ParentSandboxPort
    # over the child handle, consumed by seed-install (§5.1.4) + patch-extraction
    # (§5.1.5). Keyword-only with a None default so every existing caller
    # (skeleton/finalizer tests) stays constructible.
    def test_accepts_child_sandbox_port(self) -> None:
        import asyncio
        from unittest.mock import MagicMock
        from app.application.services.coordinator_child_runner import (
            CoordinatorChildRunner,
        )
        port = MagicMock()
        r = CoordinatorChildRunner(cancel_event=asyncio.Event(), child_sandbox=port)
        assert r._child_sandbox is port

    def test_accepts_budget_and_metrics_kwargs(self) -> None:
        """[C2b budget D2/D10] ctor accepts optional budget +
        coordinator_metrics; default None keeps every legacy caller valid."""
        import asyncio
        from unittest.mock import MagicMock

        from app.domain.services.permission.child_permission_context import (
            ChildBudget,
        )

        budget = ChildBudget(
            max_tool_calls=25, max_token_cost_usd=0.5, max_wallclock_seconds=300,
        )
        metrics = MagicMock()
        r = CoordinatorChildRunner(
            cancel_event=asyncio.Event(), budget=budget,
            coordinator_metrics=metrics,
        )
        assert r._budget is budget
        assert r._coordinator_metrics is metrics
        assert r._budget_callback is None

        legacy = CoordinatorChildRunner(cancel_event=asyncio.Event())
        assert legacy._budget is None
        assert legacy._coordinator_metrics is None

    def test_attach_budget_callback_stores_ref(self) -> None:
        import asyncio

        r = CoordinatorChildRunner(cancel_event=asyncio.Event())
        cb = object()
        r.attach_budget_callback(cb)
        assert r._budget_callback is cb


# [PR-4 Task 4.7] PR-3's `TestRunWorkUnitDeferredToPr4.test_raises_notimplemented`
# has been removed: run_work_unit is now a full implementation. Finalizer
# matrix coverage lives in tests/application/services/
# test_coordinator_child_runner_finalizers.py. Re-adding a NotImplementedError
# gate would block the PR-4 worker contract.
