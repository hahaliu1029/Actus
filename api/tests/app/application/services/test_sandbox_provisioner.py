"""Tests for the on_demand sandbox provisioner (SPM PR-1c).

This module starts with the ``SandboxProvisionError`` typed error + the
tool-wrapper three-class envelope mapping (Task 13). Task 14 grows the
provisioner state-machine + OnDemand accessor tests here.

Async runner note: this repo ships **pytest-anyio**, not pytest-asyncio (see
``tests/conftest.py``). The module-level ``pytestmark = pytest.mark.anyio`` +
a local ``anyio_backend`` fixture are authoritative; the task brief's
illustrative ``@pytest.mark.asyncio`` decorators are intentionally dropped
(under this repo's plugin set a bare ``asyncio`` marker would leave the
coroutine un-awaited → false-green). The marker is a no-op for the pre-existing
sync ``TestProvisionErrorEnvelope`` methods (anyio only intercepts coroutines).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from app.application.services.sandbox_accessors import (
    OnDemandBrowserAccessor,
    OnDemandSandboxAccessor,
)
from app.application.services.sandbox_lifecycle_service import SandboxLifecycleService
from app.application.services.sandbox_provisioner import (
    SandboxProvisioner,
    _WatchdogIdleGuard,
)
from app.domain.errors.sandbox_lifecycle import (
    SandboxProvisionError,
    SandboxProvisionInvalidated,
    SessionFinalizedError,
    SessionSuspendedError,
    SessionUnboundError,
)
from app.domain.services.execution_watchdog import ExecutionWatchdog, WatchdogVerdict
from app.domain.services.tools.langchain_tools import _exception_outcome
from tests.app.application.services.test_sandbox_provision_flight import (
    UNBOUND,
    FakeRegistry,
    FakeSandboxCls,
    FakeUoW,
    _Fakes,
    _unbound_session,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class TestProvisionErrorEnvelope:
    def test_provision_error_maps_to_stable_code(self):
        exc = SandboxProvisionError("s1", phase="hooks", trigger="tool_call")
        out = _exception_outcome("file_read", exc)
        assert (
            out.reason.code == "SANDBOX_PROVISION_FAILED"
            and out.reason.type == "exception"
            and out.retryable is True
        )

    def test_suspended_passthrough_maps_not_retryable(self):
        out = _exception_outcome("shell_execute", SessionSuspendedError("s1"))
        assert (
            out.reason.code == "SANDBOX_SUSPENDED"
            and out.reason.type == "exception"
            and out.retryable is False
        )

    def test_finalized_maps(self):
        out = _exception_outcome("file_write", SessionFinalizedError("s1", destroyed_at=None))
        assert (
            out.reason.code == "SANDBOX_FINALIZED"
            and out.reason.type == "exception"
            and out.retryable is False
        )

    def test_generic_exception_unchanged(self):
        out = _exception_outcome("t", RuntimeError("x"))
        assert out.reason.code == "RuntimeError" and out.retryable is False  # 既有行为不动

    def test_provision_error_rejects_bad_phase(self):
        with pytest.raises(ValueError):
            SandboxProvisionError("s1", phase="bogus", trigger="tool_call")


# ── Task 14: async provisioner fakes + fixtures ───────────────────────────────


class _FakeProvBrowser:
    """Duck-typed ``Browser`` — records ``aclose()``. ``get_browser`` hands a
    fresh instance each call (mirrors ``docker_sandbox.py:530-532``)."""

    def __init__(self) -> None:
        self.aclosed = False

    async def aclose(self) -> None:
        self.aclosed = True


class _FakeProvHandle:
    """Duck-typed ``SandboxHandle``. ``release()`` decrements the owning fake
    lifecycle's open-handle counter so leak assertions are exact."""

    def __init__(self, lifecycle: "_FakeProvLifecycle") -> None:
        self._lifecycle = lifecycle
        self.ensure_raises: BaseException | None = None
        self.ensure_calls = 0
        self.get_browser_calls = 0
        self.released = 0

    async def ensure_sandbox(self) -> None:
        self.ensure_calls += 1
        if self.ensure_raises is not None:
            raise self.ensure_raises

    async def get_browser(self) -> _FakeProvBrowser:
        self.get_browser_calls += 1
        return _FakeProvBrowser()

    def release(self) -> None:
        self.released += 1
        self._lifecycle._open = max(0, self._lifecycle._open - 1)


class _FakeProvLifecycle:
    """Programmable stand-in for ``SandboxLifecycleService`` (only ``acquire`` /
    ``bind_new`` / ``resume`` are exercised by the provisioner).

    Default: ``acquire`` raises ``SessionUnboundError`` so the provisioner falls
    to ``bind_new``, which returns ``self.handle`` (and books one open handle).
    """

    def __init__(self) -> None:
        self._open = 0
        self.handle = _FakeProvHandle(self)
        self.bind_calls = 0
        self.resume_calls = 0
        # ── acquire knobs ──
        self.acquire_returns: _FakeProvHandle | None = None       # registry-hit path
        self.acquire_raises: BaseException | None = None
        self.acquire_raises_after_swallowed_cancel = False        # guard re-raise sim
        # ── bind_new knobs ──
        self.bind_raises: BaseException | None = None
        self.bind_gate: asyncio.Event | None = None
        self.bind_raises_after_gate: BaseException | None = None
        self.bind_hangs = False
        self.bind_cancelled = False

    async def acquire(self, session_id: str) -> _FakeProvHandle:
        if self.acquire_raises_after_swallowed_cancel:
            # Mirrors the PR-1a read-commit guard re-raising a swallowed cancel
            # (orphan _transition(DESTROYED) commit sub-window, R24-CLASS1b).
            raise asyncio.CancelledError()
        if self.acquire_raises is not None:
            raise self.acquire_raises
        if self.acquire_returns is not None:
            self._open += 1
            return self.acquire_returns
        raise SessionUnboundError(session_id)

    async def bind_new(
        self, session_id: str, *, user_id: str | None = None
    ) -> _FakeProvHandle:
        self.bind_calls += 1
        if self.bind_raises is not None:
            raise self.bind_raises
        if self.bind_gate is not None:
            await self.bind_gate.wait()
            if self.bind_raises_after_gate is not None:
                raise self.bind_raises_after_gate
        if self.bind_hangs:
            try:
                await asyncio.Event().wait()      # never set → hang until cancelled
            except asyncio.CancelledError:
                self.bind_cancelled = True         # cancel propagated INTO bind_new
                raise
        self._open += 1
        return self.handle

    async def resume(self, session_id: str) -> None:
        # INV-SPM-9: the provisioner must NEVER call this. Counted so the
        # pass-through test can assert zero calls.
        self.resume_calls += 1


class _FakeMetrics:
    """Recording ``SandboxProvisionMetrics`` fake — tests assert
    ``metrics.last().outcome``."""

    def __init__(self) -> None:
        self.records: list[SimpleNamespace] = []

    def record_provision(
        self, *, mode: str, trigger: str, outcome: str, latency: float | None
    ) -> None:
        self.records.append(
            SimpleNamespace(mode=mode, trigger=trigger, outcome=outcome, latency=latency)
        )

    def last(self) -> SimpleNamespace:
        return self.records[-1]


@dataclass
class _ProvFakes:
    """Test-facing handle over the fakes wired into the default provisioner."""

    lifecycle: _FakeProvLifecycle
    metrics: _FakeMetrics
    handle: _FakeProvHandle
    prov: SandboxProvisioner  # the fixture's default provisioner

    @property
    def registry_open_handles(self) -> int:
        return self.lifecycle._open

    def make_provisioner(
        self, *, hooks: list | None = None, timeout_seconds: float = 0.05
    ) -> SandboxProvisioner:
        """Build a fresh provisioner sharing the same fake lifecycle/metrics.
        Bare hook callables are registered via ``add_hook`` (default trigger)."""
        prov = SandboxProvisioner(
            session_id="s1",
            user_id="u1",
            lifecycle=self.lifecycle,
            timeout_seconds=timeout_seconds,
            trigger="tool_call",
            metrics=self.metrics,
        )
        for hook in hooks or []:
            prov.add_hook(hook)
        return prov

    def make_hook_fail_once(self) -> None:
        """Add a hook to the DEFAULT provisioner that raises on its first call →
        drives a ``hooks_failed`` state with a retained handle."""
        state = {"n": 0}

        async def hook(handle) -> None:
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("hook boom")

        self.prov.add_hook(hook)

    def provisioner_yields_new_handle(self) -> None:
        """Simulate ``unprovisioned → re-provision`` producing a NEW handle
        object: release the currently-held handle (state→unprovisioned) and arm
        the lifecycle to hand back a distinct handle on the next ``bind_new``."""
        self.lifecycle.handle = _FakeProvHandle(self.lifecycle)
        self.prov.release_held_handle()


@pytest.fixture
def prov_and_fakes() -> tuple[SandboxProvisioner, _ProvFakes]:
    lifecycle = _FakeProvLifecycle()
    metrics = _FakeMetrics()
    prov = SandboxProvisioner(
        session_id="s1",
        user_id="u1",
        lifecycle=lifecycle,
        timeout_seconds=0.05,
        trigger="tool_call",
        metrics=metrics,
    )
    fakes = _ProvFakes(
        lifecycle=lifecycle, metrics=metrics, handle=lifecycle.handle, prov=prov
    )
    return prov, fakes


@dataclass
class _RealLifecycleEnv:
    """A provisioner-over-REAL-``SandboxLifecycleService`` harness (fake sandbox
    class + fake UoW/registry reused from the flight facade)."""

    session_id: str
    svc: SandboxLifecycleService
    fakes: _Fakes

    def release_create(self) -> None:
        assert self.fakes.sandbox_cls.create_gate is not None
        self.fakes.sandbox_cls.create_gate.set()


@pytest.fixture
def real_lifecycle_env():
    """Factory building a fresh ``(SandboxLifecycleService, _Fakes)`` topology —
    mirrors ``svc_and_fakes`` from ``test_sandbox_provision_flight`` but returns
    an env object so a provisioner can drive the real PR-1a rollback/dispose."""

    def _build(*, create_hangs: bool = False) -> _RealLifecycleEnv:
        session_id = "sess-real"
        sessions = {session_id: _unbound_session(session_id)}
        events: list = []
        audit_rows: list[dict] = []
        uow = FakeUoW(sessions, events, audit_rows)
        sandbox_cls = FakeSandboxCls()
        registry = FakeRegistry()
        svc = SandboxLifecycleService(sandbox_cls=sandbox_cls, uow_factory=lambda: uow)
        svc._registry = registry  # type: ignore[assignment]
        fakes = _Fakes(
            session_id=session_id,
            uow=uow,
            sandbox_cls=sandbox_cls,
            registry=registry,
            events=events,
            audit_rows=audit_rows,
        )
        if create_hangs:
            sandbox_cls.create_gate = asyncio.Event()
        return _RealLifecycleEnv(session_id=session_id, svc=svc, fakes=fakes)

    return _build


# ── Task 14 test classes ──────────────────────────────────────────────────────


class TestProvisionerSingleFlight:
    async def test_orphan_transition_cancel_records_cancelled_not_failed(self, prov_and_fakes):
        """r24/codex R24-CLASS1b: acquire→registry-miss→orphan _transition(DESTROYED)
        commit cancel swallowed → lifecycle guard re-raises CancelledError →
        provisioner ``except asyncio.CancelledError`` records ``cancelled`` (NOT the
        ``except Exception`` failed)."""
        prov, fakes = prov_and_fakes
        fakes.lifecycle.acquire_raises_after_swallowed_cancel = True
        with pytest.raises(asyncio.CancelledError):
            await prov.get()
        assert fakes.metrics.last().outcome == "cancelled"

    async def test_concurrent_gets_share_one_provision(self, prov_and_fakes):
        prov, fakes = prov_and_fakes
        fakes.lifecycle.bind_gate = asyncio.Event()
        gathered = asyncio.gather(prov.get(), prov.get(), prov.get())
        await asyncio.sleep(0)
        fakes.lifecycle.bind_gate.set()
        r = await asyncio.wait_for(gathered, timeout=2)
        assert fakes.lifecycle.bind_calls == 1 and len(set(map(id, r))) == 1

    async def test_invalidated_flight_translates_to_finalized(self, prov_and_fakes):
        """spec R10①: bind_new raises SandboxProvisionInvalidated → translated to
        SessionFinalizedError (non-retryable, tool-face SANDBOX_FINALIZED)."""
        prov, fakes = prov_and_fakes
        fakes.lifecycle.bind_raises = SandboxProvisionInvalidated("s1", "delete")
        with pytest.raises(SessionFinalizedError):
            await prov.get()
        assert prov.state == "unprovisioned"

    async def test_failure_reported_to_all_waiters_and_retryable(self, prov_and_fakes):
        prov, fakes = prov_and_fakes
        fakes.lifecycle.bind_raises = RuntimeError("boom")
        with pytest.raises(SandboxProvisionError) as e1:
            await prov.get()
        assert e1.value.phase == "create" and prov.state == "unprovisioned"
        fakes.lifecycle.bind_raises = None  # failure not cached, retryable (INV-SPM-4)
        assert await prov.get() is fakes.handle

    async def test_concurrent_failure_same_report_single_bind_no_stuck_waiter(self, prov_and_fakes):
        """INV-SPM-4 concurrent face: 3 waiters same report, single bind, no stuck waiter."""
        prov, fakes = prov_and_fakes
        fakes.lifecycle.bind_gate = asyncio.Event()
        fakes.lifecycle.bind_raises_after_gate = RuntimeError("boom")
        results = asyncio.gather(prov.get(), prov.get(), prov.get(), return_exceptions=True)
        await asyncio.sleep(0)
        fakes.lifecycle.bind_gate.set()
        r = await asyncio.wait_for(results, timeout=2)
        assert fakes.lifecycle.bind_calls == 1
        assert all(isinstance(x, SandboxProvisionError) for x in r)
        assert len({x.attempt for x in r}) == 1  # same attempt (one provision)

    async def test_last_waiter_cancel_propagates_into_bind(self, prov_and_fakes):
        """codex planR1#2: ALL waiters cancel (run-stop/watchdog) → cancel propagates into bind_new."""
        prov, fakes = prov_and_fakes
        fakes.lifecycle.bind_hangs = True
        t1 = asyncio.create_task(prov.get())
        t2 = asyncio.create_task(prov.get())
        await asyncio.sleep(0)
        t1.cancel()
        t2.cancel()
        for t in (t1, t2):
            with pytest.raises(asyncio.CancelledError):
                await t
        await asyncio.sleep(0.01)
        assert fakes.lifecycle.bind_cancelled

    async def test_single_waiter_cancel_keeps_flight_for_others(self, prov_and_fakes):
        prov, fakes = prov_and_fakes
        fakes.lifecycle.bind_gate = asyncio.Event()
        t1 = asyncio.create_task(prov.get())
        t2 = asyncio.create_task(prov.get())
        await asyncio.sleep(0)
        t1.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t1
        fakes.lifecycle.bind_gate.set()
        assert await t2 is fakes.handle


class TestProvisionerPassthroughAndPhases:
    async def test_suspended_passes_through_untouched(self, prov_and_fakes):
        """场景①核心: SUSPENDED passes through untouched, never resume (INV-SPM-9)."""
        prov, fakes = prov_and_fakes
        fakes.lifecycle.acquire_raises = SessionSuspendedError("s1")
        with pytest.raises(SessionSuspendedError):
            await prov.get()
        assert fakes.lifecycle.resume_calls == 0
        assert fakes.lifecycle.bind_calls == 0  # stop-then-tool-call does not revive

    async def test_acquire_hit_ensure_fail_is_phase_ready(self, prov_and_fakes):
        prov, fakes = prov_and_fakes
        fakes.lifecycle.acquire_returns = fakes.handle
        fakes.handle.ensure_raises = RuntimeError("dead")
        with pytest.raises(SandboxProvisionError) as e:
            await prov.get()
        assert e.value.phase == "ready"

    async def test_hooks_failure_keeps_handle_and_retries_hooks_only(self, prov_and_fakes):
        prov, fakes = prov_and_fakes
        boom = {"n": 0}

        async def flaky_hook(handle):
            boom["n"] += 1
            if boom["n"] == 1:
                raise RuntimeError("minio down")

        prov = fakes.make_provisioner(hooks=[flaky_hook])
        with pytest.raises(SandboxProvisionError) as e:
            await prov.get()
        assert e.value.phase == "hooks" and prov.state == "hooks_failed"
        await prov.get()  # only retries hooks
        assert fakes.lifecycle.bind_calls == 1 and boom["n"] == 2


class TestProvisionerTimeout:
    async def test_timeout_in_create_yields_typed_error(self, prov_and_fakes):
        """场景⑦ (provision half): timeout → phase=create typed error; container
        cleanup owned by lifecycle."""
        prov, fakes = prov_and_fakes  # timeout_seconds=0.05
        fakes.lifecycle.bind_hangs = True
        with pytest.raises(SandboxProvisionError) as e:
            await prov.get()
        assert e.value.phase == "create"
        assert fakes.lifecycle.bind_cancelled  # cancel propagated into bind_new

    async def test_timeout_through_real_lifecycle_cleans_container(self, real_lifecycle_env):
        """r3 (codex planR2 A-P2-4): provisioner(timeout=0.05) × REAL
        SandboxLifecycleService (fake sandbox create hangs) — cancel propagates into
        bind_new → BaseException rollback (PR-1a) → container dispose + binding UNBOUND."""
        env = real_lifecycle_env(create_hangs=True)
        prov = SandboxProvisioner(
            session_id=env.session_id,
            user_id="u1",
            lifecycle=env.svc,
            timeout_seconds=0.05,
            trigger="tool_call",
        )
        with pytest.raises(SandboxProvisionError) as e:
            await prov.get()
        assert e.value.phase == "create"
        env.release_create()  # late completion → late disposer cleans up
        await asyncio.sleep(0.01)
        assert env.fakes.binding_state(env.session_id) == UNBOUND
        assert all(c.destroyed for c in env.fakes.sandbox_cls.created)  # zero leak

    async def test_timeout_in_hooks_is_phase_hooks_and_hooks_failed(self, prov_and_fakes):
        prov, fakes = prov_and_fakes

        async def slow_hook(handle):
            await asyncio.sleep(10)

        prov = fakes.make_provisioner(hooks=[slow_hook], timeout_seconds=0.05)
        with pytest.raises(SandboxProvisionError) as e:
            await prov.get()
        assert e.value.phase == "hooks" and prov.state == "hooks_failed"


class TestPeekAndBrowserAccessor:
    async def test_peek_only_when_ready(self, prov_and_fakes):
        prov, fakes = prov_and_fakes
        assert prov.peek() is None
        await prov.get()
        assert prov.peek() is fakes.handle

    async def test_browser_cached_same_handle_zero_rebuild(self, prov_and_fakes):
        """r5: single-run handle identity is constant → zero browser rebuild."""
        prov, fakes = prov_and_fakes
        acc = OnDemandBrowserAccessor(OnDemandSandboxAccessor(prov))
        b1 = await acc.get()
        assert await acc.get() is b1 and fakes.handle.get_browser_calls == 1

    async def test_browser_rebuilds_when_handle_identity_changes(self, prov_and_fakes):
        """r5/r7: re-provision produces a NEW handle object → ``handle is self._handle``
        False (object-identity, not id()) → aclose old browser + rebuild. Defensive
        transition (no in-run regen trigger in a single run; see §4/spec §5.2b)."""
        prov, fakes = prov_and_fakes
        acc = OnDemandBrowserAccessor(OnDemandSandboxAccessor(prov))
        b1 = await acc.get()
        fakes.provisioner_yields_new_handle()
        b2 = await acc.get()
        assert b2 is not b1 and b1.aclosed

    async def test_browser_concurrent_first_calls_single_flight(self, prov_and_fakes):
        """r7/codex R6-F4: two concurrent browser first-calls do not each build an
        instance / lose one (get_browser returns a fresh instance each call)."""
        prov, fakes = prov_and_fakes
        acc = OnDemandBrowserAccessor(OnDemandSandboxAccessor(prov))
        b1, b2, b3 = await asyncio.gather(acc.get(), acc.get(), acc.get())
        assert b1 is b2 is b3
        assert fakes.handle.get_browser_calls == 1  # in-lock 2nd identity check → build once

    async def test_release_held_handle_covers_hooks_failed(self, prov_and_fakes):
        """r7/codex R6-F3: hooks_failed peek()=None but the provisioner still holds
        a handle → release_held_handle releases it, registry open-handle no leak."""
        prov, fakes = prov_and_fakes
        fakes.make_hook_fail_once()
        with pytest.raises(SandboxProvisionError):
            await prov.get()
        assert prov.state == "hooks_failed" and prov.peek() is None
        prov.release_held_handle()
        assert fakes.registry_open_handles == 0 and prov._handle is None

    async def test_acquire_hit_ensure_fail_no_leak(self, prov_and_fakes):
        """r7/codex R6-F3: acquire-hit + ensure fail (phase=ready) → self._handle
        already assigned → _reset_for_phase releases it, no registry leak."""
        prov, fakes = prov_and_fakes
        fakes.lifecycle.acquire_returns = fakes.handle
        fakes.handle.ensure_raises = RuntimeError("supervisord dead")
        with pytest.raises(SandboxProvisionError) as e:
            await prov.get()
        assert e.value.phase == "ready" and fakes.registry_open_handles == 0

    async def test_browser_peek_and_aclose(self, prov_and_fakes):
        prov, fakes = prov_and_fakes
        acc = OnDemandBrowserAccessor(OnDemandSandboxAccessor(prov))
        assert acc.peek() is None
        b1 = await acc.get()
        assert acc.peek() is b1
        await acc.aclose()
        assert b1.aclosed and acc.peek() is None
        await acc.aclose()  # idempotent no-op


class TestProvisionerReleaseSemantics:
    """PR-1b whole-PR audit carry-forward #1: post-release ``get()`` must be
    EXPLICIT (the Eager accessor's silent-None-after-release was flagged as a
    precedent NOT to inherit). ``release_held_handle()`` clears the handle +
    state=unprovisioned, so a subsequent ``get()`` RE-PROVISIONS cleanly."""

    async def test_release_held_handle_then_get_reprovisions(self, prov_and_fakes):
        prov, fakes = prov_and_fakes
        h1 = await prov.get()
        assert prov.state == "ready" and h1 is fakes.handle
        prov.release_held_handle()
        assert prov.state == "unprovisioned" and prov.peek() is None
        h2 = await prov.get()  # re-provisions cleanly (not a silent None)
        assert prov.state == "ready" and h2 is fakes.handle
        assert fakes.lifecycle.bind_calls == 2  # provisioned twice


# ── Task 18: watchdog idle-suppression wiring (INV-SPM-13) ────────────────────
#
# ``WatchdogVerdict`` is a str Enum {HEALTHY, SOFT_RECOVER, HARD_TERMINATE}
# (execution_watchdog.py). ``evaluate()`` checks total BEFORE the pause branch,
# so "pause suppresses idle but total still counts" is the existing behavior —
# the watchdog itself needs ZERO changes here.


def _is_kill_verdict(verdict) -> bool:
    return verdict is WatchdogVerdict.HARD_TERMINATE


class TestWatchdogSuppression:
    """INV-SPM-13: the on_demand provisioner pauses the flow's real
    ``ExecutionWatchdog`` idle evaluation for the WHOLE provision attempt
    (create + post-provision hooks) via the ``_WatchdogIdleGuard`` adapter, so a
    slow container cold-start is never misread as a stalled graph."""

    async def test_idle_paused_for_entire_get_including_hooks(self, prov_and_fakes):
        """Create fast + hook slow (0.2s >> 0.05 idle): HEALTHY mid-hooks (pause
        active) → after get() completes + idle re-elapses, SOFT_RECOVER then
        HARD_TERMINATE (guard resumed on success → evaluation live again).
        Guards against a false-green that only exercises the create segment."""
        _, fakes = prov_and_fakes
        wd = ExecutionWatchdog(total_timeout_seconds=600, idle_timeout_seconds=0.05)

        async def slow_hook(handle):
            await asyncio.sleep(0.2)  # >> idle

        prov = fakes.make_provisioner(hooks=[slow_hook], timeout_seconds=5)
        prov.bind_idle_guard(_WatchdogIdleGuard(wd, ("sandbox_provision", "s1")))
        get_task = asyncio.create_task(prov.get())
        await asyncio.sleep(0.1)  # mid-hooks
        assert wd.evaluate() is WatchdogVerdict.HEALTHY  # suppressed (pause branch)
        await get_task
        await asyncio.sleep(0.06)  # idle resumes counting after guard.resume()
        assert wd.evaluate() is WatchdogVerdict.SOFT_RECOVER  # first over-limit
        await asyncio.sleep(0.06)
        assert wd.evaluate() is WatchdogVerdict.HARD_TERMINATE  # second = terminate

    async def test_slow_create_not_idle_killed(self, prov_and_fakes):
        """INV-SPM-13 create-phase face: create blocked (>> idle) is suppressed
        (the ~25s real cold-start, scaled down to 0.1s create >> 0.05 idle)."""
        _, fakes = prov_and_fakes
        wd = ExecutionWatchdog(total_timeout_seconds=600, idle_timeout_seconds=0.05)
        fakes.lifecycle.bind_gate = asyncio.Event()  # block create until set
        prov = fakes.make_provisioner(timeout_seconds=5)
        prov.bind_idle_guard(_WatchdogIdleGuard(wd, ("k",)))
        get_task = asyncio.create_task(prov.get())
        await asyncio.sleep(0.1)  # create still blocked, >> idle
        assert not _is_kill_verdict(wd.evaluate())
        fakes.lifecycle.bind_gate.set()
        await get_task

    async def test_guard_resumed_on_failure_too(self, prov_and_fakes):
        """``finally: _guard_resume()`` runs on the failure path too → the pause
        key is not leaked (a leaked key would permanently blind idle detection)."""
        prov, fakes = prov_and_fakes
        wd = ExecutionWatchdog(total_timeout_seconds=600, idle_timeout_seconds=0.05)
        prov.bind_idle_guard(_WatchdogIdleGuard(wd, ("k",)))
        fakes.lifecycle.bind_raises = RuntimeError("boom")
        with pytest.raises(SandboxProvisionError):
            await prov.get()
        assert wd._idle_pause_keys == set()

    async def test_total_timeout_still_enforced_during_pause(self):
        """spec §5.2e: idle is suppressed but STILL counts toward total — pause
        does NOT exempt the total cap. ``evaluate()`` checks total before the
        pause branch; this locks that order (turns red if pause is moved ahead)."""
        wd = ExecutionWatchdog(total_timeout_seconds=0.05, idle_timeout_seconds=0.01)
        wd.pause_idle(("k",))
        await asyncio.sleep(0.08)
        assert wd.check_total_only() is True
        assert wd.evaluate() is WatchdogVerdict.HARD_TERMINATE


@pytest.fixture
def minimal_flow():
    """Minimal ``PlannerReActFlow`` for ``_create_execution_watchdog`` unit tests.

    Uses ``object.__new__`` (mirrors the ``_make_runner`` fake-self pattern in
    ``test_runner_watchdog_emit_notification``) and sets only the 4 attributes
    the method touches — it never triggers graph execution."""
    from app.domain.services.flows.planner_react import PlannerReActFlow

    def _build(*, on_execution_watchdog, child_permission_context):
        flow = object.__new__(PlannerReActFlow)
        flow._on_execution_watchdog = on_execution_watchdog
        flow._child_permission_context = child_permission_context
        # Mirrors ctor default (planner_react.py:421); the merged condition
        # also suppresses the watchdog for mailbox-liveness-managed subagents.
        flow._mailbox_liveness_managed = False
        flow._execution_config = SimpleNamespace(
            total_timeout_seconds=600.0, idle_timeout_seconds=120.0
        )
        return flow

    return _build


@pytest.fixture
def runner_env(monkeypatch):
    """Build a full ``AgentTaskRunner`` with a spy ``PlannerReActFlow`` ctor that
    captures its kwargs. ``mode="on_demand"`` wires a real ``SandboxProvisioner``
    (over the fake lifecycle); ``mode="always"`` wires none. Mirrors the
    monkeypatch pattern in ``test_agent_task_runner_apply_preselected_skills``."""
    from unittest.mock import MagicMock

    from app.application.services.sandbox_accessors import (
        EagerBrowserAccessor,
        EagerSandboxAccessor,
    )
    from app.domain.models.app_config import A2AConfig, AgentConfig, MCPConfig
    from app.domain.services.agent_task_runner import AgentTaskRunner

    class _DummyFlow:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._overflow_config = SimpleNamespace(tool_result_max_chars=8000)
            self._assembler = None
            self._telemetry = None
            self._memory_config = SimpleNamespace(half_life_days=30, mmr_lambda=0.5)

        async def close(self):
            pass

    monkeypatch.setattr(
        "app.domain.services.agent_task_runner.PlannerReActFlow", _DummyFlow
    )

    class _FakeSandbox:
        async def ensure_sandbox(self):
            return None

    def _build(*, mode: str):
        session_id = "wd-wire-s1"
        provisioner = None
        if mode == "on_demand":
            provisioner = SandboxProvisioner(
                session_id=session_id,
                user_id="u1",
                lifecycle=_FakeProvLifecycle(),
                timeout_seconds=5,
                trigger="tool_call",
            )
        runner = AgentTaskRunner(
            uow_factory=lambda: MagicMock(),
            llm=object(),
            agent_config=AgentConfig(
                max_iterations=100, max_retries=3, max_search_results=10
            ),
            mcp_config=MCPConfig(mcpServers={}),
            a2a_config=A2AConfig(a2a_servers=[]),
            session_id=session_id,
            user_id="u1",
            file_storage=object(),
            browser_accessor=EagerBrowserAccessor(object()),
            search_engine=object(),
            sandbox_accessor=EagerSandboxAccessor(_FakeSandbox()),
            sandbox_provision_mode=mode,
            sandbox_provisioner=provisioner,
        )
        return SimpleNamespace(
            runner=runner,
            flow_ctor_kwargs=runner._flow.kwargs,
            provisioner=provisioner,
            session_id=session_id,
        )

    return _build


class TestWatchdogWiring:
    """The flow→runner→provisioner binding path itself. Without these, every
    suppression test above could pass while the real wire stays disconnected
    (they all bind the guard by hand)."""

    async def test_runner_passes_callback_to_flow_when_provisioner_present(self, runner_env):
        r = runner_env(mode="on_demand")
        assert callable(r.flow_ctor_kwargs["on_execution_watchdog"])

    async def test_runner_omits_callback_without_provisioner(self, runner_env):
        r = runner_env(mode="always")
        assert r.flow_ctor_kwargs.get("on_execution_watchdog") is None

    async def test_callback_binds_guard_onto_provisioner(self, runner_env):
        r = runner_env(mode="on_demand")
        wd = ExecutionWatchdog(total_timeout_seconds=600, idle_timeout_seconds=0.05)
        r.flow_ctor_kwargs["on_execution_watchdog"](wd)  # flow calls it post-create
        prov = r.provisioner
        prov._guard_pause()  # guard bound → pause reaches wd
        assert ("sandbox_provision", r.session_id) in wd._idle_pause_keys
        prov._guard_resume()
        assert wd._idle_pause_keys == set()

    def test_flow_create_watchdog_invokes_callback_with_same_instance(self, minimal_flow):
        """Behavioral: ``_create_execution_watchdog()`` return value IS the object
        handed to the callback (non-child path)."""
        seen: list = []
        flow = minimal_flow(on_execution_watchdog=seen.append, child_permission_context=None)
        wd = flow._create_execution_watchdog()
        assert wd is not None and seen == [wd]

    def test_flow_coordinator_child_gets_no_watchdog_and_no_callback(self, minimal_flow):
        """D5 condition lock: coordinator child (``_child_permission_context`` non-None)
        → watchdog=None AND callback NOT invoked (never idle-kill a long-lived child)."""
        seen: list = []
        flow = minimal_flow(
            on_execution_watchdog=seen.append, child_permission_context=object()
        )
        assert flow._create_execution_watchdog() is None and seen == []

    def test_flow_skips_callback_when_none(self, minimal_flow):
        """None callback → no crash (the ``and self._on_execution_watchdog`` guard)."""
        flow = minimal_flow(on_execution_watchdog=None, child_permission_context=None)
        assert flow._create_execution_watchdog() is not None

    def test_invoke_site_uses_extracted_method(self):
        """Anchor guard: the ONLY direct ``ExecutionWatchdog(...)`` construction in
        the class lives inside ``_create_execution_watchdog`` (nobody bypasses it)."""
        import inspect

        from app.domain.services.flows.planner_react import PlannerReActFlow

        src = inspect.getsource(PlannerReActFlow)
        assert src.count("ExecutionWatchdog(") == 1
