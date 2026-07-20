"""Per-pod registry of MailboxSupervisor tasks (C3 spec §3.2 M8 + §6.2).

Detects supervisor task crash via ``asyncio.Task.exception()`` and restarts
on the same root with a new instance_id. Does NOT handle pod-level recovery
(``reconcile_orphans`` owns that, per spec §3.2 M8 separation).

Layer note: this module lives in ``application/`` because it depends on
``MailboxSupervisor`` (also application). ``SupervisorRegistry`` itself is
infrastructure-light — just an asyncio.Task bag + a restart loop.

PR-3c will wire the registry into FastAPI lifespan + the agent_task_runner
spawn path; PR-3b ships the registry shell + crash-recovery loop.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:  # pragma: no cover — used only for type annotations
    from app.application.services.mailbox_supervisor import MailboxSupervisor


logger = logging.getLogger(__name__)


HealthState = str  # one of: "alive" | "crashed" | "restarting"


@dataclass
class _SupervisorSlot:
    """In-memory registry entry for one root_session_id.

    Mutated by ``spawn`` / ``_restart_crashed`` to swap in the new task +
    instance_id when the previous supervisor crashes. ``restart_count``
    is monotonic over the slot's lifetime — past
    ``_max_restart_count`` we stop trying (avoid restart-storm DoS).
    ``terminal_logged`` is the codex r1 [P2] log-spam guard: once we've
    surfaced the terminal "crash limit reached" line, suppress repeats on
    every subsequent restart tick so a dead supervisor isn't a continuous
    log noise source.
    """

    root_session_id: str
    supervisor: "MailboxSupervisor"
    task: asyncio.Task
    instance_id: str
    restart_count: int = 0
    terminal_logged: bool = False


class SupervisorRegistry:
    """Per-pod registry of MailboxSupervisor asyncio.Tasks.

    Spawn-then-watch model:

    1. ``spawn(root)`` constructs a ``MailboxSupervisor`` via the injected
       factory, awaits its ``_ready_event`` (so caller sees XGROUP CREATE
       complete), and pins the resulting ``asyncio.Task`` into a slot.
    2. A single background ``_restart_loop`` polls every
       ``restart_interval_s`` seconds and resurrects any slot whose task
       has terminated unexpectedly. Cancelled tasks (clean shutdown via
       ``stop()`` / ``stop_all()``) are NOT resurrected.
    3. ``health_check()`` exposes the current state for ops dashboards.

    Hard rules:
    - Idempotent ``spawn`` per root (calling twice is a no-op).
    - ``stop_all()`` cancels every task + the background loop — required
      for graceful pod shutdown so asyncio teardown doesn't warn.
    - Past ``max_restart_count`` restart attempts we log an error and
      leave the slot dead. Operator intervention required.
    """

    def __init__(
        self,
        supervisor_factory: Callable[[str], "MailboxSupervisor"],
        *,
        restart_interval_s: float = 5.0,
        max_restart_count: int = 5,
        ready_timeout_s: float = 5.0,
    ) -> None:
        self._factory = supervisor_factory
        self._slots: dict[str, _SupervisorSlot] = {}
        self._restart_interval_s = restart_interval_s
        self._max_restart_count = max_restart_count
        self._ready_timeout_s = ready_timeout_s
        self._restart_task: Optional[asyncio.Task] = None
        self._stopping = asyncio.Event()

    async def spawn(self, root_session_id: str) -> None:
        """Spawn a supervisor for ``root_session_id`` if one isn't already
        running. Idempotent — calling twice is a no-op.

        Blocks until the supervisor signals readiness via ``_ready_event``
        (XGROUP CREATE has completed) or ``ready_timeout_s`` elapses. The
        timeout path is non-fatal: XAUTOCLAIM at startup will pick up any
        envelopes that arrived during the gap.
        """
        if root_session_id in self._slots:
            return  # idempotent

        instance_id = uuid.uuid4().hex[:8]
        sup = self._factory(root_session_id)
        ready_event: asyncio.Event = asyncio.Event()
        # Inject the readiness event — MailboxSupervisor.run() checks for it.
        sup._ready_event = ready_event  # type: ignore[attr-defined]
        # C3 PR-5 (spec §11.6 rollback runbook) — inject the registry-side
        # ``stop`` so ``MailboxSupervisor._check_should_stop_for_rollback``
        # can pop this slot from ``self._slots`` AND cancel the run task
        # atomically. See ``_inject_stop_self_callback`` for the lifecycle
        # contract + fake-supervisor compatibility.
        self._inject_stop_self_callback(sup, root_session_id)
        task = asyncio.create_task(
            sup.run(),
            name=f"mailbox-sup:{root_session_id}:{instance_id}",
        )
        # Codex r1 [P0] fix — register the slot SYNCHRONOUSLY immediately
        # after create_task, BEFORE any await. The prior code put the slot
        # write after ``await asyncio.wait_for(ready_event.wait(), ...)``,
        # which is a cooperative yield point. Two concurrent
        # ``spawn(root_X)`` calls could both observe ``not in _slots``,
        # both create tasks, both suspend on the await, then both write
        # — last write wins and the first task becomes an orphan that
        # ``stop_all`` can't reach. Registering the slot the moment
        # create_task returns means the second caller's idempotent
        # short-circuit at the top fires and only one task is ever spawned
        # per root. Also covers the ``spawn`` cancellation case (slot is
        # findable so ``stop_all`` can cancel even if the readiness wait
        # is interrupted).
        self._slots[root_session_id] = _SupervisorSlot(
            root_session_id=root_session_id,
            supervisor=sup,
            task=task,
            instance_id=instance_id,
        )
        # Lazy-start the restart loop on first spawn — keeps the registry
        # cheap for tests that never spawn anything.
        if self._restart_task is None:
            self._restart_task = asyncio.create_task(
                self._restart_loop(), name="mailbox-sup-restart"
            )
        try:
            await asyncio.wait_for(ready_event.wait(), timeout=self._ready_timeout_s)
        except asyncio.TimeoutError:
            logger.warning(
                "supervisor not ready after %ss root=%s — continuing anyway "
                "(XAUTOCLAIM 0-0 will replay missed entries)",
                self._ready_timeout_s,
                root_session_id,
            )

    def _inject_stop_self_callback(self, sup, root_session_id: str) -> None:
        """C3 PR-5 (spec §11.6) — wire the registry-side ``stop`` into the
        supervisor's ``ctx.stop_self_callback`` so
        ``_check_should_stop_for_rollback`` can pop the slot and cancel
        the task atomically (otherwise ``self.stop()`` alone leaves the
        slot in place + ``_restart_crashed`` resurrects it).

        codex r1 [F1, HIGH TEST] — duck-typed registry fakes (see
        ``_FakeSupervisor`` in
        ``tests/app/application/services/test_supervisor_registry.py``)
        do not implement ``_ctx``. The injection therefore guards on
        ``hasattr(sup, "_ctx")`` and on the field existing on ctx
        (``SupervisorContext.stop_self_callback`` — a real
        ``MailboxSupervisor`` always has it; a partial fake may not).
        The fallback path inside
        ``MailboxSupervisor._check_should_stop_for_rollback`` handles
        the "no callback" case by setting ``self._stopping`` so the
        run loop exits on the next iteration (without popping the
        slot; the slot stays in ``_slots`` as ``"crashed"`` in
        ``health_check`` until ``stop_all`` / explicit removal). That
        is the right behaviour for tests using the fake (which do not
        exercise rollback).

        Same mutate-ctx-once-at-startup pattern as
        ``register_cancel_state`` / ``clear_child_tracking`` bound in
        ``MailboxSupervisor.__init__``; the supervisor only ever *calls*
        the hook. ``self.stop`` is idempotent (``pop(..., None)``) so a
        duplicate fire (rollback-check + parallel external stop) is
        safe.
        """
        ctx = getattr(sup, "_ctx", None)
        if ctx is None:
            return
        if not hasattr(ctx, "stop_self_callback"):
            return
        ctx.stop_self_callback = (
            lambda rid=root_session_id: self.stop(rid)
        )

    async def stop(self, root_session_id: str) -> None:
        """Drain + remove the supervisor for one root.

        External callers use ``MailboxSupervisor.stop()`` so an in-flight
        terminal-envelope side effect can finish before the run task exits.
        The rollback self-stop callback cannot await its own task, so that
        path only requests a stop and lets the run loop return naturally.
        """
        slot = self._slots.pop(root_session_id, None)
        if slot is None:
            return

        if asyncio.current_task() is slot.task:
            request_stop = getattr(slot.supervisor, "request_stop", None)
            if callable(request_stop):
                request_stop()
            else:  # pragma: no cover - compatibility for partial test fakes
                slot.task.cancel()
            return

        try:
            graceful_stop = getattr(slot.supervisor, "stop", None)
            if callable(graceful_stop):
                await graceful_stop()
        finally:
            # ``MailboxSupervisor.stop`` is bounded. If its drain timeout
            # expires, the registry still owns task cleanup and must not leave
            # an untracked run loop behind after popping the slot.
            if not slot.task.done():
                slot.task.cancel()
            try:
                await slot.task
            except (asyncio.CancelledError, Exception):
                pass

    async def stop_all(self) -> None:
        """Cancel every supervisor task + the restart loop. Required for
        clean pod shutdown — leaving tasks alive across pytest teardown
        produces "Task was destroyed but it is pending" warnings.
        """
        self._stopping.set()
        for slot in list(self._slots.values()):
            slot.task.cancel()
        for slot in list(self._slots.values()):
            try:
                await slot.task
            except (asyncio.CancelledError, Exception):
                pass
        self._slots.clear()
        if self._restart_task is not None:
            self._restart_task.cancel()
            try:
                await self._restart_task
            except (asyncio.CancelledError, Exception):
                pass
            self._restart_task = None

    async def health_check(self) -> dict[str, HealthState]:
        """Snapshot the current state of every slot.

        Returns ``{root_session_id: "alive" | "crashed"}``. ``"restarting"``
        is reserved for a future variant that surfaces a transient state
        while the restart loop is mid-resurrection; PR-3b uses only the
        two terminal states.
        """
        state: dict[str, HealthState] = {}
        for root, slot in self._slots.items():
            state[root] = "crashed" if slot.task.done() else "alive"
        return state

    # ──────────────────────────────────────────────────────────────────────
    # Internal — restart loop
    # ──────────────────────────────────────────────────────────────────────

    async def _restart_loop(self) -> None:
        """Background poller — ticks every ``restart_interval_s`` seconds
        and resurrects crashed slots. CancelledError is the only exit
        path; transient errors are logged + swallowed so the loop survives
        beyond a single bad supervisor.
        """
        try:
            while not self._stopping.is_set():
                await asyncio.sleep(self._restart_interval_s)
                if self._stopping.is_set():
                    return
                try:
                    await self._restart_crashed()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("supervisor restart sweep failed")
        except asyncio.CancelledError:
            raise

    async def _restart_crashed(self) -> None:
        """Single sweep — resurrect any slot whose task has terminated
        unexpectedly. Cancelled tasks (clean shutdown) are skipped; only
        tasks with an exception or normal-return terminations trigger a
        restart.
        """
        for root, slot in list(self._slots.items()):
            if not slot.task.done():
                continue
            # Skip CancelledError — that's a clean shutdown, not a crash.
            if slot.task.cancelled():
                continue
            exc = slot.task.exception()
            if slot.restart_count >= self._max_restart_count:
                # Codex r1 [P2] fix — log once on terminal entry, not every
                # restart_interval_s tick. Prior code spammed the error
                # line every 5s for the lifetime of the dead slot.
                if not slot.terminal_logged:
                    logger.error(
                        "supervisor crash limit reached root=%s restart_count=%d — "
                        "leaving slot dead, operator must intervene",
                        root,
                        slot.restart_count,
                    )
                    slot.terminal_logged = True
                continue
            slot.restart_count += 1
            new_instance_id = uuid.uuid4().hex[:8]
            sup = self._factory(root)
            # Inject a readiness event — but the restart path does NOT
            # block on it (the original spawn already validated this root
            # can come up; we just need the new task running ASAP).
            ready_event: asyncio.Event = asyncio.Event()
            sup._ready_event = ready_event  # type: ignore[attr-defined]
            # C3 PR-5 codex r1 [F2, HIGH CONTRACT] — restart-created
            # supervisors MUST get the same registry-side stop hook the
            # original ``spawn`` wired, otherwise
            # ``_check_should_stop_for_rollback`` would fall back to
            # ``self.stop()`` after a restart and the registry would
            # resurrect the slot on the next ``_restart_crashed`` tick
            # (spec §11.6 rollback would no longer be terminal).
            self._inject_stop_self_callback(sup, root)
            new_task = asyncio.create_task(
                sup.run(),
                name=f"mailbox-sup:{root}:{new_instance_id}",
            )
            slot.task = new_task
            slot.supervisor = sup
            slot.instance_id = new_instance_id
            logger.warning(
                "supervisor restarted root=%s instance=%s restart_count=%d "
                "prev_exception=%r",
                root,
                new_instance_id,
                slot.restart_count,
                exc,
            )
