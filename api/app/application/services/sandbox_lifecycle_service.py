"""Sandbox lifecycle service — single-writer for sandbox binding state (I3).

Must be initialized as a process-level singleton via FastAPI lifespan,
same pattern as ``checkpointer_pool`` (eng review decision #9).

PR1 scope: single worker only. Multi-worker coordination (Postgres
advisory lock + Redis pub/sub invalidation) deferred to §12 Q5 spec.

See docs/superpowers/specs/2026-04-15-sandbox-lifecycle-design.md §8.3.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Optional, Type, cast

from app.domain.errors.sandbox_lifecycle import (
    SandboxAlreadyDestroyed,
    SandboxBindingMissing,
    SandboxDaemonUnreachable,
    SandboxLifecycleError,
    SandboxProvisionInvalidated,
    SessionCreatingError,
    SessionDestroyingError,
    SessionFinalizedError,
    SessionSuspendedError,
    SessionUnboundError,
)
from app.application.services.sandbox_provision_flight import (
    FlightOutcome,
    ProvisionFlightTable,
)
from app.domain.external.policy_snapshot_sink import (
    NoopPolicySnapshotSink,
    PolicySnapshotSink,
)
from app.domain.external.sandbox import Sandbox, SandboxHandle
from app.domain.external.supervisor_registry import SupervisorRegistryPort
from app.domain.models.event import SandboxStateChangedEvent
from app.domain.models.session import (
    DestroyReason,
    SandboxBinding,
    SandboxBindingState,
)
from app.domain.repositories.uow import IUnitOfWork
from app.infrastructure.external.sandbox.sandbox_handle import SandboxHandleImpl
from app.infrastructure.external.sandbox.sandbox_registry import SandboxRegistry

logger = logging.getLogger(__name__)

# Shorthand aliases
UNBOUND = SandboxBindingState.UNBOUND
CREATING = SandboxBindingState.CREATING
ACTIVE = SandboxBindingState.ACTIVE
SUSPENDED = SandboxBindingState.SUSPENDED
DESTROYING = SandboxBindingState.DESTROYING
DESTROYED = SandboxBindingState.DESTROYED


def _flight_outcome_for(reason: DestroyReason) -> FlightOutcome:
    """Map a :class:`DestroyReason` to the flight-invalidation outcome (SPM Task 4).

    A *delete-class* reason — its enum name OR value contains ``"delete"``
    (case-insensitive; today only ``SESSION_DELETE`` / ``"session_delete"``) —
    yields ``"delete"`` so ``ProvisionFlightTable.invalidate`` registers a
    deletion tombstone. That tombstone refuses any later ``bind_new`` even after
    the session row is hard-deleted (covers the destroy-returns → row-hard-delete
    → second-bind_new window; spec §5.2c, DD-17).

    Every other reason (watchdog / reconcile-orphan / subagent-terminal /
    cancel-ack / orphan-timeout / force-terminate / terminal-child-reaper …) is an
    ordinary teardown and maps to ``"destroy"`` — signal an in-flight provision to
    abort, but leave NO tombstone (the session id may legitimately be re-bound).

    The name-OR-value substring test (not a hard-coded member allowlist) keeps the
    mapping correct if new delete-flavored members are introduced later.
    """
    haystack = f"{reason.name}\x00{reason.value}".lower()
    return "delete" if "delete" in haystack else "destroy"


class SandboxLifecycleService:
    """Sole authority for sandbox binding state transitions.

    External code accesses sandboxes exclusively through this service's
    ``acquire()`` / ``bind_new()`` / ``resume()`` / ``suspend()`` /
    ``destroy()`` API. Direct ``DockerSandbox.get()`` / ``.create()`` /
    ``.destroy()`` calls are forbidden outside this service and the registry
    (enforced by CI gate 1).

    **Nested acquire is undefined behavior**: asyncio.Lock is not reentrant.
    If tool X inside ``_checked_call`` triggers another ``acquire()`` on
    the same session_id, it will deadlock. Current injection pattern
    (agent_service acquires once then injects handle) avoids this.
    """

    def __init__(
        self,
        sandbox_cls: Type[Sandbox],
        uow_factory: Callable[[], IUnitOfWork],
        quiesce_timeout_seconds: float = 10.0,
        supervisor_registry: Optional[SupervisorRegistryPort] = None,
        sink: "PolicySnapshotSink | None" = None,        # C5a observe-only sink
        policy_snapshot_enabled: bool = False,           # C5a flag, captured ONCE (INV-0)
        runtime_hardening_enabled: bool = False,         # C5c flag, captured ONCE (INV-0)
    ) -> None:
        self._sandbox_cls = sandbox_cls
        self._uow_factory = uow_factory
        self._registry = SandboxRegistry()
        self._quiesce_timeout = quiesce_timeout_seconds
        self._per_session_locks: dict[str, asyncio.Lock] = {}
        # SPM Task 2/3: in-flight provision registry + deletion tombstones.
        # Owned exclusively by this service (INV-SPM-10).
        self._flights = ProvisionFlightTable()
        # SPM Task 3: strong refs to in-flight bind_new state-changing transition
        # tasks (CREATING/ACTIVE) + compensation tasks. Instance-level ONLY as a
        # GC guard (holds many sessions' tasks concurrently); the per-call
        # `pending_transition` bookkeeping is a bind_new call-stack local, NOT an
        # instance attribute (r16/codex R15-P2-1: an instance attr would be
        # clobbered by concurrent A/B sessions since the service is an app
        # singleton with per-session locks). Done-callback discards prevent leaks.
        self._bind_transition_tasks: set[asyncio.Task] = set()  # type: ignore[type-arg]
        # SPM Task 3 (late-disposer 要点2): strong refs to late container
        # disposers spawned when a create shield is cancelled — the container may
        # still finish building after the flight rolled back, so we dispose it
        # asynchronously once it lands. GC guard only.
        self._late_dispose_tasks: set[asyncio.Task] = set()  # type: ignore[type-arg]
        # C3 PR-3c: when injected, reconcile_orphans re-ensures a mailbox
        # supervisor task exists per root that still has in-flight subagents
        # on the mailbox plane. None keeps the legacy (pre-mailbox) behavior
        # so existing tests / smaller integration harnesses don't need to
        # pass a registry through.
        self._supervisor_registry: Optional[SupervisorRegistryPort] = (
            supervisor_registry
        )

        # C5a: observe-only policy-snapshot sink (default = no-op).
        self._policy_sink: PolicySnapshotSink = (
            sink if sink is not None else NoopPolicySnapshotSink()
        )
        # C5a flag captured at construction → bind_new's OFF path does ZERO work
        # (one bool check; NO get_settings call/import, NO helper, NO await) —
        # strict INV-0 byte-identical [codex planR4 P1]. Seam B reads the flag off
        # the already-present `_settings` so it pays nothing; Seam A (bind_new)
        # had NO get_settings call pre-C5a, so it must NOT add one on the OFF path.
        self._policy_snapshot_enabled = policy_snapshot_enabled
        # C5c flag captured at construction → bind_new's OFF path does ZERO extra
        # work (one bool check; NO get_settings/compile on the OFF path → INV-0).
        self._runtime_hardening_enabled = runtime_hardening_enabled

        # Single-worker runtime check (§8.6 layer 2)
        web_concurrency = os.environ.get("WEB_CONCURRENCY", "1")
        if web_concurrency != "1":
            raise RuntimeError(
                f"SandboxLifecycleService requires single-worker mode "
                f"(WEB_CONCURRENCY=1), got WEB_CONCURRENCY={web_concurrency}. "
                f"Multi-worker support requires §12 Q5 spec."
            )

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a per-session lock."""
        if session_id not in self._per_session_locks:
            self._per_session_locks[session_id] = asyncio.Lock()
        return self._per_session_locks[session_id]

    def _pop_lock_for(self, session_id: str) -> None:
        """Drop the per-session lock entry.

        Used by terminal-success ``destroy`` paths (missing row, UNBOUND,
        DESTROYED) and by the normal DESTROYED completion so that lock
        entries never leak for one-shot session ids (C3 PR-1 codex round 6
        P2).
        """
        self._per_session_locks.pop(session_id, None)

    # ── State transition helper ──

    async def _transition(
        self,
        session_id: str,
        *,
        target: SandboxBindingState,
        generation_delta: int = 0,
        sandbox_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        destroyed_at: Optional[datetime] = None,
        destroy_reason: Optional[DestroyReason] = None,
        event_reason: Optional[str] = None,
    ) -> SandboxBinding:
        """Persist a binding state transition via UoW.

        Also emits a ``SandboxStateChangedEvent`` to the session event stream
        so the frontend can react (PR2 §10.2).

        ``event_reason`` (SPM Task 1, spec DD-14): when provided, overrides the
        emitted event's ``reason`` **payload only** (e.g. ``"provision_failed"``
        / ``"provision_cancelled"``); the audit-log row keeps its legacy
        ``destroy_reason``-derived value. Omitted (default ``None``) → the event
        reason stays byte-identical to the pre-SPM derivation (INV-SPM-2).

        Returns the new SandboxBinding after commit.
        """
        async with self._uow_factory() as uow:
            session = await uow.session.get_by_id(session_id)
            if session is None:
                raise ValueError(f"Session {session_id} not found")

            old_binding = session.sandbox_binding
            new_binding = SandboxBinding(
                id=sandbox_id if sandbox_id is not None else old_binding.id,
                state=target,
                generation=old_binding.generation + generation_delta,
                created_at=created_at if created_at is not None else old_binding.created_at,
                destroyed_at=destroyed_at if destroyed_at is not None else old_binding.destroyed_at,
                destroy_reason=destroy_reason if destroy_reason is not None else old_binding.destroy_reason,
            )
            session.sandbox_binding = new_binding
            await uow.session.save(session)

            # Build SandboxStateChangedEvent (PR2 §10.2)
            reason_str = (
                new_binding.destroy_reason.value
                if new_binding.destroy_reason is not None
                else None
            )
            event = SandboxStateChangedEvent(
                old_state=old_binding.state.value,
                new_state=new_binding.state.value,
                generation=new_binding.generation,
                sandbox_id=new_binding.id,
                reason=event_reason if event_reason is not None else reason_str,
            )

            # Push to live SSE stream FIRST to obtain the Redis stream ID,
            # then persist to PG with that same ID. This ensures the event
            # has a single canonical ID across both channels so that
            # get_events_since() dedup won't treat them as two events.
            # Same pattern as agent_service._emit_control_event().
            sink = self._registry.get_live_event_sink(session_id)
            if sink is not None:
                try:
                    stream_id = await sink(event)
                    if stream_id:
                        event.id = stream_id
                except Exception:
                    logger.debug(
                        "Failed to push lifecycle event to live SSE sink "
                        "for session %s",
                        session_id,
                    )

            await uow.session.add_event(session_id, event)

            # Write audit log entry (PR2 §10.5)
            await uow.sandbox_lifecycle_log.create(
                session_id=session_id,
                old_state=old_binding.state.value,
                new_state=new_binding.state.value,
                generation=new_binding.generation,
                sandbox_id=new_binding.id,
                reason=reason_str,
            )

        logger.info(
            "sandbox binding transition session=%s %s→%s gen=%d",
            session_id,
            old_binding.state.value,
            new_binding.state.value,
            new_binding.generation,
        )
        return new_binding

    # ── Public API ──

    async def acquire(self, session_id: str) -> SandboxHandle:
        """Acquire a handle on the sandbox bound to session_id.

        Only accepts ACTIVE state. SUSPENDED must go through resume() first.

        Raises:
            SessionUnboundError: binding.state == UNBOUND
            SessionCreatingError: binding.state == CREATING
            SessionSuspendedError: binding.state == SUSPENDED
            SessionDestroyingError: binding.state == DESTROYING
            SessionFinalizedError: binding.state == DESTROYED
        """
        async with self._get_lock(session_id):
            return await self._acquire_locked(session_id)

    async def _acquire_locked(self, session_id: str) -> SandboxHandle:
        """Internal acquire — caller must hold per-session lock."""
        async with self._uow_factory() as uow:
            session = await uow.session.get_by_id(session_id)
        # read-commit 取消守卫（r20/R20-P2-T1）: provisioner.get() always calls
        # acquire()→_acquire_locked first, and this read UoW has the same
        # cancel-swallow commit sub-window as bind_new's. Honor a swallowed cancel
        # here (BEFORE registry acquire / rehydrate / hooks) so it can't slip past
        # into a handle assignment that races runner cleanup (last-waiter-cancel).
        self._raise_if_read_swallowed_cancel()
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        binding = session.sandbox_binding

        if binding.state == UNBOUND:
            raise SessionUnboundError(session_id)
        if binding.state == CREATING:
            raise SessionCreatingError(session_id)
        if binding.state == SUSPENDED:
            raise SessionSuspendedError(session_id)
        if binding.state == DESTROYING:
            raise SessionDestroyingError(session_id)
        if binding.state == DESTROYED:
            raise SessionFinalizedError(session_id, destroyed_at=binding.destroyed_at)

        # state == ACTIVE — try registry hit, then rehydrate
        sandbox = self._registry.get_sandbox(session_id)
        if sandbox is not None:
            if self._registry.get_generation(session_id) != binding.generation:
                self._registry.update_generation(session_id, binding.generation)
            return cast(SandboxHandle, self._registry.acquire_handle(session_id))

        # Registry miss — rehydrate or mark orphan
        return await self._rehydrate_or_mark_orphan(session_id, binding)

    async def bind_new(
        self, session_id: str, *, user_id: str | None = None
    ) -> SandboxHandle:
        """UNBOUND → CREATING → ACTIVE. Creates a new sandbox container.

        ``user_id`` drives the M1 memory bind-mount. Caller normally passes
        the authenticated user; if omitted we fall back to ``session.user_id``
        so old call sites keep working without a signature churn.

        Raises:
            SessionSuspendedError: if already SUSPENDED (use resume instead)
            SessionFinalizedError: if DESTROYED
            SandboxLifecycleError: if binding already has a sandbox
            SandboxProvisionInvalidated: if a concurrent destroy/delete/quiesce
                invalidated the in-flight provision at a CAS window (SPM §5.2c).

        SPM Task 3: this method integrates the ``ProvisionFlightTable`` with a
        BaseException-safe rollback + double CAS-invalidation check + create
        shield + late disposer + deletion tombstone entry. The happy-path
        branches remain behaviorally byte-equivalent to the pre-SPM ``always``
        mode (INV-SPM-2); the added scaffolding only hardens the exception /
        cancellation paths and threads ``session_id`` / ``attempt`` container
        metadata into ``create``.
        """
        # 场景⑤入口（锁外快速拒绝）: a delete already tombstoned this session, so
        # even a brand-new flight must be refused before we take the lock.
        if self._flights.is_tombstoned(session_id):
            raise SessionFinalizedError(session_id, destroyed_at=None)
        async with self._get_lock(session_id):
            async with self._uow_factory() as uow:                    # read UoW (原样)
                session = await uow.session.get_by_id(session_id)
            # read-commit 取消守卫（要点⑦, r19b/R19-P2 + r20/R20-P2-T1): the read
            # UoW `__aexit__` empty-transaction commit is a cancel-swallow
            # sub-window (`db_uow.py:71` logs but does NOT `uncancel()`), so a
            # cancel landing there leaves `cancelling() > 0` — honor it here,
            # AFTER the read UoW exits and BEFORE any state dispatch /
            # flight.begin() / container work, for a clean zero-side-effect abort.
            self._raise_if_read_swallowed_cancel()
            if session is None:
                raise ValueError(f"Session {session_id} not found")

            binding = session.sandbox_binding

            if binding.state == ACTIVE:
                # Already active — just acquire a handle
                return await self._acquire_locked(session_id)
            if binding.state == SUSPENDED:
                raise SessionSuspendedError(session_id)
            if binding.state in (DESTROYING, DESTROYED):
                raise SessionFinalizedError(session_id, destroyed_at=binding.destroyed_at)
            if binding.state == CREATING:
                raise SessionCreatingError(session_id)
            # UNBOUND — proceed to create

            # Resolve user_id: explicit arg wins, otherwise fall back to session
            effective_user_id = user_id if user_id is not None else session.user_id

            flight = self._flights.begin(session_id)   # 先登记再做任何 await
            # 复查 (FIX-A / P1-1): close the window between the entry tombstone
            # check and flight.begin() (codex planR1#1) — a delete arriving
            # in-between only lands a tombstone (no flight to mark). HOISTED out of
            # the try so a tombstone hit aborts CLEANLY: zero transition, zero
            # audit, zero container. If it stayed inside the try its raise would
            # run the ``except BaseException`` compensation and emit a spurious
            # UNBOUND→UNBOUND ``provision_failed`` event + audit row on a
            # still-UNBOUND binding. Nothing between begin() and here can raise or
            # await, so the explicit finish() (the finally now sits past the try
            # and no longer covers this path) cannot leak the flight.
            if self._flights.is_tombstoned(session_id):
                self._flights.finish(session_id)
                raise SessionFinalizedError(session_id, destroyed_at=None)
            sandbox: Optional[Sandbox] = None
            active_committed = False   # set to a reliable value after ACTIVE await-to-determinacy
            new_binding: Optional[SandboxBinding] = None
            # bind_new 调用栈局部变量（非 self.——app 单例 + per-session 锁, an instance
            # attr would be clobbered by concurrent A/B sessions; r16/codex R15-P2-1）:
            pending_transition: Optional[asyncio.Task] = None  # type: ignore[type-arg]
            # FIX-H: defined before the try so the compensation handler can always
            # reference it (deterministic-failure skip keys on creating_task).
            creating_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
            try:
                # Step 1: UNBOUND → CREATING. shield 不推迟外层取消 → 强引用 named
                # task + 取消后 await 到 done（镜像 _set_terminal_status）。CREATING 段
                # 补偿总回滚 UNBOUND（CREATING→UNBOUND 合法 / UNBOUND→UNBOUND 良性;
                # 容器未建 dispose no-op），故只需 await task 到 done 防与回滚并发。
                creating_task = asyncio.create_task(
                    self._transition(session_id, target=CREATING),
                    name=f"spm-bind-creating:{session_id}",  # FIX-D: diagnosable
                )
                self._bind_transition_tasks.add(creating_task)
                creating_task.add_done_callback(self._bind_transition_tasks.discard)
                pending_transition = creating_task
                await asyncio.shield(creating_task)
                pending_transition = None
                # —— flight 生命期自此覆盖 [CREATING 转移, ACTIVE commit/回滚]（§5.2c-1）——

                # Step 2: create the sandbox container. C5c ON path compiles a
                # hardened ContainerRuntimePolicy; OFF path stays call-shape
                # identical to pre-C5c except for the SPM session_id/attempt
                # metadata (Task 5 lands the real kwargs; here fakes drive it).
                if self._runtime_hardening_enabled:
                    from app.application.services.sandbox_runtime_policy import (
                        compile_runtime_policy,
                    )
                    from core.config import get_settings

                    create_coro = self._sandbox_cls.create(
                        user_id=effective_user_id,
                        runtime_policy=compile_runtime_policy(
                            get_settings(), worker_type=session.worker_type
                        ),
                        session_id=session_id,
                        attempt=flight.attempt,
                    )
                else:
                    create_coro = self._sandbox_cls.create(
                        user_id=effective_user_id,
                        session_id=session_id,
                        attempt=flight.attempt,
                    )
                create_task = asyncio.ensure_future(create_coro)
                try:
                    sandbox = await asyncio.shield(create_task)      # create shield（§5.2c-4）
                except BaseException:
                    # Outer cancel/failure at the shield: the shielded create may
                    # still finish building a container in the background → hand
                    # it to a late disposer so it never leaks (要点2).
                    self._spawn_late_disposer(session_id, flight, create_task)
                    raise
                if flight.invalidated:                               # CAS-1: after create returns
                    await self._dispose_container(sandbox)
                    raise SandboxProvisionInvalidated(session_id, flight.invalidated)
                await sandbox.ensure_sandbox()
                if flight.invalidated:                               # CAS-2: before ACTIVE transition
                    await self._dispose_container(sandbox)
                    raise SandboxProvisionInvalidated(session_id, flight.invalidated)

                # Step 3: CREATING → ACTIVE with generation++. shield 不推迟外层取消
                # → 强引用 named task + 取消后 await 到确定态 → 可靠 active_committed
                # （镜像 _set_terminal_status 强引用 terminal_task）。
                active_task = asyncio.create_task(
                    self._transition(
                        session_id,
                        target=ACTIVE,
                        generation_delta=1,  # I7 rule (a)
                        sandbox_id=sandbox.id,
                        created_at=datetime.now(UTC),
                    ),
                    name=f"spm-bind-active:{session_id}",  # FIX-D: diagnosable
                )
                self._bind_transition_tasks.add(active_task)         # 强引用防 GC + 观测异常
                active_task.add_done_callback(self._bind_transition_tasks.discard)
                pending_transition = active_task
                try:
                    new_binding = await asyncio.shield(active_task)  # happy path 返回 binding
                    active_committed = True
                    pending_transition = None
                except asyncio.CancelledError:
                    # 外层取消已立即抛到此; active_task（强引用）仍在跑 → await 到确定态:
                    await self._await_to_done(active_task)           # asyncio.wait, 不 shield
                    active_committed = (
                        not active_task.cancelled()
                        and active_task.exception() is None
                    )
                    pending_transition = None
                    raise                                            # 重抛外层取消
            except BaseException as exc:
                # 先把仍在跑的 pending transition await 到 done（不传播其异常——R16-P2:
                # transition 以 RuntimeError/commit-error 结束时旧 `await shield(pending)`
                # 会重抛该异常越过补偿）:
                if pending_transition is not None:
                    await self._await_to_done(pending_transition)
                    pending_transition = None
                # 只走合法状态边——状态机仅定义 CREATING→UNBOUND，无 ACTIVE→UNBOUND。
                if active_committed:
                    # ACTIVE commit 已落库 → binding 合法 ACTIVE、容器有效但 registry
                    # 未注册 → 既有 c-1 lazy-rehydrate 兜底（下次 acquire rehydrate），
                    # 不回滚、不 dispose（always parity: 完成 bind 后 run-cancel 留
                    # ACTIVE 沙箱由 session teardown / TTL 清）。仅 re-raise。
                    raise
                # 未 ACTIVE: CREATING 已 committed（确定）或更早 → 合法 CREATING→UNBOUND。
                reason = (
                    "provision_cancelled"
                    if isinstance(exc, (asyncio.CancelledError, SandboxProvisionInvalidated))
                    else "provision_failed"
                )

                # FIX-H: skip a provably-redundant rollback transition. When the
                # CREATING transition task itself failed with a NON-cancel
                # exception (commit error → its UoW rolled back → DB provably still
                # UNBOUND), a _transition(UNBOUND) here would only emit a phantom
                # UNBOUND→UNBOUND event + audit row. Skip JUST that transition in
                # this deterministic-failure case (the container dispose branch
                # still runs — sandbox is None here anyway, since create is never
                # reached). ALL cancel / ambiguous paths (creating_task cancelled,
                # or it committed CREATING and a later step failed) keep the
                # unconditional rollback — same-state rollback is benign there and
                # a committed CREATING genuinely needs CREATING→UNBOUND.
                creating_failed_deterministically = (
                    creating_task is not None
                    and creating_task.done()
                    and not creating_task.cancelled()
                    and creating_task.exception() is not None
                )

                async def _compensate() -> None:
                    if sandbox is not None and not isinstance(
                        exc, SandboxProvisionInvalidated
                    ):
                        await self._dispose_container(sandbox)       # 半成功清理（§5.2c-3）
                    if creating_failed_deterministically:
                        logger.debug(
                            "CREATING commit rolled back deterministically — "
                            "no rollback transition needed"
                        )
                        return
                    try:
                        await self._transition(
                            session_id, target=UNBOUND, event_reason=reason
                        )
                    except Exception:
                        logger.exception(
                            "bind_new rollback transition failed for %s", session_id
                        )

                # dispose + rollback 打包进一个强引用 compensation task + await-to-
                # determinacy（裸 await 抗不住重复取消: 二次取消可能跳过 rollback 留半
                # 容器，或 rollback 后台在 finally.finish + 释放锁后才写 UNBOUND）。
                comp_task = asyncio.create_task(
                    _compensate(), name=f"spm-bind-compensate:{session_id}"  # FIX-D
                )
                self._bind_transition_tasks.add(comp_task)
                comp_task.add_done_callback(self._bind_transition_tasks.discard)
                await self._await_to_done(comp_task)                 # 补偿完整落定
                raise                                                # 补偿后重抛原异常
            finally:
                self._flights.finish(session_id)                     # flight 生命期终点

            # commit 后（既有代码原样）: registry.register + policy snapshot + acquire_handle
            # happy path guaranteed by active_committed; explicit guard instead of
            # bare `assert` so it survives `python -O` (assert stripping).
            if new_binding is None:
                raise RuntimeError(
                    "bind_new: ACTIVE transition returned no binding"
                )
            self._registry.register(
                session_id, sandbox, generation=new_binding.generation
            )

            # C5a Seam A: observe-only container-create policy snapshot. INV-0 —
            # the flag was captured at construction, so the OFF path is a single
            # bool check: NO get_settings, NO import, NO coroutine, NO await.
            if self._policy_snapshot_enabled:
                await self._observe_container_policy(
                    session_id=session_id,
                    user_id=effective_user_id,
                    new_binding=new_binding,
                    session=session,
                    sandbox=sandbox,
                )

            return cast(SandboxHandle, self._registry.acquire_handle(session_id))

    # ── SPM Task 3 cancellation-safety helpers ──

    @staticmethod
    def _raise_if_read_swallowed_cancel() -> None:
        """Honor a cancellation swallowed by a read-only UoW's `__aexit__`.

        r19b/R19-P2 + r20/R20-P2-T1: a read-only UoW's `__aexit__` empty-
        transaction commit swallows `CancelledError` and does NOT `uncancel()`
        (`db_uow.py:71`✓), so `current_task().cancelling()` stays > 0 and is
        detectable after the fact. Call this in EVERY provisioner-reachable path
        right after a read UoW exits and before any side effect (flight.begin /
        registry acquire / rehydrate / transition) to convert that swallowed
        cancel into a clean abort. The cancel-lands-on-the-`await` case propagates
        naturally and never reaches here (cancelling() == 0 pre-read). Used by
        `bind_new`, `_acquire_locked`, and `_rehydrate_or_mark_orphan`.
        """
        t = asyncio.current_task()
        if t is not None and t.cancelling() > 0:
            raise asyncio.CancelledError()

    @staticmethod
    async def _await_to_done(task: "asyncio.Task") -> None:  # type: ignore[type-arg]
        """Wait until `task` is `done()` without propagating its result/exception
        and without being interrupted by repeated cancellation of the caller.

        r16/codex R16-P2: uses `asyncio.wait` (NOT `asyncio.shield`) — shield
        re-raises the task's exception, which would let a transition ending in a
        RuntimeError/commit-error jump over the compensation, or an inner
        exception mask the outer cancel. `asyncio.wait` waits to `done` but never
        raises the task's exception; the caller inspects
        `task.cancelled()`/`task.exception()` and keeps its own outer exception.
        """
        while not task.done():
            try:
                await asyncio.wait({task})     # 等 done; task 异常不在此 raise
            except asyncio.CancelledError:
                continue                        # 我方再被取消 → 继续等 task 完成
        # Mark the task's exception "retrieved" so it does not resurface via the
        # event loop's default handler (which also fails strict test loops). The
        # compensation caller does NOT inspect it (unlike the ACTIVE inner
        # handler), so retrieve it here. A cancelled task has nothing to retrieve
        # (task.exception() would re-raise CancelledError), so guard on it.
        if not task.cancelled():
            task.exception()

    async def _dispose_container(self, sandbox: "Sandbox | None") -> None:
        """Best-effort container teardown (要点1). `destroy()` is the Sandbox
        protocol's existing teardown method; swallow any error so disposal can
        never mask the root cause that triggered it."""
        if sandbox is None:
            return
        try:
            await sandbox.destroy()
        except Exception:
            logger.warning(
                "bind_new container dispose failed for sandbox %s",
                getattr(sandbox, "id", "?"),
            )

    async def _best_effort_remove_container(self, binding_id: Optional[str]) -> None:
        """FIX-C (P1-2 hardening): physically remove a container before finalizing
        DESTROYED on a strict-probe ``None``.

        ``get_strict`` returns ``None`` both when a container is truly gone
        (NotFound) AND when one EXISTS in a non-running (exited/paused) state — in
        the latter case finalizing DESTROYED without removal leaves the container
        on disk until the next startup label sweep. Removing here is cheap and
        idempotent (``remove_container`` swallows NotFound itself); any OTHER
        failure only logs and we STILL finalize (same net behavior as before).

        Guarded on ``remove_container`` presence + a non-null ``binding_id`` so
        external-sandbox / legacy fakes and the ``binding.id is None`` orphan path
        are a no-op. Never raises (must not perturb the DESTROYED finalize)."""
        if not binding_id:
            return
        remover = getattr(self._sandbox_cls, "remove_container", None)
        if remover is None:
            return
        try:
            await asyncio.to_thread(remover, binding_id)
        except Exception:
            logger.warning(
                "best-effort container removal before DESTROYED finalize failed "
                "for %s; finalizing anyway",
                binding_id,
                exc_info=True,
            )

    def _spawn_late_disposer(
        self, session_id: str, flight, create_task: "asyncio.Task"  # type: ignore[type-arg]
    ) -> None:
        """要点2: the create shield was cancelled/failed while the shielded create
        task may still be building a container in the background. Spawn a
        detached disposer that awaits the container and destroys it once it lands
        (the flight has already rolled back, so the container is ownerless).
        Strong ref held in `self._late_dispose_tasks` (GC guard) + done-discard."""
        task = asyncio.create_task(
            self._late_dispose(session_id, flight, create_task),
            name=f"spm-late-dispose:{session_id}",  # FIX-D: diagnosable
        )
        self._late_dispose_tasks.add(task)
        task.add_done_callback(self._late_dispose_tasks.discard)

    async def _late_dispose(
        self, session_id: str, flight, create_task: "asyncio.Task"  # type: ignore[type-arg]
    ) -> None:
        """Await the ownerless container from a cancelled create shield, then
        dispose it. Swallow the create task's own failure (nothing to clean up)."""
        try:
            sandbox = await create_task
        except BaseException:
            return
        await self._dispose_container(sandbox)

    async def _observe_container_policy(
        self, *, session_id: str, user_id: str | None, new_binding, session, sandbox
    ) -> None:
        """C5a/C5c container-create policy emission (additive, best-effort).

        Only ever called when the C5a snapshot flag is ON (gated by the caller).
        C5c: when the hardening flag is on AND the sandbox carries an applied
        runtime policy (a real _create_task ran), emit the honest applied/enforce
        snapshot; otherwise (hardening off, external_address, or a fake lacking the
        property) fall back to the C5a configured/observe_only snapshot. The
        ``applied_runtime_policy`` read is GUARDED via getattr so the C5a observe
        tests (FakeSandbox has no such property) keep passing. Swallows ALL errors
        so an observe failure can never fail a sandbox bind (INV-0). Reads
        get_settings() LAZILY (ON path only).
        """
        try:
            from app.domain.models.sandbox_policy import (
                ContainerCreateInput,
                build_settings_view,
            )
            from app.domain.services.safety.sandbox_policy_compiler import (
                SandboxPolicyCompiler,
            )
            from core.config import get_settings  # lazy — reached only on the ON path

            inp = ContainerCreateInput(
                session_id=session_id,
                user_id=user_id,
                sandbox_id=new_binding.id,  # SandboxBinding.id == sandbox.id
                sandbox_generation=new_binding.generation,
                worker_type=session.worker_type,
                depth=session.depth,
                settings=build_settings_view(get_settings()),
            )
            compiler = SandboxPolicyCompiler()
            applied = (
                getattr(sandbox, "applied_runtime_policy", None)
                if self._runtime_hardening_enabled
                else None
            )
            if applied is not None:
                snapshot = compiler.compile_applied_container_create(applied, inp)
            else:
                snapshot = compiler.compile_container_create(inp)
            await self._policy_sink.record(snapshot)
        except Exception as exc:  # noqa: BLE001 — observe must never fail a bind
            logger.warning(
                "sandbox.policy observe failed surface=container_create exc=%s",
                type(exc).__name__,
            )

    async def suspend(self, session_id: str) -> None:
        """ACTIVE → SUSPENDED. Container stays alive, can resume later.

        Generation does NOT increment (I7: ACTIVE ↔ SUSPENDED doesn't
        poison holders).
        """
        # SPM Task 4 (intent-before-lock): signal any in-flight provision to abort
        # with ``"quiesce"`` BEFORE contending for the per-session lock, so the
        # flight's CAS checks observe the intent and roll back instead of racing
        # this suspend. ``"quiesce"`` never tombstones — the session may be resumed
        # / re-bound later. No active flight → harmless no-op (INV-SPM-2).
        self._flights.invalidate(session_id, "quiesce")
        async with self._get_lock(session_id):
            async with self._uow_factory() as uow:
                session = await uow.session.get_by_id(session_id)
            if session is None:
                raise ValueError(f"Session {session_id} not found")

            binding = session.sandbox_binding
            if binding.state == SUSPENDED:
                return  # Already suspended, idempotent
            if binding.state != ACTIVE:
                raise SandboxLifecycleError(
                    f"Cannot suspend session {session_id}: "
                    f"state is {binding.state.value}, expected ACTIVE"
                )

            await self._transition(session_id, target=SUSPENDED, generation_delta=0)

    async def resume(self, session_id: str) -> SandboxHandle:
        """SUSPENDED → ACTIVE, then return a handle.

        Idempotent on ACTIVE — if already active, returns a handle.
        Generation does NOT increment (I7).

        Raises:
            SessionUnboundError: UNBOUND
            SessionCreatingError: CREATING
            SessionFinalizedError: DESTROYING or DESTROYED
        """
        async with self._get_lock(session_id):
            async with self._uow_factory() as uow:
                session = await uow.session.get_by_id(session_id)
            if session is None:
                raise ValueError(f"Session {session_id} not found")

            binding = session.sandbox_binding

            if binding.state == UNBOUND:
                raise SessionUnboundError(session_id)
            if binding.state == CREATING:
                raise SessionCreatingError(session_id)
            if binding.state in (DESTROYING, DESTROYED):
                raise SessionFinalizedError(session_id, destroyed_at=binding.destroyed_at)

            if binding.state == SUSPENDED:
                # Real resume: SUSPENDED → ACTIVE
                await self._transition(session_id, target=ACTIVE, generation_delta=0)
                # Re-read binding to avoid stale snapshot (§8.6 Codex Round 6 fix)
                async with self._uow_factory() as uow:
                    session = await uow.session.get_by_id(session_id)
                binding = session.sandbox_binding
                if binding.state != ACTIVE:
                    raise RuntimeError(
                        f"Expected ACTIVE after transition for session {session_id}, "
                        f"got {binding.state!r}"
                    )

            # binding.state == ACTIVE (either already was, or just transitioned)
            sandbox = self._registry.get_sandbox(session_id)
            if sandbox is not None:
                if self._registry.get_generation(session_id) != binding.generation:
                    self._registry.update_generation(session_id, binding.generation)
                return cast(SandboxHandle, self._registry.acquire_handle(session_id))
            return await self._rehydrate_or_mark_orphan(session_id, binding)

    async def try_register_from_binding(self, session_id: str) -> bool:
        """Reaper-scoped cross-process registry rehydrate (C2 cancel Part B).

        Populate the in-memory registry for ``session_id`` from its persisted
        binding ONLY when ``Sandbox.get(binding.id)`` actually finds the
        container. Returns True iff the registry ends up populated (already
        present in-process, OR freshly registered from a found container);
        False when there is nothing live to register (gone / unreachable /
        UNBOUND / DESTROYED / missing row / null ``binding.id``).

        Unlike the shared ``destroy()`` ACTIVE/SUSPENDED branch this NEVER
        advances ``sandbox_state`` and NEVER raises on a ``None`` ``Sandbox.get``
        — a reaper-scoped helper so the general cross-process ``destroy()``
        registry-miss behavior (and the ``delete_session`` path that depends on
        it) is left byte-for-byte unchanged (R5 P2 / NG9). The reaper calls this
        before ``destroy()`` so a fresh-process destroy of a still-live child
        container actually ``docker rm``s it instead of marking the row
        DESTROYED while the container leaks (R4 P1).
        """
        async with self._get_lock(session_id):
            if self._registry.get_sandbox(session_id) is not None:
                # Already populated in-process — destroy() will use it.
                return True
            async with self._uow_factory() as uow:
                session = await uow.session.get_by_id(session_id)
            if session is None:
                # No destroy() handoff on a False return — pop the lock we just
                # created so a startup sweep of many gone children can't leak
                # per-session locks (mirrors destroy()'s pop-to-avoid-leak).
                self._pop_lock_for(session_id)
                return False
            binding = session.sandbox_binding
            if binding.state not in (ACTIVE, SUSPENDED) or binding.id is None:
                self._pop_lock_for(session_id)
                return False
            try:
                sandbox = await self._sandbox_cls.get(binding.id)
            except Exception:
                # Daemon unreachable / lookup error -> not registerable; never
                # raise (reaper-scoped). Row stays ACTIVE for the next boot.
                logger.warning(
                    "try_register_from_binding: Sandbox.get raised for session "
                    "%s binding.id=%s — treating as not-registerable",
                    session_id,
                    binding.id,
                    exc_info=True,
                )
                self._pop_lock_for(session_id)
                return False
            if sandbox is None:
                # Container gone OR daemon unreachable (both collapse to None in
                # DockerSandbox.get). Leave the row ACTIVE; never silently
                # DESTROYED. A harmless phantom re-scanned next boot.
                self._pop_lock_for(session_id)
                return False
            self._registry.register(
                session_id, sandbox, generation=binding.generation
            )
            logger.info(
                "try_register_from_binding: registry rehydrated for session %s "
                "from binding.id=%s",
                session_id,
                binding.id,
            )
            return True

    async def destroy(self, session_id: str, reason: DestroyReason) -> None:
        """ACTIVE|SUSPENDED → DESTROYING → DESTROYED.

        Two-phase destroy + quiesce barrier (I6). See spec §8.5.

        C3 PR-1 (spec §3.2 M2 + §7.3 + plan Step 3.4) — raises typed signals
        instead of silent return on idempotent paths, and propagates infra
        failures instead of swallowing them:

        - binding.state == DESTROYED → :class:`SandboxAlreadyDestroyed`
        - binding.state == UNBOUND → :class:`SandboxBindingMissing`
        - CREATING state → :class:`SandboxBindingMissing`
          (a CREATING binding ALWAYS has ``binding.id is None`` — see that
          branch below — so this funnels through it; SPM Task 4)
        - missing session row → :class:`SandboxBindingMissing`
        - infra teardown failure → :class:`SandboxLifecycleError`
          (still records DESTROYING state so reconcile can pick up; the
          DESTROYED transition does NOT happen on infra failure)
        - happy path (ACTIVE/SUSPENDED/DESTROYING resume) → None

        SPM Task 4 (intent-before-lock): the FIRST thing this method does — BEFORE
        taking the per-session lock — is invalidate any in-flight provision so a
        concurrent ``bind_new`` observes the intent via its CAS checks and rolls
        back. A delete-class ``reason`` ALSO registers a deletion tombstone at that
        point (``invalidate("delete")`` always tombstones), so even when the
        binding is UNBOUND / missing and this raises ``SandboxBindingMissing``, a
        late / second ``bind_new`` is still refused.

        Callers must catch the two terminal-success subclasses if they need
        idempotent semantics (mailbox handlers, session delete path,
        reconcile pass).
        """
        # SPM Task 4 (intent-before-lock): signal the in-flight provision (if any)
        # to abort, and — for a delete-class reason — register the deletion
        # tombstone, BEFORE contending for the lock. This must happen even on the
        # UNBOUND / missing-row / already-destroyed early-return paths below, which
        # is exactly why it sits above ``async with self._get_lock`` rather than
        # inside it (INV-SPM-2 ``always``-mode hardening; happy path unchanged).
        self._flights.invalidate(session_id, _flight_outcome_for(reason))
        async with self._get_lock(session_id):
            async with self._uow_factory() as uow:
                session = await uow.session.get_by_id(session_id)
            if session is None:
                # C3 PR-1 (spec §3.2 M2): missing binding row → typed signal.
                # C3 PR-1 (codex round 6 P2): pop lock to avoid leak — caller
                # treats SandboxBindingMissing as terminal-success and will
                # not retry.
                self._pop_lock_for(session_id)
                raise SandboxBindingMissing(session_id)

            binding = session.sandbox_binding

            if binding.state == DESTROYED:
                # C3 PR-1 (spec §3.2 M2 + §7.3): typed signal instead of silent return.
                # C3 PR-1 (codex round 6 P2): pop lock to avoid leak.
                self._pop_lock_for(session_id)
                raise SandboxAlreadyDestroyed(session_id)
            if binding.state == UNBOUND:
                # C3 PR-1 (plan Step 3.4): UNBOUND has no sandbox to destroy →
                # terminal-success signal identical to a missing row.
                # C3 PR-1 (codex round 6 P2): pop lock to avoid leak.
                self._pop_lock_for(session_id)
                raise SandboxBindingMissing(session_id)
            # C3 PR-1 (codex round 15 P2): defensive check for data
            # inconsistency. If the binding state is non-terminal (ACTIVE /
            # SUSPENDED / DESTROYING / CREATING) but ``binding.id`` is missing,
            # there is no sandbox to destroy — treat as terminal-success
            # identical to UNBOUND. Without this, the destroy flow would fall
            # through to ``cancel_and_drain`` + ``destroy_infra`` (both silent
            # no-ops when the registry has no entry) and then advance the
            # binding to DESTROYED — polluting forensic audit with a
            # phantom-destroy row for a row that never had a container.
            #
            # SPM Task 4: CREATING is handled HERE. ``bind_new`` only stamps
            # ``sandbox_id`` at the ACTIVE commit, so a CREATING binding ALWAYS has
            # ``binding.id is None`` and lands in this branch — the old CREATING
            # ``elif`` branch (a state-guarded ``elif`` further below) was
            # unreachable dead code and has been removed. When a provision is
            # actually in flight, this destroy has
            # already invalidated it BEFORE the lock (intent-before-lock), so the
            # flight's CAS check rolls the binding back to UNBOUND while it holds
            # the lock; this destroy then acquires the lock and sees UNBOUND (the
            # branch above), not CREATING. A CREATING row with no live flight
            # (e.g. a crashed / abandoned provision) falls here and is likewise a
            # ``SandboxBindingMissing`` terminal-success (session_service already
            # tolerates it).
            if binding.id is None:
                self._pop_lock_for(session_id)
                logger.warning(
                    "destroy: session %s has binding.state=%s but binding.id "
                    "is None — data inconsistency; treating as "
                    "SandboxBindingMissing",
                    session_id,
                    binding.state.value,
                )
                raise SandboxBindingMissing(session_id)
            if binding.state == DESTROYING:
                # SPM Task 6: if the registry entry was lost (e.g. process
                # restart between a first destroy() that left binding=DESTROYING
                # + container alive and this retry), the downstream
                # ``cancel_and_drain`` / ``destroy_infra`` would silently no-op
                # (registry sees no entry) and the binding would advance to
                # DESTROYED while the container leaks. Rehydrate from
                # ``binding.id`` before continuing so destroy_infra actually
                # targets the live container.
                #
                # ``get_strict`` (Task 5) resolves the round-14 ambiguity the
                # plain ``get`` had: it distinguishes a definite NotFound
                # (returns ``None`` → container already gone → terminal success,
                # short-circuit to DESTROYED here) from a Docker-daemon blip
                # (raises ``SandboxDaemonUnreachable`` → retryable; preserve
                # DESTROYING for the next reconcile / operator retry). A legacy
                # fake / external Sandbox class WITHOUT ``get_strict`` falls back
                # to the pre-Task-6 conservative behavior (ambiguous ``None`` →
                # raise, keep DESTROYING).
                if self._registry.get_sandbox(session_id) is None and binding.id:
                    probe = getattr(self._sandbox_cls, "get_strict", None)
                    if probe is not None:
                        try:
                            rehydrated = await probe(binding.id)
                        except SandboxDaemonUnreachable as e:
                            logger.warning(
                                "destroy: DESTROYING retry for session %s — "
                                "Docker daemon unreachable probing binding.id="
                                "%s; preserving DESTROYING for next reconcile "
                                "pass",
                                session_id,
                                binding.id,
                            )
                            raise SandboxLifecycleError(
                                f"destroy: DESTROYING retry for session "
                                f"{session_id} could not rehydrate registry — "
                                f"Docker daemon unreachable ({e!r}). Preserving "
                                f"DESTROYING for next reconcile pass."
                            ) from e

                        if rehydrated is None:
                            # Definite NotFound → container already gone →
                            # terminal success. Short-circuit to DESTROYED
                            # (preserving the originally-persisted destroy_reason),
                            # release the lock, and return — mirroring the happy
                            # path's teardown tail. DESTROYING→DESTROYED does NOT
                            # bump generation (I7).
                            # FIX-C (P1-2): get_strict None also covers an
                            # exited/paused container still on disk — physically
                            # remove it before finalizing (best-effort; NotFound
                            # is swallowed, other failures still finalize).
                            await self._best_effort_remove_container(binding.id)
                            now = datetime.now(UTC)
                            await self._transition(
                                session_id,
                                target=DESTROYED,
                                generation_delta=0,
                                destroyed_at=now,
                                destroy_reason=binding.destroy_reason or reason,
                            )
                            self._pop_lock_for(session_id)
                            logger.info(
                                "destroy: DESTROYING retry for session %s — "
                                "container already gone (get_strict NotFound); "
                                "finalized DESTROYED",
                                session_id,
                            )
                            return

                        self._registry.register(
                            session_id,
                            rehydrated,
                            generation=binding.generation,
                        )
                        logger.info(
                            "destroy: DESTROYING retry for session %s — registry "
                            "rehydrated from binding.id",
                            session_id,
                        )
                    else:
                        # Legacy fake / external class without get_strict:
                        # preserve the pre-Task-6 ambiguous-None behavior (both
                        # NotFound and daemon-error collapse to None → raise a
                        # retryable error, keep DESTROYING).
                        try:
                            rehydrated = await self._sandbox_cls.get(binding.id)
                        except Exception as e:
                            logger.exception(
                                "destroy: DESTROYING retry for session %s — "
                                "sandbox lookup failed for binding.id=%s",
                                session_id,
                                binding.id,
                            )
                            raise SandboxLifecycleError(
                                f"destroy: DESTROYING retry for session "
                                f"{session_id} could not rehydrate registry — "
                                f"Sandbox.get raised ({e!r}). Preserving "
                                f"DESTROYING for next reconcile pass."
                            ) from e

                        if rehydrated is None:
                            raise SandboxLifecycleError(
                                f"destroy: DESTROYING retry for session "
                                f"{session_id} could not rehydrate registry "
                                f"(Sandbox.get returned None — could be NotFound "
                                f"OR Docker daemon unreachable). Preserving "
                                f"DESTROYING for next reconcile pass."
                            )

                        self._registry.register(
                            session_id,
                            rehydrated,
                            generation=binding.generation,
                        )
                        logger.info(
                            "destroy: DESTROYING retry for session %s — registry "
                            "rehydrated from binding.id",
                            session_id,
                        )
                # Another destroy in progress — continue the flow
            elif binding.state in (ACTIVE, SUSPENDED):
                # Step 1: persist DESTROYING + generation++ (I7 rule b)
                # C3 PR-1 (codex round 4 P2): persist destroy_reason now so a
                # subsequent reconcile-driven DESTROYED transition preserves the
                # original reason (instead of being audited as RECONCILE_ORPHAN
                # when destroy_infra raises and reconcile picks up later).
                await self._transition(
                    session_id,
                    target=DESTROYING,
                    generation_delta=1,
                    destroy_reason=reason,
                )
                # C3 PR-1 (codex round 10 P2): sync registry generation so any
                # in-flight ``SandboxHandle`` referencing the OLD generation
                # fails ``_check_generation()`` after this point. Without this,
                # handles can still dispatch commands at a DESTROYING sandbox
                # if ``destroy_infra`` raises and leaves the registry entry
                # alive for retry. ``_transition`` is the canonical generation
                # source (it bumped the DB row), so derive the new value as
                # ``binding.generation + 1`` to avoid an extra DB roundtrip.
                new_generation = binding.generation + 1
                self._registry.update_generation(session_id, new_generation)

            # Step 2-4: quiesce + infra destroy
            try:
                await self._registry.cancel_and_drain(
                    session_id, timeout=self._quiesce_timeout
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Sandbox quiesce drain timed out for session %s — forcing destroy",
                    session_id,
                )

            # C3 PR-1 (spec §7.3): propagate infra failure as retryable
            # SandboxLifecycleError. Binding stays DESTROYING; reconcile or
            # an explicit retry can pick up on the next pass.
            try:
                await self._registry.destroy_infra(session_id)
            except SandboxLifecycleError:
                logger.warning(
                    "destroy_infra raised typed SandboxLifecycleError for "
                    "session %s — propagating",
                    session_id,
                )
                raise
            except Exception as e:
                logger.warning(
                    "destroy_infra failed for session %s; binding stays "
                    "DESTROYING until next reconcile/retry: %s",
                    session_id, e,
                )
                raise SandboxLifecycleError(
                    f"destroy_infra failed for {session_id}: {e}"
                ) from e

            self._registry.remove(session_id)

            # Step 5: persist DESTROYED + destroyed_at
            # C3 PR-1 (codex round 6 P2): preserve the originally persisted
            # destroy_reason on the DESTROYING-resume path. If
            # ``binding.destroy_reason`` is already set from a prior cycle that
            # failed at destroy_infra, that value is canonical for forensic
            # classification; only fall back to the current ``reason`` if no
            # value was previously persisted.
            await self._transition(
                session_id,
                target=DESTROYED,
                generation_delta=0,  # I7: DESTROYING→DESTROYED doesn't increment
                destroyed_at=datetime.now(UTC),
                destroy_reason=binding.destroy_reason or reason,
            )

            # Cleanup lock (eng review decision #3)
            self._pop_lock_for(session_id)

    async def reconcile_orphans(self, *, docker_dependent_enabled: bool = True) -> None:
        """App startup reconciliation (I11).

        1. Scan DESTROYING: container dead → DESTROYED; alive → continue destroy
        2. PR1 default: lazy rehydrate (first acquire triggers). Proactive
           ACTIVE/SUSPENDED scan deferred to PR2.

        SPM PR-3 Task 28 (spec §5.2c-5 / INV-SPM-3): ``docker_dependent_enabled``
        gates the Docker-dependent stages. Under ``off`` startup (main.py passes
        ``False``) the **DESTROYING probe/drain/destroy loop** and the
        **label-sweep** are skipped (no Docker socket) with one log line —
        DESTROYING bindings are left intact for the pre-switch runbook / TTL to
        drain. The **non-Docker** stages still run regardless: the
        CREATING→UNBOUND DB repair AND the mailbox supervisor / PEL recovery
        (R24-CLASS3 — PEL drain is Redis, not Docker).
        """
        logger.info(
            "Starting sandbox lifecycle reconcile_orphans scan "
            "(docker_dependent_enabled=%s)",
            docker_dependent_enabled,
        )

        try:
            async with self._uow_factory() as uow:
                # Find all sessions in DESTROYING state
                all_sessions = await uow.session.get_all()
        except Exception:
            logger.exception(
                "reconcile_orphans: failed to query sessions, skipping"
            )
            return

        destroying_sessions = [
            s for s in all_sessions
            if s.sandbox_binding.state == DESTROYING
        ]

        # SPM PR-3 Task 28: the DESTROYING probe/drain/destroy loop and the
        # label-sweep below both touch the Docker socket. Under off startup they
        # are skipped with one log line (bindings kept intact for the runbook /
        # TTL). The CREATING→UNBOUND repair + mailbox/PEL recovery further down
        # are non-Docker and run regardless.
        if not docker_dependent_enabled:
            logger.info(
                "reconcile_orphans: docker_dependent_enabled=False (off mode) — "
                "skipping %d DESTROYING probe/drain/destroy + label-sweep "
                "(Docker-dependent); CREATING→UNBOUND repair + mailbox/PEL "
                "recovery still run",
                len(destroying_sessions),
            )
        for session in (destroying_sessions if docker_dependent_enabled else []):
            binding = session.sandbox_binding
            session_id = session.id

            try:
                if binding.id:
                    # SPM Task 6: probe via ``get_strict`` (Task 5) so a Docker
                    # daemon blip (raises ``SandboxDaemonUnreachable``) is NOT
                    # mis-read as "container gone" and mis-finalized DESTROYED —
                    # keep DESTROYING for the next pass instead. ``None`` is now a
                    # DEFINITE NotFound → finalize DESTROYED below. A legacy fake
                    # / external class without ``get_strict`` falls back to the
                    # plain ``get`` (pre-Task-6 behavior).
                    probe = getattr(self._sandbox_cls, "get_strict", None)
                    if probe is not None:
                        try:
                            sandbox = await probe(binding.id)
                        except SandboxDaemonUnreachable:
                            logger.warning(
                                "reconcile_orphans: session %s DESTROYING probe "
                                "— Docker daemon unreachable for binding.id=%s; "
                                "keeping DESTROYING for next reconcile pass",
                                session_id,
                                binding.id,
                            )
                            continue
                    else:
                        sandbox = await self._sandbox_cls.get(binding.id)
                else:
                    sandbox = None

                if sandbox is None:
                    # Container dead — finalize to DESTROYED.
                    # FIX-C (P1-2): a strict-probe None also covers an
                    # exited/paused container still on disk — physically remove it
                    # before finalizing so it isn't left for the next startup
                    # label sweep (best-effort; no-op when binding.id is None).
                    await self._best_effort_remove_container(binding.id)
                    await self._transition(
                        session_id,
                        target=DESTROYED,
                        generation_delta=0,
                        destroyed_at=datetime.now(UTC),
                        destroy_reason=binding.destroy_reason or DestroyReason.RECONCILE_ORPHAN,
                    )
                    logger.info(
                        "reconcile_orphans: session %s DESTROYING → DESTROYED "
                        "(container gone)",
                        session_id,
                    )
                else:
                    # Container still alive — continue destroy flow
                    logger.info(
                        "reconcile_orphans: session %s DESTROYING with live "
                        "container, continuing destroy",
                        session_id,
                    )
                    self._registry.register(
                        session_id, sandbox, generation=binding.generation
                    )
                    try:
                        await self._registry.cancel_and_drain(
                            session_id, timeout=self._quiesce_timeout
                        )
                    except asyncio.TimeoutError:
                        logger.error(
                            "reconcile_orphans: drain timed out for session %s",
                            session_id,
                        )
                    infra_failed = False
                    try:
                        await self._registry.destroy_infra(session_id)
                    except Exception as e:
                        logger.exception(
                            "reconcile_orphans: docker rm failed for session %s; "
                            "leaving DESTROYING for next reconcile pass: %s",
                            session_id, e,
                        )
                        infra_failed = True

                    if infra_failed:
                        # C3 PR-1 (codex round 3 P2 + round 4 P2): preserve
                        # DESTROYING AND keep the registry entry so the next
                        # reconcile cycle (or an explicit destroy()) can actually
                        # retry destroy_infra against the live container.
                        # Clearing the registry here would make subsequent
                        # destroy_infra a silent no-op and leak the container.
                        continue

                    self._registry.remove(session_id)

                    await self._transition(
                        session_id,
                        target=DESTROYED,
                        generation_delta=0,
                        destroyed_at=datetime.now(UTC),
                        destroy_reason=binding.destroy_reason or DestroyReason.RECONCILE_ORPHAN,
                    )
                    logger.info(
                        "reconcile_orphans: session %s destroy completed",
                        session_id,
                    )
            except Exception:
                logger.exception(
                    "reconcile_orphans: failed to reconcile session %s, "
                    "skipping (Docker daemon may be unreachable)",
                    session_id,
                )

        # Phase 2: Recover sessions stuck in CREATING (process crash during bind_new)
        creating_sessions = [
            s for s in all_sessions
            if s.sandbox_binding.state == CREATING
        ]
        for session in creating_sessions:
            try:
                await self._transition(session.id, target=UNBOUND)
                logger.warning(
                    "reconcile_orphans: session %s CREATING → UNBOUND "
                    "(recovered from interrupted bind_new)",
                    session.id,
                )
            except Exception:
                logger.exception(
                    "reconcile_orphans: failed to recover CREATING session %s",
                    session.id,
                )

        logger.info(
            "reconcile_orphans complete: processed %d DESTROYING + %d CREATING sessions",
            len(destroying_sessions),
            len(creating_sessions),
        )

        # SPM Task 6 (spec §5.2c-5): label-sweep orphan-container cleanup runs
        # AFTER the CREATING→UNBOUND repair (binding repair first) and BEFORE the
        # mailbox PEL recovery below (which it must NOT perturb). It is fully
        # fail-safe internally, so it can never abort the reconcile pass.
        # SPM PR-3 Task 28: it enumerates + `docker rm`s containers by label, so
        # it is Docker-dependent → skipped under off startup (the DESTROYING loop
        # above already skipped; one consolidated log line covers both).
        if docker_dependent_enabled:
            await self._sweep_orphan_containers()

        # C3 PR-3c (plan §11.3) — mailbox supervisor recovery after pod restart.
        # When a pod dies, every per-pod ``MailboxSupervisor`` task dies with it.
        # The mailbox stream's PEL still holds undelivered envelopes; we need a
        # fresh supervisor on each affected root so the startup XAUTOCLAIM
        # (PR-3b spec §5.6) can drain the orphans.
        #
        # Gated on ``supervisor_registry`` injection so legacy / pre-mailbox
        # deployments and existing tests that didn't pass a registry stay
        # untouched.
        #
        # Failure isolation rules:
        # * DB query failure → log + bail out. There is NO in-process retry:
        #   ``reconcile_orphans`` is called once at FastAPI lifespan startup
        #   (``app/main.py`` step 9) and not again until the next pod boot.
        #   The log message at the bail-out site spells this out so ops know
        #   to restart instead of waiting for a non-existent retry cycle.
        # * Per-root ``spawn`` failure → log + continue with the rest of the
        #   list. ``health_check()`` ran ONCE before the loop so a transient
        #   spawn failure is genuinely scoped to that one root.
        # * ``health.get(root_id) in ("alive", "restarting", "crashed")``
        #   short-circuits so (a) successful per-pod restart-loop ticks
        #   that already brought the slot back don't get re-spawned and
        #   (b) crashed slots are left to the restart loop's recovery
        #   path (it owns crash recovery; ``spawn`` would no-op on the
        #   already-registered slot anyway — see codex r1 [HIGH ARCH]).
        if self._supervisor_registry is not None:
            try:
                async with self._uow_factory() as uow:
                    running_root_ids = (
                        await uow.session.find_running_mailbox_plane_root_ids()
                    )
            except Exception:
                # codex r2 [HIGH CONTRACT] — be honest about the retry
                # cadence. ``reconcile_orphans`` is wired into FastAPI
                # lifespan startup (``app/main.py`` step 9) and is NOT
                # called again until the next pod boot — there is no
                # in-process retry cycle. If the mailbox-plane query
                # fails here, this pod runs without supervisor recovery
                # for the affected roots until either: (a) the operator
                # manually triggers another reconcile via admin tooling
                # (none ships in PR-3c — TODO PR-4+), or (b) the pod
                # restarts. Surfacing this in the log so ops know to
                # restart instead of waiting for an auto-recovery that
                # never comes.
                logger.exception(
                    "reconcile_orphans: failed to query mailbox plane roots; "
                    "this pod's mailbox supervisor recovery is now disabled "
                    "until the next pod restart re-runs reconcile_orphans"
                )
                return

            try:
                health = await self._supervisor_registry.health_check()
            except Exception:
                logger.exception(
                    "reconcile_orphans: supervisor_registry.health_check failed; "
                    "treating all slots as missing and re-spawning"
                )
                health = {}

            # codex r1 [HIGH ARCH] — skip "crashed" too: the per-pod
            # ``_restart_loop`` (SupervisorRegistry §spec 6.2) owns crashed-slot
            # recovery and will resurrect it within ``restart_interval_s``.
            # ``spawn()`` is idempotent on existing slots (returns no-op when
            # ``root in self._slots``), so calling it on a crashed slot was a
            # no-op already — making the skip explicit avoids misleading
            # "ensured mailbox supervisor" log lines for roots whose recovery
            # is actually the restart loop's job.
            #
            # codex r1 [MEDIUM PERF] — parallelize per-root spawn. Serial await
            # at 5s per-slot ``ready_timeout_s`` could blow startup latency at
            # O(N) for N RUNNING mailbox-plane roots; ``asyncio.gather`` with
            # ``return_exceptions=True`` keeps per-root failure isolation while
            # collapsing wall-clock to ~5s regardless of N.
            spawn_tasks = []
            spawn_root_ids: list[str] = []
            for root_id in running_root_ids:
                state = health.get(root_id)
                if state in ("alive", "restarting", "crashed"):
                    continue
                spawn_tasks.append(self._supervisor_registry.spawn(root_id))
                spawn_root_ids.append(root_id)
            if spawn_tasks:
                results = await asyncio.gather(*spawn_tasks, return_exceptions=True)
                for root_id, result in zip(spawn_root_ids, results):
                    if isinstance(result, Exception):
                        logger.exception(
                            "reconcile_orphans: failed to spawn supervisor "
                            "for root %s — continuing with remaining roots",
                            root_id,
                            exc_info=result,
                        )
                    else:
                        logger.info(
                            "reconcile_orphans: ensured mailbox supervisor for root %s",
                            root_id,
                        )

    async def account_lingering_after_off(self, metrics: Any) -> int:
        """SPM PR-3 Task 28 (spec §5.9, r9/codex R8-G3): off-startup residual
        sandbox accounting.

        Counts DB bindings that STILL hold a live-ish sandbox after switching to
        off — the predicate is ``sandbox_id IS NOT NULL AND state ∈ {ACTIVE,
        SUSPENDED, DESTROYING}`` (NOT the loose "non-terminal": the default
        ``UNBOUND`` / ``id=None`` binding every session carries would all
        false-count). Writes the count to the provision-metrics singleton via the
        REAL ``record_lingering_after_off`` (the G3 anti-fake guard — a snapshot
        assertion proves the production metric is wired, not a stub field).

        Returns ``N`` for the caller's log line. Best-effort: a query failure
        logs + returns 0 (metrics never mask a boot; nothing is recorded).
        """
        try:
            async with self._uow_factory() as uow:
                all_sessions = await uow.session.get_all()
        except Exception:
            logger.exception(
                "account_lingering_after_off: session query failed; "
                "skipping off-startup lingering accounting"
            )
            return 0

        lingering = sum(
            1
            for s in all_sessions
            if s.sandbox_binding.id is not None
            and s.sandbox_binding.state in (ACTIVE, SUSPENDED, DESTROYING)
        )
        metrics.record_lingering_after_off(count=lingering)
        return lingering

    # ── Internal helpers ──

    async def _sweep_orphan_containers(self) -> None:
        """SPM Task 6 — label-sweep orphan-container cleanup (fail-safe).

        Enumerate platform-managed containers (those carrying the
        ``actus.session_id`` label; Task 5 ``list_managed_containers``) and
        remove any that (a) have NO live binding claiming them AND (b) are past
        a 120s grace window (Task 5 ``remove_container``).

        **Keep** a container when its ``session_id``'s binding exists AND
        ``binding.id == container.name`` AND ``binding.state`` is one of
        ACTIVE / SUSPENDED / DESTROYING (an in-use or actively-tearing-down
        sandbox). Everything else that is past the grace window is an orphan.

        **Fail-safe (宁漏勿误删):** ANY enumeration or DB-query error skips the
        entire round (log + return) — we would rather miss an orphan this pass
        than mis-delete a live container on partial data. Per-container removal
        errors are likewise isolated so one bad container can't abort cleanup of
        the rest, and nothing here can escape to abort the reconcile pass.

        Guarded on ``list_managed_containers`` presence so external-sandbox
        deployments / legacy fakes without the classmethod are a no-op.
        (PR-3 / Task 28 will add the off-mode gate; here it runs
        unconditionally.)
        """
        lister = getattr(self._sandbox_cls, "list_managed_containers", None)
        if lister is None:
            # External sandbox / legacy fake without the label-sweep primitive.
            return

        # ``list_managed_containers`` is a SYNCHRONOUS classmethod → off-loop it.
        # Fail-safe: enumeration OR the session query raising ANY exception skips
        # this round entirely (no container removed on partial data).
        try:
            rows = await asyncio.to_thread(lister)
            async with self._uow_factory() as uow:
                all_sessions = await uow.session.get_all()
            # Derive the keep-set snapshot INSIDE the fail-safe try so a raise
            # from the comprehension (or the small locals) cannot escape to the
            # unguarded call site and abort the reconcile pass before the
            # mailbox-PEL section — nothing here can escape (docstring invariant).
            bindings_by_session = {s.id: s.sandbox_binding for s in all_sessions}
            keep_states = (ACTIVE, SUSPENDED, DESTROYING)
            now = datetime.now(UTC)
            grace = timedelta(seconds=120)
        except Exception:
            logger.exception(
                "reconcile_orphans: label-sweep enumeration/query failed; "
                "skipping this round (fail-safe — no containers removed)"
            )
            return

        for row in rows:
            # Per-row fail-safe: a malformed row (bad ``created_at`` type,
            # non-dict, etc.) OR a ``remove_container`` error is isolated to this
            # container so it can neither mis-delete a live one nor abort the
            # sweep (and thus the reconcile pass) for the remaining containers.
            try:
                name = row.get("name")
                if not name:
                    continue
                binding = bindings_by_session.get(row.get("session_id"))
                keep = (
                    binding is not None
                    and binding.id == name
                    and binding.state in keep_states
                )
                if keep:
                    continue
                created_at = row.get("created_at")
                # Missing / unknown creation time → treat as brand-new (skip) so
                # an un-datable container is never mis-deleted (宁漏勿误删).
                if created_at is None or (now - created_at) <= grace:
                    continue
                await asyncio.to_thread(self._sandbox_cls.remove_container, name)
                logger.info(
                    "reconcile_orphans: label-sweep removed orphan container %s "
                    "(session_id=%s — no live binding, past 120s grace)",
                    name,
                    row.get("session_id"),
                )
            except Exception:
                logger.exception(
                    "reconcile_orphans: label-sweep failed processing container "
                    "row %r; continuing with remaining containers",
                    row.get("name") if isinstance(row, dict) else row,
                )

    async def _rehydrate_or_mark_orphan(
        self, session_id: str, binding: SandboxBinding
    ) -> SandboxHandle:
        """Called with per_session_lock held. Tries to rehydrate from Docker,
        or marks as orphan DESTROYED if container is gone (I10)."""
        rehydrated: Optional[Sandbox] = None
        if binding.id:
            try:
                rehydrated = await self._sandbox_cls.get(binding.id)
            except Exception:
                logger.exception(
                    "Rehydrate docker lookup failed for session %s sandbox %s",
                    session_id,
                    binding.id,
                )

        if rehydrated is not None:
            # Container alive — register and return handle
            self._registry.register(
                session_id, rehydrated, generation=binding.generation
            )
            return cast(SandboxHandle, self._registry.acquire_handle(session_id))

        # Container dead — ACTIVE/SUSPENDED → DESTROYED (I10)
        now = datetime.now(UTC)
        await self._transition(
            session_id,
            target=DESTROYED,
            generation_delta=1,  # Poison any stale handles
            destroyed_at=now,
            destroy_reason=DestroyReason.RECONCILE_ORPHAN,
        )
        # orphan guard（r22/R22-CLASS1, 最后一个 read-commit swallow 实例）: the
        # DESTROYED `_transition` above writes a UoW whose commit sub-window
        # swallows cancel (cancelling() stays > 0). Honor it BEFORE raising
        # SessionFinalizedError so a run-cancel here surfaces as CancelledError
        # (provisioner records `cancelled`) rather than being masked as a
        # SessionFinalizedError (provisioner would mis-record `failed`). The
        # DESTROYED commit is intentionally kept (not rolled back).
        self._raise_if_read_swallowed_cancel()
        raise SessionFinalizedError(
            session_id,
            destroyed_at=now,
            detail="sandbox container was terminated externally",
        )

    async def shutdown(self) -> None:
        """Graceful shutdown: release in-memory state only.

        Per spec §8.1: PR1 does NOT introduce LIFESPAN_SHUTDOWN.
        ACTIVE/SUSPENDED sandboxes are left alive and will expire via
        sandbox_ttl_minutes (container-internal supervisord timeout).
        On next startup, reconcile_orphans() handles any that died
        while the process was down.

        We do NOT call destroy() here — that would permanently terminate
        recoverable sessions (SUSPENDED → DESTROYED) on every deploy/restart,
        breaking the resume/reopen_takeover contract.

        FIX-G: the ONLY teardown work we do is a bounded drain of any in-flight
        late disposers (spawned by ``_spawn_late_disposer`` when a create shield
        was cancelled/failed while the shielded ``create`` kept running). Without
        it, loop teardown would cancel a disposer still awaiting its background
        create → a container could be born with nobody to remove it. We give them
        a bounded window and do NOT cancel them (cancelling defeats the purpose);
        any that outlast the window remain backstopped by container TTL + the
        next-startup label sweep.
        """
        if self._late_dispose_tasks:
            _done, pending = await asyncio.wait(
                set(self._late_dispose_tasks), timeout=10
            )
            if pending:
                logger.warning(
                    "SandboxLifecycleService shutdown: %d late-disposer task(s) "
                    "did not finish within the drain window; they remain "
                    "backstopped by container TTL + next-startup label sweep",
                    len(pending),
                )
        logger.info("SandboxLifecycleService shutting down (in-memory cleanup only)")
        self._registry = SandboxRegistry()  # drop all in-memory refs
        self._per_session_locks.clear()
        logger.info("SandboxLifecycleService shutdown complete")

    @property
    def registry(self) -> SandboxRegistry:
        """Expose registry for DI (e.g., WS endpoints register holders)."""
        return self._registry

    @staticmethod
    def check_single_worker_argv() -> None:
        """Best-effort check that uvicorn is not running with --workers > 1.

        Called from FastAPI lifespan startup. Each uvicorn fork retains the
        original sys.argv, so all workers fail-loud (intentional — refuse
        to start rather than silently race). See spec §8.6.
        """
        for i, arg in enumerate(sys.argv):
            if arg == "--workers" and i + 1 < len(sys.argv):
                try:
                    n = int(sys.argv[i + 1])
                    if n > 1:
                        raise RuntimeError(
                            f"Detected --workers {n} in sys.argv. "
                            f"SandboxLifecycleService requires single-worker mode. "
                            f"Multi-worker support requires §12 Q5 spec."
                        )
                except ValueError:
                    pass
