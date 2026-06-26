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
from datetime import UTC, datetime
from typing import Callable, Optional, Type, cast

from app.domain.errors.sandbox_lifecycle import (
    SandboxAlreadyDestroyed,
    SandboxBindingMissing,
    SandboxLifecycleError,
    SessionCreatingError,
    SessionDestroyingError,
    SessionFinalizedError,
    SessionSuspendedError,
    SessionUnboundError,
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
    ) -> None:
        self._sandbox_cls = sandbox_cls
        self._uow_factory = uow_factory
        self._registry = SandboxRegistry()
        self._quiesce_timeout = quiesce_timeout_seconds
        self._per_session_locks: dict[str, asyncio.Lock] = {}
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
    ) -> SandboxBinding:
        """Persist a binding state transition via UoW.

        Also emits a ``SandboxStateChangedEvent`` to the session event stream
        so the frontend can react (PR2 §10.2).

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
                reason=reason_str,
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
        """
        async with self._get_lock(session_id):
            async with self._uow_factory() as uow:
                session = await uow.session.get_by_id(session_id)
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

            # Step 1: UNBOUND → CREATING
            await self._transition(session_id, target=CREATING)

            # Resolve user_id: explicit arg wins, otherwise fall back to session
            effective_user_id = user_id if user_id is not None else session.user_id

            # Step 2: Actually create the sandbox container
            try:
                sandbox = await self._sandbox_cls.create(user_id=effective_user_id)
                await sandbox.ensure_sandbox()
            except Exception:
                # Create failed — roll back to UNBOUND
                logger.exception(
                    "Sandbox creation failed for session %s; rolling back to UNBOUND",
                    session_id,
                )
                await self._transition(session_id, target=UNBOUND)
                raise

            # Step 3: CREATING → ACTIVE with generation++
            now = datetime.now(UTC)
            new_binding = await self._transition(
                session_id,
                target=ACTIVE,
                generation_delta=1,  # I7 rule (a)
                sandbox_id=sandbox.id,
                created_at=now,
            )

            # Step 4: Register in registry and return handle
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
                )

            return cast(SandboxHandle, self._registry.acquire_handle(session_id))

    async def _observe_container_policy(
        self, *, session_id: str, user_id: str | None, new_binding, session
    ) -> None:
        """C5a observe-only emission (additive, best-effort).

        Only ever called when the flag is ON (gated by the caller). Reads
        get_settings() LAZILY here (ON path only) for the view values — the OFF
        path never touches config (INV-0). Swallows ALL errors so an observe
        failure can never fail a sandbox bind. The sink is non-suspending
        (Task 3), so this awaits without yielding.
        """
        try:
            from app.domain.models.sandbox_policy import (
                ContainerCreateInput,
                build_settings_view,
            )
            from app.domain.services.safety.sandbox_policy_compiler import (
                SandboxPolicyCompiler,
            )
            from core.config import get_settings  # lazy — reached only on the ON path (INV-0)

            inp = ContainerCreateInput(
                session_id=session_id,
                user_id=user_id,
                sandbox_id=new_binding.id,  # SandboxBinding.id == sandbox.id
                sandbox_generation=new_binding.generation,
                worker_type=session.worker_type,
                depth=session.depth,
                settings=build_settings_view(get_settings()),
            )
            snapshot = SandboxPolicyCompiler().compile_container_create(inp)
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
        - missing session row → :class:`SandboxBindingMissing`
        - infra teardown failure → :class:`SandboxLifecycleError`
          (still records DESTROYING state so reconcile can pick up; the
          DESTROYED transition does NOT happen on infra failure)
        - happy path (ACTIVE/SUSPENDED/CREATING/DESTROYING resume) → None

        Callers must catch the two terminal-success subclasses if they need
        idempotent semantics (mailbox handlers, session delete path,
        reconcile pass).
        """
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
                # C3 PR-1 (codex round 13 P2 + round 14 P2): if the registry
                # entry was lost (e.g. process restart between a first destroy()
                # that left binding=DESTROYING + container alive and this retry),
                # the downstream ``cancel_and_drain`` / ``destroy_infra`` would
                # silently no-op (registry sees no entry) and the binding would
                # advance to DESTROYED while the container leaks. Rehydrate
                # from ``binding.id`` before continuing so destroy_infra
                # actually targets the live container.
                #
                # Round 14 P2: ``Sandbox.get()`` returning ``None`` is AMBIGUOUS
                # in production — ``DockerSandbox.get()`` collapses both
                # ``NotFound`` (container externally removed; terminal success)
                # AND ``APIError`` (Docker daemon unreachable; transient
                # failure) into ``None``. We cannot safely distinguish these
                # cases here, so the conservative posture is to preserve
                # DESTROYING by raising ``SandboxLifecycleError``: an operator
                # retry (once Docker recovers) or the next ``reconcile_orphans``
                # pass will resolve it correctly. Premature DESTROYED would
                # silently mark a live container as gone during a Docker
                # outage. If/when ``DockerSandbox.get()`` is refactored to
                # distinguish NotFound from APIError (planned for PR-3a
                # supervisor lifecycle integration), the NotFound branch can
                # cleanly short-circuit to terminal-success here.
                if self._registry.get_sandbox(session_id) is None and binding.id:
                    try:
                        rehydrated = await self._sandbox_cls.get(binding.id)
                    except Exception as e:
                        logger.exception(
                            "destroy: DESTROYING retry for session %s — sandbox "
                            "lookup failed for binding.id=%s",
                            session_id,
                            binding.id,
                        )
                        raise SandboxLifecycleError(
                            f"destroy: DESTROYING retry for session {session_id} "
                            f"could not rehydrate registry — Sandbox.get raised "
                            f"({e!r}). Preserving DESTROYING for next reconcile "
                            f"pass."
                        ) from e

                    if rehydrated is None:
                        # Ambiguous: NotFound (terminal success) and APIError
                        # (transient failure) both collapse to None in
                        # DockerSandbox.get(). Treat as retryable failure so
                        # live containers aren't silently marked DESTROYED
                        # during Docker outages. Reconcile / operator retry
                        # resolves once Docker is reachable again.
                        raise SandboxLifecycleError(
                            f"destroy: DESTROYING retry for session {session_id} "
                            f"could not rehydrate registry (Sandbox.get returned "
                            f"None — could be NotFound OR Docker daemon "
                            f"unreachable). Preserving DESTROYING for next "
                            f"reconcile pass."
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
            elif binding.state == CREATING:
                # Nothing to destroy
                await self._transition(
                    session_id,
                    target=DESTROYED,
                    generation_delta=0,
                    destroyed_at=datetime.now(UTC),
                    destroy_reason=reason,
                )
                self._pop_lock_for(session_id)
                return

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

    async def reconcile_orphans(self) -> None:
        """App startup reconciliation (I11).

        1. Scan DESTROYING: container dead → DESTROYED; alive → continue destroy
        2. PR1 default: lazy rehydrate (first acquire triggers). Proactive
           ACTIVE/SUSPENDED scan deferred to PR2.
        """
        logger.info("Starting sandbox lifecycle reconcile_orphans scan")

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

        for session in destroying_sessions:
            binding = session.sandbox_binding
            session_id = session.id

            try:
                if binding.id:
                    sandbox = await self._sandbox_cls.get(binding.id)
                else:
                    sandbox = None

                if sandbox is None:
                    # Container dead — finalize to DESTROYED
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

    # ── Internal helpers ──

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
        """
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
