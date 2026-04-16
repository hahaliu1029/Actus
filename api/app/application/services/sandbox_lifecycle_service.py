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
    SandboxLifecycleError,
    SessionCreatingError,
    SessionDestroyingError,
    SessionFinalizedError,
    SessionSuspendedError,
    SessionUnboundError,
)
from app.domain.external.sandbox import Sandbox, SandboxHandle
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
    ) -> None:
        self._sandbox_cls = sandbox_cls
        self._uow_factory = uow_factory
        self._registry = SandboxRegistry()
        self._quiesce_timeout = quiesce_timeout_seconds
        self._per_session_locks: dict[str, asyncio.Lock] = {}

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

    async def bind_new(self, session_id: str) -> SandboxHandle:
        """UNBOUND → CREATING → ACTIVE. Creates a new sandbox container.

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

            # Step 2: Actually create the sandbox container
            try:
                sandbox = await self._sandbox_cls.create()
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
            return cast(SandboxHandle, self._registry.acquire_handle(session_id))

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

    async def destroy(self, session_id: str, reason: DestroyReason) -> None:
        """ACTIVE|SUSPENDED → DESTROYING → DESTROYED.

        Two-phase destroy + quiesce barrier (I6). See spec §8.5.
        """
        async with self._get_lock(session_id):
            async with self._uow_factory() as uow:
                session = await uow.session.get_by_id(session_id)
            if session is None:
                raise ValueError(f"Session {session_id} not found")

            binding = session.sandbox_binding

            if binding.state == DESTROYED:
                return  # Already destroyed, idempotent
            if binding.state == DESTROYING:
                # Another destroy in progress — continue the flow
                pass
            elif binding.state in (ACTIVE, SUSPENDED):
                # Step 1: persist DESTROYING + generation++ (I7 rule b)
                await self._transition(
                    session_id, target=DESTROYING, generation_delta=1
                )
            elif binding.state in (UNBOUND, CREATING):
                # Nothing to destroy
                await self._transition(
                    session_id,
                    target=DESTROYED,
                    generation_delta=0,
                    destroyed_at=datetime.now(UTC),
                    destroy_reason=reason,
                )
                self._per_session_locks.pop(session_id, None)
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

            try:
                await self._registry.destroy_infra(session_id)
            except Exception:
                logger.exception(
                    "Docker rm failed during destroy session=%s", session_id
                )

            self._registry.remove(session_id)

            # Step 5: persist DESTROYED + destroyed_at
            await self._transition(
                session_id,
                target=DESTROYED,
                generation_delta=0,  # I7: DESTROYING→DESTROYED doesn't increment
                destroyed_at=datetime.now(UTC),
                destroy_reason=reason,
            )

            # Cleanup lock (eng review decision #3)
            self._per_session_locks.pop(session_id, None)

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
                    try:
                        await self._registry.destroy_infra(session_id)
                    except Exception:
                        logger.exception(
                            "reconcile_orphans: docker rm failed for session %s",
                            session_id,
                        )
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
