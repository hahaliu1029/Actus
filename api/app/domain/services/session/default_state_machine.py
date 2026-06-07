"""DefaultSessionStateMachine — repo-backed SSM with no-op event publisher.

event_publisher is wired here as a sink for A4-0 SessionModeChangedEvent.
PE-0 ships it as Optional and falls back to a no-op; A4-0 will plumb
the real SSE publisher.
"""

from __future__ import annotations
from datetime import datetime
from typing import Any, Callable, Mapping, Optional, Protocol

from app.domain.models.session import SessionStatus
from app.domain.repositories.session_repository import SessionRepository
from app.domain.repositories.uow import IUnitOfWork
from app.domain.services.permission.errors import SessionModeViolation
from app.domain.services.session.session_state_machine import SessionStateMachine


class SseEventPublisher(Protocol):
    async def publish(self, session_id: str, event: dict[str, Any]) -> None: ...


class _NoopPublisher:
    async def publish(self, session_id: str, event: dict[str, Any]) -> None:
        return None


class DefaultSessionStateMachine(SessionStateMachine):
    """Per-call UoW pattern — no long-lived AsyncSession captured.

    SSM is a domain service, so it consumes the SessionRepository
    Protocol exposed on IUnitOfWork (`uow.session: SessionRepository`).
    No infrastructure import in this file.
    """

    def __init__(
        self,
        *,
        uow_factory: Callable[[], IUnitOfWork],
        redis: Any = None,
        event_publisher: Optional[SseEventPublisher] = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._redis = redis  # reserved for hot-path cache, A4-0 wires
        self._publisher = event_publisher or _NoopPublisher()

    async def get_mode(self, session_id: str) -> SessionStatus:
        async with self._uow_factory() as uow:
            mode, _ = await uow.session.read_status_with_revision(session_id)
            return mode

    async def get_mode_with_revision(
        self, session_id: str,
    ) -> tuple[SessionStatus, int]:
        async with self._uow_factory() as uow:
            return await uow.session.read_status_with_revision(session_id)

    async def transition(
        self,
        session_id: str,
        from_state: SessionStatus,
        to_state: SessionStatus,
        reason: str,
        extra_values: Mapping[str, Any] | None = None,
    ) -> bool:
        # IUnitOfWork.__aexit__ commits on clean exit; the CAS UPDATE
        # is therefore atomic with the SSM call boundary. flush() inside
        # repo.transition_status guarantees rowcount is settled before
        # we read it.
        async with self._uow_factory() as uow:
            ok = await uow.session.transition_status(
                session_id=session_id,
                from_state=from_state,
                to_state=to_state,
                extra_values=extra_values,
            )
        if ok:
            await self._publisher.publish(
                session_id,
                {
                    "type": "session_mode_changed",
                    "session_id": session_id,
                    "to": to_state.value,
                    "reason": reason,
                },
            )
        return ok

    async def request_takeover(self, session_id: str, reason: str) -> None:
        ok = await self.transition(
            session_id, SessionStatus.RUNNING, SessionStatus.TAKEOVER_PENDING,
            reason=reason,
        )
        if not ok:
            current, _ = await self.get_mode_with_revision(session_id)
            raise SessionModeViolation(
                f"cannot request takeover from {current.value}"
            )

    async def release_takeover(self, session_id: str) -> None:
        ok = await self.transition(
            session_id, SessionStatus.TAKEOVER, SessionStatus.RUNNING,
            reason="release_takeover",
        )
        if not ok:
            current, _ = await self.get_mode_with_revision(session_id)
            raise SessionModeViolation(
                f"cannot release takeover from {current.value}"
            )

    async def enter_finishing(self, session_id: str) -> None:
        ok = await self.transition(
            session_id, SessionStatus.RUNNING, SessionStatus.FINISHING,
            reason="enter_finishing",
        )
        if not ok:
            current, _ = await self.get_mode_with_revision(session_id)
            raise SessionModeViolation(
                f"cannot enter finishing from {current.value}"
            )

    async def complete(self, session_id: str) -> None:
        # PE-0 round 31 P2 fix: write terminal metadata atomically with the
        # status CAS so downstream stats / supervisor recovery / phase
        # filtering do not see a COMPLETED row missing completed_at /
        # terminal_reason / execution_phase. update_to_terminal() owns the
        # same field set; we mirror it here for the SSM-driven path.
        #
        # PE-0 round 32 P2 fix: terminal_reason must use a value from the
        # Session.terminal_reason Literal allow-list (domain/models/session.py:
        # "natural" | "user_cancel" | "server_restart" | "resume_state_lost"
        # | "watchdog_timeout"). Using "complete" breaks SessionModel.to_domain
        # Pydantic validation when the row is later read back. SSM.complete()
        # represents the natural FINISHING -> COMPLETED transition, so
        # "natural" is the correct semantic value.
        #
        # PE-0 round 33 P1 fix: completed_at column is
        # `Mapped[Optional[datetime]] = mapped_column(DateTime, ...)` —
        # TIMESTAMP WITHOUT TIME ZONE (no `timezone=True`). asyncpg refuses
        # to bind a timezone-aware datetime to a naive column, so we mirror
        # update_to_terminal() and emit a naive UTC-wall-clock value.
        now = datetime.now()
        ok = await self.transition(
            session_id, SessionStatus.FINISHING, SessionStatus.COMPLETED,
            reason="complete",
            extra_values={
                "completed_at": now,
                "terminal_reason": "natural",
                "execution_phase": "terminated",
            },
        )
        if not ok:
            current, _ = await self.get_mode_with_revision(session_id)
            raise SessionModeViolation(
                f"cannot complete from {current.value}"
            )

    async def set_mode(
        self,
        session_id: str,
        to: SessionStatus,
        reason: str,
        *,
        session_repo: SessionRepository,
    ) -> None:
        # A4-1 caller-owned pure mutator: exactly one repo call, no UoW, no emit.
        # `reason` is intentionally unused in A4-1 (telemetry / A4-2 single-emitter).
        await session_repo.update_status(session_id, to)

    async def terminate(
        self,
        session_id: str,
        to: SessionStatus,
        terminal_reason: str,
        *,
        session_repo: SessionRepository,
    ) -> bool:
        # A4-1 caller-owned pure mutator: returns the repo idempotency bool.
        return await session_repo.update_to_terminal(
            session_id, to, terminal_reason
        )
