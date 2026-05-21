"""C3 PR-1 (codex P1) — SessionService.delete_session lifecycle error handling.

Contract:
- ``SandboxAlreadyDestroyed`` / ``SandboxBindingMissing`` are terminal-success
  signals: delete proceeds to remove the DB row.
- Any other ``SandboxLifecycleError`` (docker daemon down, generation mismatch,
  unbound state, etc.) MUST abort the delete BEFORE ``delete_by_id`` runs so
  ``reconcile_orphans`` can find the still-bound sandbox on retry. The
  exception is propagated to the caller for surface-up.
- C3 PR-1 (codex round 13 P2): background-execution slot cleanup is idempotent
  and MUST run regardless of destroy() success/failure so a sandbox teardown
  failure does not strand the Redis quota slot until TTL.

Pre-fix bug: the broad ``except Exception`` swallowed real
``SandboxLifecycleError`` and proceeded to hard-delete the session row, which
left the sandbox unreclaimable.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.session_service import SessionService
from app.domain.errors.sandbox_lifecycle import (
    SandboxAlreadyDestroyed,
    SandboxBindingMissing,
    SandboxLifecycleError,
)
from app.domain.models.session import Session

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_uow_and_factory(session: Session):
    """UoW mock that returns the given session from get_by_id and tracks delete_by_id."""
    uow = MagicMock()
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)
    uow.session = MagicMock()
    uow.session.get_by_id = AsyncMock(return_value=session)
    uow.session.delete_by_id = AsyncMock()
    return uow, lambda: uow


def _make_session(session_id: str = "s-del", user_id: str = "u-1") -> Session:
    return Session(id=session_id, user_id=user_id, title="t")


class TestDeleteSessionLifecycleErrors:
    async def test_already_destroyed_is_terminal_success_proceeds_to_delete(self) -> None:
        """SandboxAlreadyDestroyed → delete_by_id still runs (no-op terminal-success)."""
        session = _make_session("s-1", "u-1")
        uow, factory = _make_uow_and_factory(session)
        lifecycle = MagicMock()
        lifecycle.destroy = AsyncMock(side_effect=SandboxAlreadyDestroyed("s-1"))

        service = SessionService(
            uow_factory=factory,
            sandbox_lifecycle_service=lifecycle,
        )

        await service.delete_session("s-1", user_id="u-1")

        lifecycle.destroy.assert_awaited_once()
        uow.session.delete_by_id.assert_awaited_once_with("s-1")

    async def test_binding_missing_is_terminal_success_proceeds_to_delete(self) -> None:
        """SandboxBindingMissing → delete_by_id still runs (binding never reached ACTIVE)."""
        session = _make_session("s-2", "u-1")
        uow, factory = _make_uow_and_factory(session)
        lifecycle = MagicMock()
        lifecycle.destroy = AsyncMock(side_effect=SandboxBindingMissing("s-2"))

        service = SessionService(
            uow_factory=factory,
            sandbox_lifecycle_service=lifecycle,
        )

        await service.delete_session("s-2", user_id="u-1")

        lifecycle.destroy.assert_awaited_once()
        uow.session.delete_by_id.assert_awaited_once_with("s-2")

    async def test_generic_lifecycle_error_aborts_delete_and_propagates(self) -> None:
        """Non-terminal SandboxLifecycleError → delete_by_id NOT called, exception propagates.

        This is the core P1 regression: pre-fix, the broad ``except Exception``
        swallowed this and hard-deleted the session row.

        Codex round 13 P2 follow-up: background slot cleanup is idempotent and
        MUST still run before the propagation so Redis quota isn't stranded
        until TTL on a sandbox teardown failure.
        """
        session = _make_session("s-3", "u-1")
        uow, factory = _make_uow_and_factory(session)
        lifecycle = MagicMock()
        lifecycle.destroy = AsyncMock(
            side_effect=SandboxLifecycleError("docker daemon unreachable")
        )

        service = SessionService(
            uow_factory=factory,
            sandbox_lifecycle_service=lifecycle,
        )

        # Spy on the (private) slot cleanup helper to verify it ran exactly
        # once even though destroy() raised. We patch the bound method so the
        # service's real coroutine isn't invoked — the supervisor is None in
        # this test anyway, so the real helper would return early, but we
        # still want a strong call-count assertion that survives future
        # refactors of the helper's internals.
        slot_cleanup_spy = AsyncMock()
        service._cleanup_background_slot_if_needed = slot_cleanup_spy  # type: ignore[assignment]

        with pytest.raises(SandboxLifecycleError, match="docker daemon unreachable"):
            await service.delete_session("s-3", user_id="u-1")

        lifecycle.destroy.assert_awaited_once()
        uow.session.delete_by_id.assert_not_awaited()
        # C3 PR-1 (codex round 13 P2): slot cleanup MUST have run despite
        # destroy() failure.
        slot_cleanup_spy.assert_awaited_once()
        call = slot_cleanup_spy.await_args
        assert call is not None
        # Positional or kwarg-friendly: tolerate either calling convention.
        # Current impl calls with (session, reason="session_delete").
        assert call.kwargs.get("reason") == "session_delete"

    async def test_slot_cleanup_runs_on_destroy_failure(self) -> None:
        """C3 PR-1 (codex round 13 P2) — Even on SandboxLifecycleError, the
        background-execution slot cleanup must run BEFORE the exception is
        re-raised. The cleanup is idempotent, so running it on the failure
        path costs nothing but prevents Redis-quota strand-until-TTL when
        destroy() crashes mid-flight.

        Pre-fix: the ``raise`` happened inside the except clause, skipping
        ``_cleanup_background_slot_if_needed`` entirely. For background
        sessions, this stranded the Redis slot until TTL/external cleanup,
        blocking quota for future background tasks even though the user had
        already requested deletion.
        """
        session = _make_session("s-4", "u-1")
        uow, factory = _make_uow_and_factory(session)

        lifecycle = MagicMock()
        lifecycle.destroy = AsyncMock(
            side_effect=SandboxLifecycleError("docker daemon unreachable")
        )

        service = SessionService(
            uow_factory=factory,
            sandbox_lifecycle_service=lifecycle,
        )

        # Track call ordering between destroy (the failing step) and the slot
        # cleanup (idempotent step that must still run).
        call_order: list[str] = []

        async def _track_destroy(*args, **kwargs):
            call_order.append("destroy")
            raise SandboxLifecycleError("docker daemon unreachable")

        async def _track_slot_cleanup(*args, **kwargs):
            call_order.append("slot_cleanup")

        lifecycle.destroy = AsyncMock(side_effect=_track_destroy)
        service._cleanup_background_slot_if_needed = AsyncMock(  # type: ignore[assignment]
            side_effect=_track_slot_cleanup
        )

        with pytest.raises(SandboxLifecycleError, match="docker daemon unreachable"):
            await service.delete_session("s-4", user_id="u-1")

        # destroy() raised; then slot cleanup ran; then the exception
        # propagated. delete_by_id never ran.
        assert call_order == ["destroy", "slot_cleanup"]
        uow.session.delete_by_id.assert_not_awaited()
