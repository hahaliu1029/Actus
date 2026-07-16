"""Sandbox lifecycle domain errors.

These are domain-level exceptions raised by SandboxLifecycleService when
callers attempt operations on sessions whose sandbox binding state doesn't
permit them. The interfaces layer catches these and maps to HTTP responses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional


class SandboxLifecycleError(Exception):
    """Base class for sandbox lifecycle errors."""


class SessionUnboundError(SandboxLifecycleError):
    """Session has no sandbox bound (binding.state == UNBOUND).

    Caller should use ``bind_new()`` to create a new sandbox.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(
            f"Session {session_id} has no sandbox bound; call bind_new() first"
        )


class SessionCreatingError(SandboxLifecycleError):
    """Sandbox is being created (binding.state == CREATING).

    Caller should wait for creation to complete.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(
            f"Session {session_id} sandbox is being created; wait for completion"
        )


class SessionSuspendedError(SandboxLifecycleError):
    """Session sandbox is suspended (binding.state == SUSPENDED).

    Caller should call ``resume()`` to reactivate, or reject the request.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(
            f"Session {session_id} sandbox is suspended; call resume() to reactivate"
        )


class SessionDestroyingError(SandboxLifecycleError):
    """Session sandbox is being destroyed (binding.state == DESTROYING).

    Treat as terminal — sandbox is not recoverable.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(
            f"Session {session_id} sandbox is being destroyed; not recoverable"
        )


class SessionFinalizedError(SandboxLifecycleError):
    """Session sandbox has been destroyed (binding.state == DESTROYED, terminal).

    The sandbox is permanently gone. Caller should direct the user to start a
    new session.
    """

    def __init__(
        self,
        session_id: str,
        destroyed_at: Optional[datetime] = None,
        detail: Optional[str] = None,
    ) -> None:
        self.session_id = session_id
        self.destroyed_at = destroyed_at
        self.detail = detail
        parts = [f"Session {session_id} sandbox is permanently destroyed"]
        if destroyed_at:
            parts.append(f"at {destroyed_at.isoformat()}")
        if detail:
            parts.append(f"({detail})")
        super().__init__("; ".join(parts))


class SandboxPoisonedError(SandboxLifecycleError):
    """SandboxHandle generation mismatch — handle is stale.

    Raised when a holder's handle generation doesn't match the current
    registry generation. This means the sandbox was destroyed or
    re-created since the handle was acquired.
    """

    def __init__(
        self,
        session_id: str,
        expected_generation: int,
        actual_generation: int,
    ) -> None:
        self.session_id = session_id
        self.expected_generation = expected_generation
        self.actual_generation = actual_generation
        super().__init__(
            f"SandboxHandle for session {session_id} is stale: "
            f"handle generation={expected_generation}, "
            f"current generation={actual_generation}"
        )


class SandboxAlreadyDestroyed(SandboxLifecycleError):
    """Raised by SandboxLifecycleService.destroy() when target is already DESTROYED.

    Mailbox handlers (spec §3.2 M2 + §7.3) treat this as terminal-success:
    idempotent no-op equivalent to a fresh destroy. Distinguishing
    already-destroyed from just-destroyed is required for forensics/audit
    classification.
    """

    def __init__(self, session_id: str) -> None:
        super().__init__(f"sandbox for session {session_id} already destroyed")
        self.session_id = session_id


class SandboxBindingMissing(SandboxLifecycleError):
    """Raised when destroy() cannot find a sandbox binding (UNBOUND / missing row).

    Equivalent terminal-success: nothing to destroy, treat as if destroy already
    happened. Distinct from SandboxAlreadyDestroyed because the binding never
    reached ACTIVE — the row may have been GCed by an earlier reconcile pass.
    """

    def __init__(self, session_id: str) -> None:
        super().__init__(f"sandbox binding missing for session {session_id}")
        self.session_id = session_id


class SandboxProvisionInvalidated(SandboxLifecycleError):
    """Raised inside ``bind_new`` when an in-flight provision was invalidated by a
    concurrent ``destroy`` / ``delete`` / ``quiesce`` (SPM spec §5.2c, DD-17).

    The provision flight's ``invalidated`` outcome is captured so the caller /
    audit can distinguish a container that was torn down mid-create (CAS-1 /
    CAS-2 windows) from an ordinary provision failure. Mirrors the ctor shape of
    the other lifecycle errors: ``(session_id, invalidation_outcome)``.
    """

    def __init__(self, session_id: str, invalidation_outcome: Optional[str] = None) -> None:
        self.session_id = session_id
        self.invalidation_outcome = invalidation_outcome
        super().__init__(
            f"provision for session {session_id} was invalidated "
            f"({invalidation_outcome or 'unknown'})"
        )


class SandboxDaemonUnreachable(SandboxLifecycleError):
    """Docker daemon unreachable while inspecting / enumerating a managed sandbox.

    Raised by ``DockerSandbox.get_strict`` / ``list_managed_containers`` (SPM
    Task 5) when the Docker API errors (``APIError``) or the client itself cannot
    be constructed (no socket / daemon down). This is deliberately DISTINCT from
    the *terminal* signals — NotFound, non-running, or no-IP, all of which map to
    ``None`` / "the sandbox is gone". A daemon-unreachable is a transient
    infrastructure fault: Task 6's reconcile / label-sweep MUST NOT treat it as
    "container no longer exists", or a momentary daemon blip would sweep live
    containers. Always chained from the underlying Docker error via
    ``raise SandboxDaemonUnreachable(...) from e`` so forensics keep the cause.
    """

    def __init__(
        self,
        detail: Optional[str] = None,
        container_id: Optional[str] = None,
    ) -> None:
        self.detail = detail
        self.container_id = container_id
        parts = ["Docker daemon unreachable"]
        if container_id:
            parts.append(f"for container {container_id}")
        if detail:
            parts.append(f"({detail})")
        super().__init__("; ".join(parts))
