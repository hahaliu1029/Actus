"""C3 PR-3c — domain-side Protocol for the per-pod MailboxSupervisor registry.

``AgentTaskRunner`` (domain layer) needs to spawn / stop a supervisor when the
root session transitions to RUNNING / terminal, but the concrete
``SupervisorRegistry`` implementation lives in ``application/`` because it
depends on ``MailboxSupervisor`` (also application). Domain → application is
the forbidden direction (CLAUDE.md backend-patterns invariant — domain may
import langchain-core / langgraph / pydantic / stdlib only).

This Protocol keeps the runner's dependency purely structural: the
application-layer ``SupervisorRegistry`` satisfies it by duck typing (no
``class SupervisorRegistry(SupervisorRegistryPort)`` inheritance needed),
so the directional arrow stays application → domain.

The Protocol is intentionally minimal — only the three methods callers
outside ``application/`` need: ``spawn`` + ``stop`` for runner lifecycle
hooks, and ``health_check`` for ``SandboxLifecycleService.reconcile_orphans``
to decide whether a slot is already alive/restarting/crashed before
re-spawning. Internal restart loop, readiness-event injection,
per-slot bookkeeping, etc. stay private to the concrete registry.
"""

from __future__ import annotations

from typing import Mapping, Protocol


class SupervisorRegistryPort(Protocol):
    """Domain-facing surface for the per-pod MailboxSupervisor registry.

    The mutating methods (``spawn`` / ``stop``) MUST be idempotent — the
    runner's hooks fire on the happy path (transition to RUNNING /
    terminal) AND on retry / restart paths (FINISHING resume,
    reconcile_orphans). The concrete ``SupervisorRegistry`` enforces this:

    - ``spawn(root_id)`` short-circuits when a slot already exists.
    - ``stop(root_id)`` no-ops on unknown root.

    ``health_check()`` is a pure read — it returns a per-root state
    mapping used by ``SandboxLifecycleService.reconcile_orphans`` to
    decide whether to re-spawn a slot.
    """

    async def spawn(self, root_session_id: str) -> None:
        """Ensure a supervisor task is running for ``root_session_id``.

        Idempotent — calling twice with the same id is a no-op.
        """
        ...

    async def stop(self, root_session_id: str) -> None:
        """Cancel + remove the supervisor for ``root_session_id``.

        Idempotent — calling for an unknown / already-stopped root is a no-op.
        """
        ...

    async def health_check(self) -> Mapping[str, str]:
        """Snapshot per-root supervisor state.

        Returns a mapping ``{root_session_id: state}`` where ``state`` is
        one of the values defined by the concrete registry (PR-3b ships
        ``"alive" | "crashed"``; ``"restarting"`` is reserved for a future
        transient state).

        Used by ``SandboxLifecycleService.reconcile_orphans`` to avoid
        re-spawning a supervisor that is already alive / in the middle of
        being restarted by the per-pod restart loop.
        """
        ...
