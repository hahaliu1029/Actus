"""C2 coordinator-cancel Part B — leaked-sandbox startup reaper (spec §3.5).

Mirrors the C2b child-row reaper (``child_terminal_reconciler.py``): a startup
match-only sweep that destroys the leaked ACTIVE sandbox of every terminal
coordinator child. On user-stop the root ``MailboxSupervisor`` is killed before
it consumes the cancelled children's ``CANCEL_ACK``, so each child's per-child
sandbox container is left ACTIVE with no reaper (F0.6/F0.7). This sweep closes
that leak idempotently at the next pod restart (restart-bounded — NG8).

The reaper rehydrates the in-process registry from the binding ONLY when the
container is actually found (``try_register_from_binding``) before calling the
shared ``destroy()`` — so a fresh-process destroy actually ``docker rm``s the
live container instead of marking the row DESTROYED while it leaks (R4 P1 / R5
P2). A gone/unreachable container leaves the row ACTIVE (no false DESTROYED).
One bad child never aborts the sweep; ``CancelledError`` propagates; a query/DI
failure propagates to the caller's OUTER best-effort try (``main.py``).

Application-layer only — no FastAPI / SQLAlchemy / domain change (INV-C7).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from app.domain.errors.sandbox_lifecycle import (
    SandboxAlreadyDestroyed,
    SandboxBindingMissing,
)
from app.domain.models.session import DestroyReason

logger = logging.getLogger(__name__)


@dataclass
class SandboxReapStats:
    """Outcome counters for one startup sweep (logged when non-trivial)."""

    scanned: int = 0
    destroyed: int = 0
    already_gone: int = 0
    errored: int = 0


async def sweep_terminal_coordinator_active_sandboxes(
    *,
    session_repo: Any,
    lifecycle_service: Any,
) -> SandboxReapStats:
    """Scan terminal coordinator children with an ACTIVE sandbox and destroy
    each leaked container (match-only, idempotent).

    The query (``find_terminal_coordinator_children_with_active_sandbox``) is
    NOT wrapped here — a query/DI failure propagates to the caller's OUTER
    best-effort try (main.py) so it cannot abort lifespan startup. The INNER
    per-child try isolates one bad child. ``asyncio.CancelledError`` always
    propagates.
    """
    stats = SandboxReapStats()
    children = await session_repo.find_terminal_coordinator_children_with_active_sandbox()
    for child in children:
        stats.scanned += 1
        try:
            if not await lifecycle_service.try_register_from_binding(child.session_id):
                # Container gone/unreachable -> leave row ACTIVE (no false
                # DESTROYED); a harmless phantom re-scanned next boot.
                stats.already_gone += 1
                continue
            await lifecycle_service.destroy(
                child.session_id, DestroyReason.TERMINAL_CHILD_REAPER
            )
            stats.destroyed += 1
        except (SandboxAlreadyDestroyed, SandboxBindingMissing):
            # Already-gone bindings are terminal-success, never errors
            # (idempotent; e.g. a respawned supervisor won the destroy race —
            # serialized by destroy()'s per-session lock, loser lands here).
            stats.already_gone += 1
        except asyncio.CancelledError:
            raise
        except Exception:
            stats.errored += 1
            logger.exception(
                "sandbox_reaper: destroy failed child=%s", child.session_id
            )
    return stats
