from __future__ import annotations

from typing import Protocol


class MemoryNotificationEmitter(Protocol):
    """Domain-side contract for writing a ``memory_system_notifications``
    row from background flow paths (gate, fs retry, etc).

    Kept as a Protocol rather than a direct repository call because the
    call site—``PlannerReActFlow._apply_llm_gate``—is a per-session domain
    object that has no awareness of the DB session pool; the emitter
    closure owns session lifecycle internally (create session → write →
    commit → close) so the flow just fires events.

    Failures are **swallowed internally** and logged; notification
    emission must not break the flush path. A missed notification is
    strictly better than a lost flush.
    """

    async def emit(
        self,
        *,
        user_id: str,
        event_type: str,
        payload: dict,
    ) -> None:
        ...
