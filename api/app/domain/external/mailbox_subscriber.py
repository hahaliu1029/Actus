"""C2 PR-3 §8.5.3 — MailboxSubscriber Protocol (child-side mailbox listener).

Distinct from ``MailboxConsumer`` (root-supervisor consumer that owns destroy /
persist). Coordinator child-side components (``CoordinatorChildCancelListener``
and ``CoordinatorTerminalEnvelopeWaiter``) use **independent** consumer groups
so they receive a parallel copy of envelopes from the same root-scoped stream
without racing the supervisor's group for delivery.

Wire stream key (set by callers): ``actus:child:{root_session_id}:mailbox``.

Domain layer constraint: Protocol + stdlib only; no Redis imports here.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Awaitable, Callable, Optional, Protocol


class MailboxSubscriber(Protocol):
    async def subscribe(
        self,
        *,
        stream_key: str,
        consumer_group: str,
        consumer_name: str,
        start_id: str = "$",
    ) -> None:
        """Create the consumer group on the stream (idempotent).

        BUSYGROUP (group already exists) MUST be swallowed so re-subscribe
        after a pod restart is a no-op. Other errors propagate.

        ``start_id`` (r4 P1-1): the XGROUP CREATE ``id`` argument. Default
        ``"$"`` is read-forward-only (new envelopes only — appropriate for
        first-time dispatch where the child hasn't published anything yet).
        Use ``"0"`` for the rehydrate path so the new consumer group sees
        terminal envelopes already buffered in the stream before subscribe.
        """
        ...

    def consume(
        self,
        *,
        stream_key: str,
        consumer_group: str,
        consumer_name: str,
        predicate: Callable[[dict[str, Any]], Awaitable[bool]],
        max_iterations: Optional[int] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield envelopes (decoded dicts) that ``predicate`` accepts.

        Always XACKs every consumed message (matched OR skipped) — supervisor's
        independent group keeps its own copy + does destroy/persist. ``predicate``
        runs after XREADGROUP delivery so filtering is consumer-local.

        ``max_iterations`` (test seam) caps the number of XREADGROUP loops;
        ``None`` means run forever (production listener loop).
        """
        ...

    async def destroy_group(
        self,
        *,
        stream_key: str,
        consumer_group: str,
    ) -> None:
        """Destroy the consumer group on the stream (idempotent).

        NOGROUP / NOKEY (group or stream already absent) MUST be swallowed
        so a re-destroy after pod restart is a no-op. Other Redis errors
        propagate so the orchestrator's finally-block can log them.

        The orchestrator calls this in its run() finally so the dead
        consumer group entry doesn't accumulate over many runs against
        the same long-lived root stream (Round 6 P2). Distinct from
        ``RedisMailboxConsumer.destroy_stream`` which destroys + deletes
        the entire stream key — here we only destroy the per-run group
        and leave the shared root stream intact for siblings.
        """
        ...
