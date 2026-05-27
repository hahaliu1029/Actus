"""C2 PR-3 §7.5 — CoordinatorTerminalEnvelopeWaiter.

Awaits the terminal envelope (RESULT_READY | CANCEL_ACK) for a given child
session from the root-scoped mailbox stream, using an **independent** consumer
group ``coordinator:waiter:{child_session_id}`` so the supervisor's group
(``actus:mailbox-supervisor:v1``) still receives its own copy + does
destroy/persist/mark_processed.

XREADGROUP semantics — Redis delivers each message once per group; multiple
groups receive independent copies. The waiter must not use ``RedisMailboxConsumer``
directly because that consumer is hard-wired to the supervisor's fixed group
name (would race the supervisor for terminal envelope delivery).

Timeout enforced via ``asyncio.wait_for`` because ``subscriber.consume`` is a
``while True`` loop that would otherwise block forever when no terminal envelope
arrives. ``cancel_event`` is informational (the child uses it to stop + emit
CANCEL_ACK); the waiter keeps consuming until terminal arrives or timeout fires.

[Round 7 P2] After the waiter's job is done (terminal arrives, timeout, or
exception), the per-waiter consumer group is destroyed in a ``finally`` block
so dead groups don't accumulate under the long-lived root mailbox stream.
The destroy is gated on a ``subscribed=True`` flag — if ``subscribe`` itself
raised, no group exists yet and ``destroy_group`` would emit a spurious
NOGROUP roundtrip. ``destroy_group`` failures are logged + swallowed so the
caller-facing exception (TimeoutError, etc.) propagates unmodified.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.domain.external.mailbox_subscriber import MailboxSubscriber
from app.domain.models.mailbox_envelope import MailboxEnvelope, MailboxEnvelopeType


logger = logging.getLogger(__name__)


_TERMINAL_TYPE_VALUES: frozenset[str] = frozenset({
    MailboxEnvelopeType.RESULT_READY.value,
    MailboxEnvelopeType.CANCEL_ACK.value,
})


class CoordinatorTerminalEnvelopeWaiter:
    def __init__(self, *, subscriber: MailboxSubscriber) -> None:
        self._subscriber = subscriber

    async def await_terminal(
        self,
        *,
        child_session_id: str,
        root_session_id: str,
        cancel_event: asyncio.Event,
        timeout: float = 600.0,
        coordinator_run_id: str | None = None,
    ) -> MailboxEnvelope:
        """Block until a terminal envelope arrives for this child OR ``timeout``.

        Returns the validated ``MailboxEnvelope``. Raises ``asyncio.TimeoutError``
        when no terminal envelope is observed within ``timeout`` seconds.

        ``coordinator_run_id`` is optional [codex R5 P1] — when set the
        predicate additionally matches ``env['correlation_id']`` so the
        waiter never accepts a stale RESULT_READY from a different
        coordinator run sharing the same child_session_id (e.g. PR-7
        rehydrate of a different attempt, or a re-issued child). PR-5
        applies patch_manifests from these envelopes directly to the
        parent sandbox, so a mismatched run_id would cross-contaminate
        apply plans. Kept optional with default ``None`` for backward
        compat with PR-3 tests that haven't migrated yet.
        """
        stream_key = f"actus:child:{root_session_id}:mailbox"
        consumer_group = f"coordinator:waiter:{child_session_id}"
        consumer_name = f"waiter-{child_session_id}"
        subscribed = False
        await self._subscriber.subscribe(
            stream_key=stream_key,
            consumer_group=consumer_group,
            consumer_name=consumer_name,
        )
        subscribed = True

        async def _is_terminal_for_this_child(env: dict[str, Any]) -> bool:
            if env.get("type") not in _TERMINAL_TYPE_VALUES:
                return False
            if env.get("child_session_id") != child_session_id:
                return False
            # [codex R5 P1] When the caller provides
            # coordinator_run_id, require the envelope's
            # correlation_id to match. Without this guard a stale
            # RESULT_READY from a prior run with the same
            # child_session_id could be applied to the wrong run.
            if (
                coordinator_run_id is not None
                and env.get("correlation_id") != coordinator_run_id
            ):
                return False
            return True

        async def _consume_one() -> MailboxEnvelope:
            async for env_dict in self._subscriber.consume(
                stream_key=stream_key,
                consumer_group=consumer_group,
                consumer_name=consumer_name,
                predicate=_is_terminal_for_this_child,
            ):
                return MailboxEnvelope.model_validate(env_dict)
            raise asyncio.TimeoutError(
                f"subscriber exhausted before terminal for {child_session_id}"
            )

        try:
            try:
                return await asyncio.wait_for(_consume_one(), timeout=timeout)
            except asyncio.TimeoutError:
                raise asyncio.TimeoutError(
                    f"no terminal envelope for {child_session_id} after {timeout}s"
                )
        finally:
            # [Round 7 P2] Destroy the per-waiter consumer group so it doesn't
            # accumulate as a dead XPENDING/group-metadata entry under the
            # long-lived root stream. Skip when ``subscribed=False`` (subscribe
            # itself raised — nothing to destroy). Best-effort: log + swallow
            # on any error — the group is per-waiter and idempotent destroy is
            # the adapter contract.
            if subscribed:
                try:
                    await self._subscriber.destroy_group(
                        stream_key=stream_key,
                        consumer_group=consumer_group,
                    )
                except Exception:
                    logger.warning(
                        "CoordinatorTerminalEnvelopeWaiter: destroy_group"
                        " failed child=%s group=%s — dead group may accumulate"
                        " in Redis",
                        child_session_id, consumer_group,
                        exc_info=True,
                    )
