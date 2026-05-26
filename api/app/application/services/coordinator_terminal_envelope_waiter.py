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
"""
from __future__ import annotations

import asyncio
from typing import Any

from app.domain.external.mailbox_subscriber import MailboxSubscriber
from app.domain.models.mailbox_envelope import MailboxEnvelope, MailboxEnvelopeType


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
    ) -> MailboxEnvelope:
        """Block until a terminal envelope arrives for this child OR ``timeout``.

        Returns the validated ``MailboxEnvelope``. Raises ``asyncio.TimeoutError``
        when no terminal envelope is observed within ``timeout`` seconds.
        """
        stream_key = f"actus:child:{root_session_id}:mailbox"
        consumer_group = f"coordinator:waiter:{child_session_id}"
        consumer_name = f"waiter-{child_session_id}"
        await self._subscriber.subscribe(
            stream_key=stream_key,
            consumer_group=consumer_group,
            consumer_name=consumer_name,
        )

        async def _is_terminal_for_this_child(env: dict[str, Any]) -> bool:
            return (
                env.get("type") in _TERMINAL_TYPE_VALUES
                and env.get("child_session_id") == child_session_id
            )

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
            return await asyncio.wait_for(_consume_one(), timeout=timeout)
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"no terminal envelope for {child_session_id} after {timeout}s"
            )
