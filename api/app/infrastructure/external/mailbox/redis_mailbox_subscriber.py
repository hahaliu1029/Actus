"""C2 PR-3 §8.5.3 — RedisMailboxSubscriber.

Independent consumer group from ``RedisMailboxConsumer`` (the latter is hard-
wired to ``MAILBOX_CONSUMER_GROUP_NAME = 'actus:mailbox-supervisor:v1'``).
Coordinator child-side callers pass their own ``consumer_group`` so they
receive a parallel copy of stream envelopes without racing the supervisor.

XREADGROUP semantics — Redis delivers each message once per group; multiple
groups all receive independent copies. ``CoordinatorChildCancelListener`` uses
``coordinator:child:{child_session_id}`` and
``CoordinatorTerminalEnvelopeWaiter`` uses ``coordinator:waiter:{child_session_id}``;
neither collides with the supervisor's destroy/persist path.
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from app.domain.external.mailbox_subscriber import MailboxSubscriber

logger = logging.getLogger(__name__)


class RedisMailboxSubscriber(MailboxSubscriber):
    """XREADGROUP + filter + always-XACK implementation."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def subscribe(
        self,
        *,
        stream_key: str,
        consumer_group: str,
        consumer_name: str,
        start_id: str = "$",
    ) -> None:
        try:
            await self._redis.xgroup_create(
                name=stream_key,
                groupname=consumer_group,
                id=start_id,
                mkstream=True,
            )
        except Exception as exc:
            # BUSYGROUP = "Consumer Group name already exists" — idempotent re-subscribe.
            if "BUSYGROUP" not in str(exc):
                raise
        logger.debug(
            "RedisMailboxSubscriber: subscribed group=%s on stream=%s as consumer=%s start_id=%s",
            consumer_group, stream_key, consumer_name, start_id,
        )

    async def destroy_group(
        self,
        *,
        stream_key: str,
        consumer_group: str,
    ) -> None:
        """[Round 6 P2] Destroy the per-run consumer group so it doesn't
        accumulate as a dead XPENDING/group-metadata entry under the
        long-lived root stream. Idempotent on NOGROUP / NOKEY (already
        destroyed or stream gone — both fine); other ResponseErrors
        propagate so the orchestrator can log them.
        """
        try:
            await self._redis.xgroup_destroy(stream_key, consumer_group)
        except ResponseError as exc:
            msg = str(exc).upper()
            if "NOGROUP" in msg or "NO SUCH KEY" in msg or "NOKEY" in msg:
                # Already destroyed / stream gone — idempotent re-destroy.
                return
            raise

    async def consume(
        self,
        *,
        stream_key: str,
        consumer_group: str,
        consumer_name: str,
        predicate: Callable[[dict[str, Any]], Awaitable[bool]],
        max_iterations: Optional[int] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        iter_count = 0
        while True:
            if max_iterations is not None and iter_count >= max_iterations:
                return
            iter_count += 1
            try:
                resp = await self._redis.xreadgroup(
                    groupname=consumer_group,
                    consumername=consumer_name,
                    streams={stream_key: ">"},
                    count=10,
                    block=2000,
                )
            except Exception as exc:
                logger.warning(
                    "RedisMailboxSubscriber xreadgroup error (group=%s, stream=%s): %s",
                    consumer_group, stream_key, exc,
                )
                continue
            if not resp:
                continue
            for _stream, entries in resp:
                for msg_id, fields in entries:
                    try:
                        # Live publisher writes field name "envelope" as JSON bytes/str.
                        raw = (
                            fields.get(b"envelope")
                            if isinstance(fields, dict) else None
                        )
                        if raw is None and isinstance(fields, dict):
                            raw = fields.get("envelope")
                        if raw is None:
                            raw = "{}"
                        if isinstance(raw, bytes):
                            raw = raw.decode("utf-8")
                        env = json.loads(raw)
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        logger.warning(
                            "RedisMailboxSubscriber bad envelope payload msg_id=%s: %s",
                            msg_id, exc,
                        )
                        try:
                            await self._redis.xack(stream_key, consumer_group, msg_id)
                        except Exception as ack_exc:
                            logger.warning(
                                "RedisMailboxSubscriber xack-after-bad-json failed "
                                "msg_id=%s: %s", msg_id, ack_exc,
                            )
                        continue
                    # r5 P1-3: isolate per-message predicate/xack errors so a
                    # transient failure doesn't kill the entire listener loop.
                    # Without this, a single bad predicate raise (e.g. KeyError
                    # in caller's lambda) would bubble out of consume() and
                    # take down the listener task.
                    try:
                        matched = await predicate(env)
                    except Exception as exc:
                        logger.warning(
                            "RedisMailboxSubscriber predicate raised msg_id=%s: %s",
                            msg_id, exc,
                        )
                        matched = False
                    try:
                        # Always XACK — supervisor keeps its own copy on its group.
                        await self._redis.xack(stream_key, consumer_group, msg_id)
                    except Exception as exc:
                        logger.warning(
                            "RedisMailboxSubscriber xack failed msg_id=%s: %s",
                            msg_id, exc,
                        )
                    if matched:
                        yield env
