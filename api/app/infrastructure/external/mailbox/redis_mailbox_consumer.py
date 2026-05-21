"""Consumer-side mailbox thin wrapper (C3 spec §5.2 / §5.3 / §5.4 / §5.6 / §5.10).

Stateless XGROUP / XREADGROUP / XACK / XAUTOCLAIM / XGROUP DESTROY shim around
``redis.asyncio.Redis``. Owns no business logic — the supervisor (PR-3a)
decides what to do with read entries. Owns no idempotency — consumer-side
dedup lives in the audit repo (``processed_at``, PR-1) per §5.8 hard rule.

Each ``RedisMailboxConsumer`` is bound to one ``root_session_id`` stream and
one logical ``(pod_id, instance_id)`` consumer name. The supervisor creates
one consumer per root it owns; the consumer is not shared across roots.
"""

from __future__ import annotations

import logging
from typing import Optional

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from app.domain.models.mailbox_envelope import (
    MAILBOX_CONSUMER_GROUP_NAME,
    MAILBOX_STREAM_KEY_TEMPLATE,
    MailboxEnvelope,
)


logger = logging.getLogger(__name__)


class RedisMailboxConsumer:
    """Thin redis-py async wrapper. No business logic, no dedup."""

    def __init__(
        self,
        redis: Redis,
        root_session_id: str,
        pod_id: str,
        instance_id: str,
    ) -> None:
        self._redis = redis
        self._root = root_session_id
        # Consumer-name uniqueness invariant: ``{pod_id}:{root_session_id}:{instance_id}``.
        # pod_id keeps cross-pod consumers distinct so XAUTOCLAIM can detect
        # the other side. instance_id avoids self-collision when one pod
        # rebuilds a supervisor for the same root (e.g., after a graceful
        # restart inside the same pod) — XPENDING would otherwise treat the
        # new supervisor as the old idle owner.
        self._consumer_name = f"{pod_id}:{root_session_id}:{instance_id}"
        self._stream_key = MAILBOX_STREAM_KEY_TEMPLATE.format(
            root_session_id=root_session_id
        )

    async def ensure_group(self) -> None:
        """``XGROUP CREATE ... 0-0 MKSTREAM`` — idempotent.

        Swallows ``BUSYGROUP Consumer Group name already exists`` because
        the supervisor is allowed to retry ensure_group on startup or
        after a XAUTOCLAIM-driven recovery.
        """
        try:
            await self._redis.xgroup_create(
                name=self._stream_key,
                groupname=MAILBOX_CONSUMER_GROUP_NAME,
                id="0-0",
                mkstream=True,
            )
        except ResponseError as e:
            if "BUSYGROUP" in str(e):
                return  # already exists, idempotent path
            raise

    async def read(
        self, *, count: int, block_ms: int
    ) -> list[tuple[bytes, MailboxEnvelope]]:
        """``XREADGROUP ... STREAMS key >`` — only new (not-yet-delivered) entries.

        Returns ``[(redis_id, parsed_envelope), ...]``. Empty list when the
        block timed out without new entries or every entry was a poison.
        """
        result = await self._redis.xreadgroup(
            groupname=MAILBOX_CONSUMER_GROUP_NAME,
            consumername=self._consumer_name,
            streams={self._stream_key: ">"},
            count=count,
            block=block_ms,
        )
        return self._parse_xread_result(result)

    async def ack(self, redis_id: bytes) -> None:
        """``XACK`` a single entry. Caller (supervisor) is responsible for
        per-entry dedup before XACK — never XACK an unprocessed entry."""
        await self._redis.xack(
            self._stream_key, MAILBOX_CONSUMER_GROUP_NAME, redis_id
        )

    async def autoclaim(
        self,
        *,
        min_idle_ms: int,
        count: int,
        start_id: str = "0-0",
    ) -> list[tuple[bytes, MailboxEnvelope]]:
        """``XAUTOCLAIM`` — startup + periodic PEL sweep (spec §5.6).

        redis-py 7.x signature::

            xautoclaim(name, groupname, consumername, min_idle_time,
                       start_id='0-0', count=None, justid=False)

        Return shape: ``(next_cursor, [(redis_id, fields), ...], deleted_ids)``.
        Adapt here if the underlying lib version changes.
        """
        result = await self._redis.xautoclaim(
            name=self._stream_key,
            groupname=MAILBOX_CONSUMER_GROUP_NAME,
            consumername=self._consumer_name,
            min_idle_time=min_idle_ms,
            start_id=start_id,
            count=count,
        )
        # redis-py returns 3-tuple: (next_start_id, claimed_entries, deleted_ids)
        _next_cursor, entries, _deleted = result
        return self._parse_entries(entries)

    async def pending_idle_ms(self, redis_id: bytes) -> Optional[int]:
        """``XPENDING ... IDLE`` filtered by id — returns ms-since-delivery
        or ``None`` if not pending (already ACKed or never delivered).
        """
        rid = redis_id.decode() if isinstance(redis_id, bytes) else redis_id
        info = await self._redis.xpending_range(
            name=self._stream_key,
            groupname=MAILBOX_CONSUMER_GROUP_NAME,
            min=rid,
            max=rid,
            count=1,
        )
        if not info:
            return None
        first = info[0]
        # redis-py returns dict with string keys when decode_responses=True,
        # bytes keys otherwise. Coalesce.
        return int(
            first.get("time_since_delivered", first.get(b"time_since_delivered", 0))
        )

    async def destroy_stream(self) -> None:
        """Root terminal cleanup (spec §5.10) — destroy the consumer group
        then delete the stream key. Idempotent on NOGROUP / missing key
        only; unexpected ResponseError (e.g. WRONGTYPE) propagates so a
        corrupted key isn't silently DEL'd.
        """
        try:
            await self._redis.xgroup_destroy(
                self._stream_key, MAILBOX_CONSUMER_GROUP_NAME
            )
        except ResponseError as e:
            msg = str(e).upper()
            if "NOGROUP" in msg or "NO SUCH KEY" in msg:
                # Group already destroyed / stream already deleted — both fine.
                pass
            else:
                # WRONGTYPE or unknown — surface, don't proceed to DEL.
                raise
        await self._redis.delete(self._stream_key)

    # ------------------------------------------------------------------
    # Internal parsers
    # ------------------------------------------------------------------

    def _parse_xread_result(
        self, result
    ) -> list[tuple[bytes, MailboxEnvelope]]:
        out: list[tuple[bytes, MailboxEnvelope]] = []
        for _stream_name, entries in (result or []):
            out.extend(self._parse_entries(entries))
        return out

    def _parse_entries(self, entries) -> list[tuple[bytes, MailboxEnvelope]]:
        """Best-effort parse. Poison entries (missing/malformed ``envelope``
        field) are logged + skipped; we do NOT re-raise — one bad entry
        must not poison the whole batch read. The supervisor's XAUTOCLAIM
        reclaim counter eventually moves poison entries to the dead-letter
        path (PR-3b).
        """
        out: list[tuple[bytes, MailboxEnvelope]] = []
        for redis_id, fields in entries or []:
            try:
                raw = fields.get(b"envelope") or fields.get("envelope")
                if raw is None:
                    logger.warning(
                        "mailbox stream entry missing 'envelope' field "
                        "id=%s — skipping (poison)",
                        redis_id,
                    )
                    continue
                parsed = MailboxEnvelope.model_validate_json(
                    raw.decode() if isinstance(raw, bytes) else raw
                )
                # Normalize redis_id: production uses decode_responses=True
                # (str ids), tests/fakeredis often run decode_responses=False
                # (bytes ids). Normalize to bytes so the wrapper's public
                # type contract (bytes ids end-to-end) doesn't depend on
                # Redis client config.
                normalized_id = (
                    redis_id
                    if isinstance(redis_id, bytes)
                    else redis_id.encode("utf-8")
                )
                out.append((normalized_id, parsed))
            except Exception as e:  # noqa: BLE001 — broad on purpose
                # NEVER re-raise: a single malformed entry must not kill
                # the batch. The PEL retains the bad entry until
                # XAUTOCLAIM increments reclaim_count past
                # MAILBOX_POISON_MAX_RECLAIM (PR-3b) at which point it
                # gets moved to the dead-letter sink.
                logger.exception(
                    "failed to parse mailbox envelope id=%s: %s",
                    redis_id,
                    e,
                )
        return out
