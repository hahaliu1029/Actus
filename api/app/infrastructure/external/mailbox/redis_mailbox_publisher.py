"""Publisher-side mailbox client (C3 spec §5.1 + §5.8 Layer 1 + §5.9).

Owns:
  - XADD with MAXLEN ~ trim (approximate=True per §5.9 — exact MAXLEN
    is ~10x more expensive per XADD, and we only need a soft cap).
  - Redis SET NX TTL fast-path dedup (publisher-side only; consumer dedup
    lives in the audit repo via ``processed_at``. **§5.8 hard rule**:
    publisher-side Redis SET dedup is a perf fast-path, NOT a substitute
    for consumer-side terminal dedup. Do NOT cross-wire these layers —
    PR-3b + PR-4 own consumer-side dedup.)
  - Payload-size guard (``APPROVAL_PAYLOAD_MAX_BYTES``). Oversize raises
    ``MailboxPublishOversize`` **before** touching Redis SET, so a caller
    that retries an oversized envelope cannot pollute the dedup cache
    with a key that suppresses a later (potentially under-cap) attempt.

Does NOT own:
  - Consumer-side dedup (mailbox_supervisor + audit repo, PR-3a/PR-4).
  - XACK / XAUTOCLAIM / XPENDING (consumer ops in PR-2 consumer + PR-3a/3b).
"""

from __future__ import annotations

import logging

from redis.asyncio import Redis

from app.domain.external.mailbox_publisher import (
    MailboxPublisher,
    MailboxPublishOversize,
)
from app.domain.models.mailbox_envelope import (
    APPROVAL_PAYLOAD_MAX_BYTES,
    MAILBOX_DEDUP_KEY_TEMPLATE,
    MAILBOX_DEDUP_TTL_SECONDS,
    MAILBOX_STREAM_KEY_TEMPLATE,
    MAILBOX_STREAM_MAXLEN_APPROX,
    MailboxEnvelope,
)


logger = logging.getLogger(__name__)


class RedisMailboxPublisher(MailboxPublisher):
    """Thin XADD wrapper. Phase 1 ``max_depth=1`` (subagent_limits.MAX_SUBAGENT_DEPTH=1)
    means ``parent_session_id == root_session_id``, so the stream key derives
    directly from ``envelope.parent_session_id``. Phase 3+ will need to walk up
    the session tree (open question §16.7 in the C3 spec).
    """

    def __init__(
        self,
        redis: Redis,
        *,
        maxlen_approx: int = MAILBOX_STREAM_MAXLEN_APPROX,
    ) -> None:
        self._redis = redis
        self._maxlen_approx = maxlen_approx

    async def publish(self, envelope: MailboxEnvelope) -> None:
        """XADD envelope to ``actus:child:{root_session_id}:mailbox``.

        Order is load-bearing:
          1. Serialize + size-check **first**. Oversize raises BEFORE we
             touch the Redis SET dedup key — otherwise a retried oversize
             envelope would acquire the dedup lease then refuse the XADD,
             leaving a poison TTL that suppresses the corrected retry.
          2. ``SET NX EX`` claim the dedup key (publisher-side fast-path,
             §5.8 Layer 1 — supervisor still re-verifies via DB
             ``processed_at`` per §5.8 hard rule).
          3. XADD with ``maxlen + approximate=True`` so the stream is
             soft-capped at ``MAILBOX_STREAM_MAXLEN_APPROX``.

        Failure semantics:
          * If serialization or size-check fails, no Redis state changes.
          * If SET NX succeeded but XADD raises, the dedup key is rolled
            back (DEL) so retry can re-publish. Brief window where a
            successful XADD-but-failed-DEL leaves a stale dedup key —
            acceptable because consumer-side DB ``processed_at`` is the
            authoritative dedup layer (§5.8 hard rule), so a duplicate
            envelope at worst is skipped consumer-side, never
            double-applied.
        """
        serialized = envelope.model_dump_json().encode("utf-8")
        if len(serialized) > APPROVAL_PAYLOAD_MAX_BYTES:
            raise MailboxPublishOversize(
                f"envelope {envelope.envelope_id} serialized size "
                f"{len(serialized)} exceeds {APPROVAL_PAYLOAD_MAX_BYTES}"
            )

        dedup_key = MAILBOX_DEDUP_KEY_TEMPLATE.format(
            parent_session_id=envelope.parent_session_id,
            envelope_id=envelope.envelope_id,
        )
        was_set = await self._redis.set(
            dedup_key, "1", nx=True, ex=MAILBOX_DEDUP_TTL_SECONDS
        )
        if not was_set:
            logger.debug(
                "publisher dedup hit envelope_id=%s parent_session_id=%s",
                envelope.envelope_id,
                envelope.parent_session_id,
            )
            return

        stream_key = MAILBOX_STREAM_KEY_TEMPLATE.format(
            root_session_id=envelope.parent_session_id
        )
        try:
            await self._redis.xadd(
                stream_key,
                fields={
                    # PR-3a supervisor reads ``envelope`` as the canonical
                    # payload. The remaining fields are observational —
                    # XLEN-style debugging from redis-cli / RedisInsight
                    # without having to parse JSON. Keep this shape stable;
                    # changing field names is a wire-break.
                    "envelope": serialized,
                    "envelope_id": envelope.envelope_id,
                    "type": envelope.type.value,
                    "producer_role": envelope.producer_role.value,
                },
                maxlen=self._maxlen_approx,
                approximate=True,
            )
        except Exception:
            # Rollback dedup so a retry isn't silently dedup-skipped.
            # Without this, a transient XADD failure (network blip, broken
            # pipe, etc.) would leave the dedup key live for 24h and the
            # next publish() call would short-circuit via ``not was_set``,
            # losing the envelope entirely.
            try:
                await self._redis.delete(dedup_key)
            except Exception:
                logger.exception(
                    "failed to rollback mailbox dedup key after xadd "
                    "failure key=%s",
                    dedup_key,
                )
            logger.warning(
                "mailbox XADD failed, rolled back dedup envelope_id=%s "
                "parent_session_id=%s",
                envelope.envelope_id,
                envelope.parent_session_id,
            )
            raise
