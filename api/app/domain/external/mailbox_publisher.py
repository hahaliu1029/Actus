"""MailboxPublisher domain protocol (C3 spec §11.1).

Implementations:
  - infrastructure/external/mailbox/redis_mailbox_publisher.py (PR-2)
  - tests use a fakes.InMemoryMailboxPublisher that captures envelopes.

Domain layer constraint: Protocol + pydantic + stdlib only. No Redis imports.
"""

from __future__ import annotations

from typing import Protocol

from app.domain.models.mailbox_envelope import MailboxEnvelope


class MailboxPublisher(Protocol):
    async def publish(self, envelope: MailboxEnvelope) -> None:
        """XADD envelope to actus:child:{root_session_id}:mailbox.

        Publisher-side idempotency (Redis SET TTL fast-path, spec §5.8 Layer 1)
        is the publisher's concern; consumer-side dedup uses DB processed_at.

        Raises:
            MailboxPublishOversize: payload exceeds APPROVAL_PAYLOAD_MAX_BYTES
            ConnectionError / RedisError: lower-level Redis failures
        """
        ...


class MailboxPublishOversize(Exception):
    """Raised when envelope payload exceeds APPROVAL_PAYLOAD_MAX_BYTES."""
