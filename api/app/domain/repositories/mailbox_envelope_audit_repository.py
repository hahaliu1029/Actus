"""Mailbox envelope audit repository protocol (C3 spec §5.8).

Consumer-side terminal-dedup authority. Publisher-side Redis SET TTL fast-path
(spec §5.8 Layer 1) lives in infrastructure/external/mailbox/ and is NOT a
substitute for this audit table — see spec §5.8 hard rule.

Domain layer constraint: Protocol + pydantic + stdlib only. No SQLAlchemy.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from app.domain.models.mailbox_envelope import MailboxEnvelope


class MailboxEnvelopeAuditRepository(Protocol):
    async def get_processed(self, parent_session_id: str, envelope_id: str) -> bool: ...

    async def upsert_processing(
        self, envelope: MailboxEnvelope, *, processing_at: datetime
    ) -> None: ...

    async def mark_processed(
        self, parent_session_id: str, envelope_id: str, *, processed_at: datetime
    ) -> None: ...

    async def increment_reclaim(
        self, parent_session_id: str, envelope_id: str, last_error: str
    ) -> int: ...

    async def fetch_raw(
        self, parent_session_id: str, envelope_id: str
    ) -> dict[str, Any]: ...
