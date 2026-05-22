"""Shared fixtures for ``tests/domain/services/`` (C3 PR-3a onward).

These fixtures supply the in-memory stubs the ``MailboxSupervisor`` needs as
a ``SupervisorContext`` bag. The repo stub matches the real
``MailboxEnvelopeAuditRepository`` Protocol surface (keyword-only
``processing_at`` / ``processed_at``) so PR-3b/PR-4 can swap in a real DB
audit repo without re-typing the tests.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from app.domain.models.mailbox_envelope import MailboxEnvelope


class _InMemoryAuditRepo:
    """In-memory ``MailboxEnvelopeAuditRepository`` stub.

    PR-3a stub handlers do not actually touch the audit repo (real audit
    writes land in PR-3b). The stub exists so the ``SupervisorContext``
    bag can be constructed and so PR-3b tests can re-use the same shape.
    """

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}

    async def get_processed(self, parent_session_id: str, envelope_id: str) -> bool:
        row = self.rows.get((parent_session_id, envelope_id))
        return row is not None and row.get("processed_at") is not None

    async def upsert_processing(
        self, envelope: MailboxEnvelope, *, processing_at: datetime
    ) -> None:
        key = (envelope.parent_session_id, envelope.envelope_id)
        row = self.rows.setdefault(key, {})
        row.update(
            type=envelope.type.value,
            producer_role=envelope.producer_role.value,
            processing_at=processing_at,
            child_session_id=envelope.child_session_id,
        )

    async def mark_processed(
        self,
        parent_session_id: str,
        envelope_id: str,
        *,
        processed_at: datetime,
    ) -> None:
        self.rows.setdefault((parent_session_id, envelope_id), {})[
            "processed_at"
        ] = processed_at

    async def increment_reclaim(
        self, parent_session_id: str, envelope_id: str, last_error: str
    ) -> int:
        key = (parent_session_id, envelope_id)
        row = self.rows.setdefault(key, {"reclaim_count": 0})
        row["reclaim_count"] = row.get("reclaim_count", 0) + 1
        row["last_error"] = last_error
        return row["reclaim_count"]

    async def fetch_raw(
        self, parent_session_id: str, envelope_id: str
    ) -> dict[str, Any]:
        return dict(self.rows.get((parent_session_id, envelope_id), {}))


class _StubSandboxLifecycle:
    """Records ``destroy()`` calls without touching Docker. PR-4 will assert
    against this stub once terminal handlers actually invoke destroy."""

    def __init__(self) -> None:
        self.destroy_calls: list[tuple[str, Any]] = []

    async def destroy(self, session_id: str, reason: Any) -> None:
        self.destroy_calls.append((session_id, reason))


class _CollectingCallback:
    """Captures envelopes passed to ``ctx.agent_service_callback``.

    Used by the PR-3a stub non-terminal handler to verify dispatch routing
    landed at the in-process callback for the active agent_task_runner.
    """

    def __init__(self) -> None:
        self.received: list[MailboxEnvelope] = []

    async def __call__(self, envelope: MailboxEnvelope) -> None:
        self.received.append(envelope)


@pytest.fixture
def audit_repo() -> _InMemoryAuditRepo:
    return _InMemoryAuditRepo()


@pytest.fixture
def stub_lifecycle() -> _StubSandboxLifecycle:
    return _StubSandboxLifecycle()


@pytest.fixture
def stub_agent_callback() -> _CollectingCallback:
    return _CollectingCallback()
