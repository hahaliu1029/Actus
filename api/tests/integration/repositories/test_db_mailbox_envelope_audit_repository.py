"""C3 PR-1 — DB-backed mailbox envelope audit repository (spec §5.8).

Integration test: requires PostgreSQL with the c3_add_mailbox_envelope_audit
migration applied (autouse `_migrate` fixture in
``tests/integration/conftest.py`` runs alembic upgrade head).

Run:
    cd api && uv run pytest \
        tests/integration/repositories/test_db_mailbox_envelope_audit_repository.py -v
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.domain.models.mailbox_envelope import (
    MailboxEnvelope,
    MailboxEnvelopeType,
    ProducerRole,
)
from app.infrastructure.repositories.db_mailbox_envelope_audit_repository import (
    DbMailboxEnvelopeAuditRepository,
)

pytestmark = pytest.mark.anyio


def _make_envelope(**overrides) -> MailboxEnvelope:
    """Build a MailboxEnvelope with unique parent/envelope ids by default."""
    prefix = uuid.uuid4().hex[:12]
    defaults = dict(
        envelope_id=f"env-{prefix}",
        type=MailboxEnvelopeType.RESULT_READY,
        parent_session_id=f"parent-{prefix}",
        child_session_id=f"child-{prefix}",
        correlation_id=f"corr-{prefix}",
        emitted_at=datetime.now(tz=timezone.utc),
        producer_role=ProducerRole.CHILD_AGENT,
        payload={"summary": "done", "outcome": "success"},
    )
    defaults.update(overrides)
    return MailboxEnvelope(**defaults)


@pytest.fixture
def repo(async_session_factory: async_sessionmaker) -> DbMailboxEnvelopeAuditRepository:
    return DbMailboxEnvelopeAuditRepository(async_session_factory)


async def test_get_processed_returns_false_when_no_row(repo):
    env = _make_envelope()
    assert await repo.get_processed(env.parent_session_id, env.envelope_id) is False


async def test_upsert_processing_then_mark_processed_round_trip(repo):
    env = _make_envelope()
    now = datetime.now(tz=timezone.utc)
    await repo.upsert_processing(env, processing_at=now)
    assert await repo.get_processed(env.parent_session_id, env.envelope_id) is False
    await repo.mark_processed(env.parent_session_id, env.envelope_id, processed_at=now)
    assert await repo.get_processed(env.parent_session_id, env.envelope_id) is True


async def test_upsert_dedups_on_parent_envelope_pair(repo):
    """Verify ON CONFLICT DO UPDATE makes duplicate (parent_session_id, envelope_id) inserts idempotent."""
    env = _make_envelope()
    now = datetime.now(tz=timezone.utc)
    await repo.upsert_processing(env, processing_at=now)
    # upsert with same key must succeed (ON CONFLICT DO UPDATE)
    await repo.upsert_processing(env, processing_at=now)
    other = env.model_copy(update={"envelope_id": f"env-other-{uuid.uuid4().hex[:8]}"})
    await repo.upsert_processing(other, processing_at=now)


async def test_increment_reclaim_returns_new_count(repo):
    # C3 PR-1 (codex P2): verify increment_reclaim returns the new count without
    # triggering expired-attribute lazy-load (production session_factory has
    # default expire_on_commit=True, so reading ``row.reclaim_count`` post-commit
    # would async-lazy-load and raise MissingGreenlet). The impl captures
    # ``new_count`` into a local before commit; this test pins the contract.
    env = _make_envelope()
    await repo.upsert_processing(env, processing_at=datetime.now(tz=timezone.utc))
    c1 = await repo.increment_reclaim(env.parent_session_id, env.envelope_id, "transient")
    c2 = await repo.increment_reclaim(env.parent_session_id, env.envelope_id, "transient")
    assert c1 == 1
    assert c2 == 2


async def test_audit_row_persists_producer_role(repo):
    """Spec §5.8 + R3 P2.1 — audit table mirrors producer_role for forensics."""
    env = _make_envelope()
    await repo.upsert_processing(env, processing_at=datetime.now(tz=timezone.utc))
    row = await repo.fetch_raw(env.parent_session_id, env.envelope_id)
    assert row["producer_role"] == "child_agent"
    assert row["type"] == "RESULT_READY"
    assert row["correlation_id"] == env.correlation_id
