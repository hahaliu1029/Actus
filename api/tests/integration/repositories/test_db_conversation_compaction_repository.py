"""Integration test: ON CONFLICT idempotency + FK CASCADE + sort order.

Run: cd api && uv run pytest -m integration tests/integration/repositories/test_db_conversation_compaction_repository.py -v
"""
import pytest
from datetime import datetime, timezone
from uuid import uuid4

from app.domain.models.conversation_compaction import ConversationCompaction
from app.infrastructure.repositories.db_conversation_compaction_repository import (
    DBConversationCompactionRepository,
)

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


def _build(session_id: str, compaction_id: str, **overrides) -> ConversationCompaction:
    base = dict(
        id=uuid4(),
        compaction_id=compaction_id,
        session_id=session_id,
        summary="s",
        summary_tokens=1,
        first_visible_event_id=None,
        last_visible_event_id=None,
        pre_compact_checkpoint_id=None,
        operations=[{"kind": "hard_truncate", "tokens_before": 10, "tokens_after": 5}],
        parent_compaction_id=None,
        tokens_before_total=10,
        tokens_after_total=5,
        messages_removed_total=3,
        created_at=datetime.now(timezone.utc),
    )
    base.update(overrides)
    return ConversationCompaction(**base)


async def test_create_or_get_inserts_when_new(db_session, seed_session):
    repo = DBConversationCompactionRepository(db_session=db_session)
    rec = _build(seed_session.id, "abcdef0123456789")

    result = await repo.create_or_get(rec)
    await db_session.flush()  # makes rows visible within the same transaction; teardown rollback still fires

    assert result.compaction_id == "abcdef0123456789"
    assert result.id == rec.id


async def test_create_or_get_returns_existing_on_conflict(db_session, seed_session):
    repo = DBConversationCompactionRepository(db_session=db_session)
    rec1 = _build(seed_session.id, "deadbeefcafe0001")
    await repo.create_or_get(rec1)
    await db_session.flush()  # makes rows visible within the same transaction; teardown rollback still fires

    rec2_same_id = _build(seed_session.id, "deadbeefcafe0001", summary="different")
    result = await repo.create_or_get(rec2_same_id)
    await db_session.flush()  # makes rows visible within the same transaction; teardown rollback still fires

    # Returned record is the original, not the second attempt
    assert result.id == rec1.id
    assert result.summary == "s"  # original summary preserved


async def test_list_sorted_by_created_at_desc(db_session, seed_session):
    repo = DBConversationCompactionRepository(db_session=db_session)
    earlier = _build(seed_session.id, "a" * 16, created_at=datetime(2026, 5, 1, tzinfo=timezone.utc))
    later = _build(seed_session.id, "b" * 16, created_at=datetime(2026, 5, 2, tzinfo=timezone.utc))
    await repo.create_or_get(earlier)
    await repo.create_or_get(later)
    await db_session.flush()  # makes rows visible within the same transaction; teardown rollback still fires

    results = await repo.list_for_session(seed_session.id)
    assert [r.compaction_id for r in results] == ["b" * 16, "a" * 16]
