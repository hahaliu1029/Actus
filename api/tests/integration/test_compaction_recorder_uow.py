"""[R2-P1-3 part A] CHECK violation raises at INSERT execute-time and the bad row never lands.

**Note on Postgres CHECK semantics ([CXR-PR1-P2-4]):** Postgres evaluates CHECK
constraints at INSERT execution time (not at COMMIT) unless explicitly declared
DEFERRABLE INITIALLY DEFERRED. The B6 migration declares CHECKs as IMMEDIATE
(default), so the IntegrityError fires from `create_or_get`'s INSERT, BEFORE any
explicit `commit()`. This test therefore verifies "execute-with-raise" semantics
of `ck_compaction_tokens_monotonic`.

The full Path A "commit-with-raise" invariant — that a commit() failure rolls
back BOTH save_memory and record_compaction AND prevents downstream SSE event
emission — is verified by Task 16a (PR2) via explicit commit() mock injection.

The test runs inside `db_session`'s rollback-managed transaction (no leaked rows
across runs). The CHECK violation is wrapped in a SAVEPOINT (`begin_nested()`)
so the outer transaction stays alive to query and assert post-condition.

Run: cd api && uv run pytest -m integration tests/integration/test_compaction_recorder_uow.py -v
"""
import pytest
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy.exc import IntegrityError

from app.domain.models.conversation_compaction import ConversationCompaction
from app.infrastructure.repositories.db_conversation_compaction_repository import (
    DBConversationCompactionRepository,
)

# Project convention: pytest.mark.anyio (NOT asyncio). pytest-asyncio NOT installed.
pytestmark = [pytest.mark.anyio, pytest.mark.integration]


async def test_check_violation_raises_at_execute_time_and_primer_survives(
    db_session, seed_session
):
    """CHECK violation aborts only the offending statement; valid primer stays queryable.

    Uses the existing `db_session` (rollback-managed by conftest) so all writes are
    automatically cleaned up at teardown. The bad-record INSERT is wrapped in a
    SAVEPOINT so the IntegrityError doesn't poison the outer transaction.
    """
    repo = DBConversationCompactionRepository(db_session=db_session)

    # Seed a valid primer
    primer = ConversationCompaction(
        id=uuid4(), compaction_id="primer0000000001",
        session_id=seed_session.id, summary="seed",
        summary_tokens=1, first_visible_event_id=None,
        last_visible_event_id=None, pre_compact_checkpoint_id=None,
        operations=[{"kind": "llm_summary"}], parent_compaction_id=None,
        tokens_before_total=10, tokens_after_total=5, messages_removed_total=1,
        created_at=datetime.now(timezone.utc),
    )
    await repo.create_or_get(primer)
    await db_session.flush()  # makes the primer visible for subsequent SELECTs in this txn

    # Hand-craft an invalid record → violates ck_compaction_tokens_monotonic
    bad = ConversationCompaction(
        id=uuid4(), compaction_id="bad00000000000ff",
        session_id=seed_session.id, summary="bad",
        summary_tokens=1, first_visible_event_id=None,
        last_visible_event_id=None, pre_compact_checkpoint_id=None,
        operations=[{"kind": "llm_summary"}], parent_compaction_id=None,
        tokens_before_total=5, tokens_after_total=10,  # CHECK violation: after > before
        messages_removed_total=0,
        created_at=datetime.now(timezone.utc),
    )

    # SAVEPOINT so the outer transaction survives the IntegrityError.
    # Postgres aborts ONLY the inner block; outer state remains queryable.
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            await repo.create_or_get(bad)
            await db_session.flush()  # forces the INSERT to execute → CHECK fires here

    # Savepoint rolled back; primer still visible, bad never landed.
    rows = await repo.list_for_session(seed_session.id)
    ids = {r.compaction_id for r in rows}
    assert "bad00000000000ff" not in ids
    assert "primer0000000001" in ids
