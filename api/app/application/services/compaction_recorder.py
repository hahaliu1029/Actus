"""Application helper: persist a Path A compaction inside an active UoW.

Spec: docs/superpowers/specs/2026-05-02-b6-compaction-metadata-persistence-design.md

Caller responsibilities:
- Pre-compute `messages_input_hash` from the input message list (use
  `compaction.compute_messages_input_hash(messages)` BEFORE try_compact mutates them).
- Hold an active UoW and call `await uow.db_session.commit()` after this returns
  to satisfy the [R2-P1-3] commit-with-raise contract.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from uuid import uuid4

from app.domain.models.conversation_compaction import ConversationCompaction
from app.domain.repositories.uow import IUnitOfWork
from app.domain.services.graphs.compaction import (
    CompactionResult,
    derive_compaction_id,
)

_SUMMARY_MAX_CHARS = 16_000


async def record_compaction(
    uow: IUnitOfWork,
    session_id: str,
    result: CompactionResult,
    messages_input_hash: str,
) -> str:
    """Persist the compaction record via the UoW's conversation compaction repo.

    Returns the resulting `compaction_id` (the new id on insert, or the existing
    one on conflict via ON CONFLICT DO NOTHING + SELECT fallback).
    """
    if not result.operations:
        raise ValueError("operations must be non-empty for a recordable compaction")

    summary_text = (result.summary_text or "")[:_SUMMARY_MAX_CHARS]
    summary_tokens_estimate = max(1, len(summary_text) // 4)  # rough char→token

    compaction_id = derive_compaction_id(
        session_id=session_id,
        messages_input_hash=messages_input_hash,
        tokens_before_total=result.tokens_before,
        tokens_after_total=result.tokens_after,
        messages_removed_total=result.messages_removed,
        summary=summary_text,
    )

    record = ConversationCompaction(
        id=uuid4(),
        compaction_id=compaction_id,
        session_id=session_id,
        summary=summary_text,
        summary_tokens=summary_tokens_estimate,
        first_visible_event_id=None,  # [R2-P2-6] always NULL in B6
        last_visible_event_id=None,
        pre_compact_checkpoint_id=None,  # B6 best-effort, populated only when threadable
        operations=copy.deepcopy(result.operations),  # [P3] deep copy so persisted op dicts can't be mutated by callers
        parent_compaction_id=None,
        # [CXR1-P1-1] CompactionResult.tokens_before / tokens_after / messages_removed
        # are ALREADY rolled up across the escalation chain by _hard_compact's
        # prior_operations branch. This is the source-of-truth for the *_total
        # columns; do not re-derive from operations[] here.
        tokens_before_total=result.tokens_before,
        tokens_after_total=result.tokens_after,
        messages_removed_total=result.messages_removed,
        created_at=datetime.now(timezone.utc),
    )

    persisted = await uow.compaction.create_or_get(record)
    return persisted.compaction_id
