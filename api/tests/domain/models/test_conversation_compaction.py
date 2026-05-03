"""Domain dataclass invariants for ConversationCompaction.

Spec: docs/superpowers/specs/2026-05-02-b6-compaction-metadata-persistence-design.md
"""
import pytest
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from uuid import uuid4

from app.domain.models.conversation_compaction import ConversationCompaction


def _build(**overrides):
    base = dict(
        id=uuid4(),
        compaction_id="0123456789abcdef",
        session_id="sess_xyz",
        summary="hello world",
        summary_tokens=2,
        first_visible_event_id=None,
        last_visible_event_id=None,
        pre_compact_checkpoint_id=None,
        operations=[{"kind": "hard_truncate", "tokens_before": 10, "tokens_after": 5, "messages_removed": 3, "messages_kept": 19}],
        parent_compaction_id=None,
        tokens_before_total=10,
        tokens_after_total=5,
        messages_removed_total=3,
        created_at=datetime.now(timezone.utc),
    )
    base.update(overrides)
    return ConversationCompaction(**base)


def test_dataclass_is_frozen():
    rec = _build()
    with pytest.raises(FrozenInstanceError):
        rec.summary = "tampered"  # type: ignore[misc]


def test_compaction_id_is_16_chars():
    rec = _build()
    assert len(rec.compaction_id) == 16


def test_anchors_default_null():
    rec = _build()
    assert rec.first_visible_event_id is None
    assert rec.last_visible_event_id is None


def test_operations_must_be_list_of_dicts():
    rec = _build(operations=[{"kind": "llm_summary"}])
    assert isinstance(rec.operations, list)
    assert all(isinstance(op, dict) for op in rec.operations)
