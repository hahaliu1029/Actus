"""Domain model for a persisted conversation compaction event.

Spec: docs/superpowers/specs/2026-05-02-b6-compaction-metadata-persistence-design.md

Pure domain — no SQLAlchemy / FastAPI imports.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID


@dataclass(frozen=True)
class ConversationCompaction:
    id: UUID
    compaction_id: str
    session_id: str
    summary: str
    summary_tokens: int
    first_visible_event_id: str | None
    last_visible_event_id: str | None
    pre_compact_checkpoint_id: str | None
    operations: list[dict[str, Any]]
    parent_compaction_id: str | None
    tokens_before_total: int
    tokens_after_total: int
    messages_removed_total: int
    created_at: datetime
