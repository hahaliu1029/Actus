"""HTTP response schemas for conversation_compaction endpoints.

Spec: docs/superpowers/specs/2026-05-02-b6-compaction-metadata-persistence-design.md § Section 3
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


CompactionKind = Literal["llm_summary", "hard_truncate"]


class CompactionListItem(BaseModel):
    compaction_id: str
    kinds: list[CompactionKind]  # [R4-P2-4] strict union
    summary_preview: str
    tokens_before_total: int
    tokens_after_total: int
    messages_removed_total: int
    first_visible_event_id: str | None
    last_visible_event_id: str | None
    has_recoverable_original: bool
    created_at: datetime


class ConversationCompactionListResponse(BaseModel):
    items: list[CompactionListItem]


class CompactionOperation(BaseModel):
    kind: CompactionKind
    tokens_before: int
    tokens_after: int
    messages_removed: int | None = None
    messages_kept: int | None = None
    summary_chars: int | None = None
    identifiers_preserved_count: int | None = None
    messages_summarized: int | None = None


class ConversationCompactionDetailResponse(BaseModel):
    compaction_id: str
    session_id: str
    summary: str
    summary_tokens: int
    operations: list[CompactionOperation]
    parent_compaction_id: str | None
    first_visible_event_id: str | None
    last_visible_event_id: str | None
    pre_compact_checkpoint_id: str | None
    tokens_before_total: int
    tokens_after_total: int
    messages_removed_total: int
    created_at: datetime


class RecoveredMessage(BaseModel):
    type: Literal["human", "ai", "tool", "system"]
    content: str | list[dict]
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    id: str | None = None


class CompactionOriginalContentResponse(BaseModel):
    compaction_id: str
    pre_compact_checkpoint_id: str
    recovered_messages: list[RecoveredMessage]
    recovered_at: datetime


class CompactionGoneResponse(BaseModel):
    error: Literal["checkpointer_expired"]
    message: str
    compaction_id: str
    summary_still_available: bool
