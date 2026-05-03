"""Repository interface for ConversationCompaction persistence.

Spec: docs/superpowers/specs/2026-05-02-b6-compaction-metadata-persistence-design.md

ABC stays in domain layer — no SQLAlchemy imports.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.models.conversation_compaction import ConversationCompaction


class ConversationCompactionRepository(ABC):
    """Persistence contract for ConversationCompaction."""

    @abstractmethod
    async def create_or_get(self, record: ConversationCompaction) -> ConversationCompaction:
        """INSERT ... ON CONFLICT (compaction_id) DO NOTHING RETURNING ...

        Returns the inserted record on success, or the existing record on conflict.
        Idempotent — safe to call multiple times with the same `compaction_id`.
        """
        ...

    @abstractmethod
    async def list_for_session(self, session_id: str) -> list[ConversationCompaction]:
        """Return all compactions for a session, sorted by created_at DESC."""
        ...

    @abstractmethod
    async def get_by_id(
        self,
        session_id: str,
        compaction_id: str,
    ) -> ConversationCompaction | None:
        """Return the compaction matching (session_id, compaction_id), or None."""
        ...
