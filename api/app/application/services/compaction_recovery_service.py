"""Best-effort recovery of pre-compaction messages via the LangGraph checkpointer.

Spec: docs/superpowers/specs/2026-05-02-b6-compaction-metadata-persistence-design.md § Section 3

Uses the existing `CheckpointerPool` (api/app/infrastructure/checkpointer_pool.py)
to construct an `AsyncPostgresSaver` and call its `aget_tuple(config)` API.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.infrastructure.checkpointer_pool import CheckpointerPool

logger = logging.getLogger(__name__)


async def recover_original_messages(
    pool: CheckpointerPool,
    session_id: str,
    checkpoint_id: str,
) -> list[BaseMessage] | None:
    """Returns the messages list from the named checkpoint, or None if unavailable.

    None means "best-effort failed" — caller should respond with 410 Gone.
    Reasons it can return None: checkpoint GC'd by TTL, pool error, schema drift.
    """
    try:
        checkpointer = AsyncPostgresSaver(pool.pool)
        config: dict[str, Any] = {
            "configurable": {
                "thread_id": session_id,
                "checkpoint_id": checkpoint_id,
            }
        }
        tuple_ = await checkpointer.aget_tuple(config)
        if tuple_ is None:
            return None
        checkpoint = tuple_.checkpoint
        channel_values = checkpoint.get("channel_values") if isinstance(checkpoint, dict) else None
        if not channel_values:
            return None
        messages = channel_values.get("messages")
        if not messages:
            return None
        return list(messages)
    except Exception as exc:
        logger.warning(
            "compaction recovery failed for session=%s checkpoint=%s: %s",
            session_id, checkpoint_id, exc,
        )
        return None
