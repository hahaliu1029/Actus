"""HTTP routes for conversation compactions.

Spec: docs/superpowers/specs/2026-05-02-b6-compaction-metadata-persistence-design.md § Section 3
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

from app.application.services.app_config_service import AppConfigService
from app.application.services.compaction_recovery_service import recover_original_messages
from app.application.services.manual_compaction_flag import set_manual_compact_pending
from app.domain.models.conversation_compaction import ConversationCompaction
from app.domain.models.session import SessionStatus
from app.domain.repositories.uow import IUnitOfWork
from app.infrastructure.storage.postgres import get_uow
from app.infrastructure.storage.redis import RedisClient, get_redis
from app.interfaces.dependencies import CurrentUser
from app.interfaces.dependencies.rate_limit import rate_limit_write
from app.interfaces.service_dependencies import get_app_config_service
from app.interfaces.schemas.conversation_compaction import (
    CompactionGoneResponse,
    CompactionListItem,
    CompactionOriginalContentResponse,
    ConversationCompactionDetailResponse,
    ConversationCompactionListResponse,
    RecoveredMessage,
)
from app.interfaces.schemas.recovered_message_sanitizer import sanitize_recovered_messages

router = APIRouter(prefix="/sessions/{session_id}/compactions", tags=["compaction"])


def _to_list_item(rec: ConversationCompaction) -> CompactionListItem:
    kinds: list[str] = []
    for op in rec.operations:
        kind = op.get("kind")
        if kind is None:
            # Operations without a "kind" indicate data corruption. Log + skip but
            # don't silently mask in the response.
            logger.warning(
                "compaction %s has operation without 'kind': %r",
                rec.compaction_id,
                op,
            )
            continue
        kinds.append(kind)
    return CompactionListItem(
        compaction_id=rec.compaction_id,
        kinds=kinds,
        summary_preview=rec.summary[:200],
        tokens_before_total=rec.tokens_before_total,
        tokens_after_total=rec.tokens_after_total,
        messages_removed_total=rec.messages_removed_total,
        first_visible_event_id=rec.first_visible_event_id,
        last_visible_event_id=rec.last_visible_event_id,
        has_recoverable_original=rec.pre_compact_checkpoint_id is not None,
        created_at=rec.created_at,
    )


async def _load_owned_session(uow: IUnitOfWork, session_id: str, current_user: CurrentUser):
    """Ownership check — match session_routes.py convention.

    Uses `uow.session.get_by_id()` (the real method on `DBSessionRepository`).
    """
    session = await uow.session.get_by_id(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session not found")
    if session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden")
    return session


_COMPACTABLE_STATUSES = frozenset({SessionStatus.COMPLETED, SessionStatus.TIMED_OUT})
_STATUS_REJECT_REASONS: dict[SessionStatus, str] = {
    SessionStatus.PENDING: "no_compactable_history",
    SessionStatus.RUNNING: "run_active",
    SessionStatus.FINISHING: "run_active",
    SessionStatus.WAITING: "waiting_resume_unsupported",
    SessionStatus.TAKEOVER: "takeover_active",
    SessionStatus.TAKEOVER_PENDING: "takeover_active",
}


def _compaction_request_decision(
    status: SessionStatus,
    *,
    manual_enabled: bool,
    overflow_guard_enabled: bool,
) -> str | None:
    """B11 §8: decide whether a manual /compact request is allowed.

    Returns None if allowed, else a machine-readable 409 reason. Priority:
    flag → overflow guard → session-status table. Only COMPLETED/TIMED_OUT
    are compactable (active runs self-compact at the water mark; WAITING resume
    doesn't consume the flag in v1; takeover ownership is ambiguous).
    """
    if not manual_enabled:
        return "manual_compaction_disabled"
    if not overflow_guard_enabled:
        return "overflow_guard_disabled"
    if status in _COMPACTABLE_STATUSES:
        return None
    return _STATUS_REJECT_REASONS.get(status, "status_not_compactable")


@router.get("", response_model=ConversationCompactionListResponse)
async def list_compactions(
    session_id: str,
    current_user: CurrentUser,
    uow: IUnitOfWork = Depends(get_uow),
) -> ConversationCompactionListResponse:
    async with uow:
        await _load_owned_session(uow, session_id, current_user)
        rows = await uow.compaction.list_for_session(session_id)
    return ConversationCompactionListResponse(items=[_to_list_item(r) for r in rows])


@router.post("", dependencies=[Depends(rate_limit_write)])
async def request_compaction(
    session_id: str,
    current_user: CurrentUser,
    uow: IUnitOfWork = Depends(get_uow),
    app_config_service: AppConfigService = Depends(get_app_config_service),
    redis_client: RedisClient = Depends(get_redis),  # DI so integration tests can override
) -> JSONResponse:
    """B11 §8: register a manual /compact request (pending-flag semantics).

    INV-B11-5: this handler performs ZERO graph-external checkpoint I/O — it only
    reads session status, reads flags, and SETs a Redis key. The compaction body
    happens later at the in-graph seam (PlannerReActFlow._run_forced_initial_compaction).
    """
    # Owner check + status/flag decision run BEFORE the manual pending-flag SET
    # (the rate_limit_write dependency may write its own Redis counter earlier —
    # that's the limiter, not our flag). We never SET manual_compact_pending on a
    # 403/404/409 path.
    async with uow:
        session = await _load_owned_session(uow, session_id, current_user)
        session_status = session.status

    agent_config = await app_config_service.get_agent_config()
    llm_config = await app_config_service.get_llm_config()
    reason = _compaction_request_decision(
        session_status,
        manual_enabled=agent_config.slash_commands.manual_compaction_enabled,
        overflow_guard_enabled=llm_config.context_overflow_guard_enabled,
    )
    if reason is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=reason)

    await set_manual_compact_pending(redis_client.client, session_id)
    return JSONResponse(
        status_code=status.HTTP_200_OK, content={"request_status": "queued"}
    )


@router.get("/{compaction_id}", response_model=ConversationCompactionDetailResponse)
async def get_compaction_detail(
    session_id: str,
    compaction_id: str,
    current_user: CurrentUser,
    uow: IUnitOfWork = Depends(get_uow),
) -> ConversationCompactionDetailResponse:
    async with uow:
        await _load_owned_session(uow, session_id, current_user)
        rec = await uow.compaction.get_by_id(session_id, compaction_id)
        if rec is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="compaction not found")
    return ConversationCompactionDetailResponse(
        compaction_id=rec.compaction_id,
        session_id=rec.session_id,
        summary=rec.summary,
        summary_tokens=rec.summary_tokens,
        operations=rec.operations,
        parent_compaction_id=rec.parent_compaction_id,
        first_visible_event_id=rec.first_visible_event_id,
        last_visible_event_id=rec.last_visible_event_id,
        pre_compact_checkpoint_id=rec.pre_compact_checkpoint_id,
        tokens_before_total=rec.tokens_before_total,
        tokens_after_total=rec.tokens_after_total,
        messages_removed_total=rec.messages_removed_total,
        created_at=rec.created_at,
    )


@router.get(
    "/{compaction_id}/original-content",
    response_model=None,  # Union[Pydantic, JSONResponse] is not a valid Pydantic field
    responses={
        200: {"model": CompactionOriginalContentResponse},
        410: {"model": CompactionGoneResponse},
    },
)
async def get_original_content(
    session_id: str,
    compaction_id: str,
    request: Request,
    current_user: CurrentUser,
    uow: IUnitOfWork = Depends(get_uow),
) -> CompactionOriginalContentResponse | JSONResponse:
    async with uow:
        await _load_owned_session(uow, session_id, current_user)
        rec = await uow.compaction.get_by_id(session_id, compaction_id)
        if rec is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="compaction not found")

    if rec.pre_compact_checkpoint_id is None:
        return _gone_response(compaction_id)

    pool = request.app.state.checkpointer_pool
    messages = await recover_original_messages(pool, session_id, rec.pre_compact_checkpoint_id)
    if messages is None:
        return _gone_response(compaction_id)

    raw = _serialize_messages(messages)
    sanitized = sanitize_recovered_messages(raw)
    return CompactionOriginalContentResponse(
        compaction_id=compaction_id,
        pre_compact_checkpoint_id=rec.pre_compact_checkpoint_id,
        recovered_messages=sanitized,
        recovered_at=datetime.now(timezone.utc),
    )


def _gone_response(compaction_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_410_GONE,
        content=CompactionGoneResponse(
            error="checkpointer_expired",
            message="压缩前的原始对话已超出 checkpointer 保留期，无法恢复",
            compaction_id=compaction_id,
            summary_still_available=True,
        ).model_dump(),
    )


def _serialize_messages(msgs) -> list[RecoveredMessage]:
    out = []
    for m in msgs:
        out.append(RecoveredMessage(
            type=type(m).__name__.replace("Message", "").lower(),
            content=m.content,
            tool_calls=getattr(m, "tool_calls", None),
            tool_call_id=getattr(m, "tool_call_id", None),
            name=getattr(m, "name", None),
            id=getattr(m, "id", None),
        ))
    return out
