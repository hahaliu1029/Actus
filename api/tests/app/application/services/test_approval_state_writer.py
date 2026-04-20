"""R5 CS4 Writer 单元测试（mock UoW，无真实 DB 依赖）。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import IntegrityError

from app.application.services.approval_state_writer import ApprovalStateWriter
from app.domain.models.approval_grant import ApprovalDecision, ApprovalGrant


def _make_uow():
    """返回 (uow_factory, uow_mock) —— factory 每次调用返回同一个被 mock 过的 uow。"""
    uow = AsyncMock()
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)
    uow.approval_grants = AsyncMock()
    uow.tool_approval_log = AsyncMock()
    uow.rollback = AsyncMock()

    factory = MagicMock(return_value=uow)
    return factory, uow


def _session_decision(
    *,
    confirmation_id: str | None = "tool-call-abc",
    effect: str = "approve",
    source_type: str = "user_click",
) -> ApprovalDecision:
    return ApprovalDecision(
        user_id="u1",
        session_id="s1",
        tool_name="shell_execute",
        tool_source="native",
        arg_digest="d1",
        primary_arg="ls *",
        dir_arg="",
        scope="session",
        effect=effect,
        source_type=source_type,
        confirmation_id=confirmation_id,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        risk_level="medium",
    )


def _existing_grant(
    *,
    decision_id: str = "existing-id",
    scope: str = "session",
    effect: str = "approve",
) -> ApprovalGrant:
    return ApprovalGrant(
        decision_id=decision_id,
        user_id="u1",
        session_id="s1",
        tool_name="shell_execute",
        tool_source="native",
        arg_digest="d1",
        primary_arg="ls *",
        dir_arg="",
        scope=scope,
        effect=effect,
        source_type="user_click",
        confirmation_id="tool-call-abc",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
    )


@pytest.mark.anyio
async def test_writer_happy_path_new_row() -> None:
    """Happy path：新 confirmation_id → 返 (decision_id, True) 且 audit 也写入。"""
    factory, uow = _make_uow()
    uow.approval_grants.create.return_value = "new-decision-id"

    writer = ApprovalStateWriter(uow_factory=factory)
    decision = _session_decision()

    decision_id, newly_created = await writer.write(decision)

    assert decision_id == "new-decision-id"
    assert newly_created is True
    uow.approval_grants.create.assert_awaited_once_with(decision)
    uow.tool_approval_log.create.assert_awaited_once()
    # approved_by 在 user_click 情况下应为 "user"
    assert uow.tool_approval_log.create.await_args.kwargs["approved_by"] == "user"
    assert uow.tool_approval_log.create.await_args.kwargs["decision_id"] == "new-decision-id"


@pytest.mark.anyio
async def test_writer_idempotent_same_scope_effect() -> None:
    """同 confirmation_id + 同 (scope, effect) → 返 existing，newly_created=False。"""
    factory, uow = _make_uow()
    uow.approval_grants.create.side_effect = IntegrityError("stmt", {}, Exception("uniq"))
    uow.approval_grants.find_by_confirmation_id.return_value = _existing_grant(
        scope="session", effect="approve"
    )

    writer = ApprovalStateWriter(uow_factory=factory)
    decision = _session_decision(effect="approve")

    decision_id, newly_created = await writer.write(decision)

    assert decision_id == "existing-id"
    assert newly_created is False
    uow.rollback.assert_awaited_once()
    uow.approval_grants.find_by_confirmation_id.assert_awaited_once_with("tool-call-abc")


@pytest.mark.anyio
async def test_writer_conflict_different_scope_or_effect() -> None:
    """同 confirmation_id + 不同 effect → raise ValueError（语义冲突）。"""
    factory, uow = _make_uow()
    uow.approval_grants.create.side_effect = IntegrityError("stmt", {}, Exception("uniq"))
    uow.approval_grants.find_by_confirmation_id.return_value = _existing_grant(
        scope="session", effect="deny"
    )

    writer = ApprovalStateWriter(uow_factory=factory)
    decision = _session_decision(effect="approve")

    with pytest.raises(ValueError, match="claim collision"):
        await writer.write(decision)


@pytest.mark.anyio
async def test_writer_null_confirmation_id_hits_smartapprove_dedup() -> None:
    """confirmation_id=None（SmartApprove）走 find_smart_approve_dedup 分支，不 assert。"""
    factory, uow = _make_uow()
    uow.approval_grants.create.side_effect = IntegrityError("stmt", {}, Exception("partial-uniq"))
    uow.approval_grants.find_smart_approve_dedup.return_value = _existing_grant(
        decision_id="smart-existing", scope="session", effect="approve"
    )

    writer = ApprovalStateWriter(uow_factory=factory)
    decision = _session_decision(
        confirmation_id=None,
        effect="approve",
        source_type="smart_approve",
    )

    decision_id, newly_created = await writer.write(decision)

    assert decision_id == "smart-existing"
    assert newly_created is False
    uow.approval_grants.find_smart_approve_dedup.assert_awaited_once()
    # 从未触发 find_by_confirmation_id 分支（confirmation_id 为 None）
    uow.approval_grants.find_by_confirmation_id.assert_not_awaited()


@pytest.mark.anyio
async def test_writer_unexpected_integrity_error_reraises() -> None:
    """IntegrityError 但 find 查不到 existing → 重新 raise（意外情况不能吞）。"""
    factory, uow = _make_uow()
    uow.approval_grants.create.side_effect = IntegrityError("stmt", {}, Exception("uniq"))
    uow.approval_grants.find_by_confirmation_id.return_value = None

    writer = ApprovalStateWriter(uow_factory=factory)
    decision = _session_decision()

    with pytest.raises(IntegrityError):
        await writer.write(decision)


@pytest.mark.anyio
async def test_writer_delete_grant_cascades_audit() -> None:
    """delete_grant 同事务删 audit + grant 两张表的行，顺序：先 log 后 grant。"""
    factory, uow = _make_uow()
    call_order: list[str] = []
    uow.tool_approval_log.delete_by_decision_id = AsyncMock(
        side_effect=lambda did: call_order.append("log")
    )
    uow.approval_grants.delete = AsyncMock(
        side_effect=lambda did: call_order.append("grant")
    )

    writer = ApprovalStateWriter(uow_factory=factory)
    await writer.delete_grant("decision-x")

    assert call_order == ["log", "grant"]
    uow.tool_approval_log.delete_by_decision_id.assert_awaited_with("decision-x")
    uow.approval_grants.delete.assert_awaited_with("decision-x")


@pytest.mark.anyio
async def test_writer_smart_approve_approved_by_passthrough() -> None:
    """source_type='smart_approve' 时 audit 的 approved_by 保留为 'smart_approve'（非 'user'）。"""
    factory, uow = _make_uow()
    uow.approval_grants.create.return_value = "did-smart"

    writer = ApprovalStateWriter(uow_factory=factory)
    decision = _session_decision(source_type="smart_approve", confirmation_id=None)

    await writer.write(decision)

    assert uow.tool_approval_log.create.await_args.kwargs["approved_by"] == "smart_approve"
