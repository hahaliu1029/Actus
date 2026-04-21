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


# ---------------------------------------------------------------
# CS4 2026-04-21: write_audit_only — once scope 单一 audit 入口
# ---------------------------------------------------------------


@pytest.mark.anyio
async def test_write_audit_only_writes_audit_row() -> None:
    """once scope 路径：write_audit_only 透传所有入参到 tool_approval_log.create。"""
    factory, uow = _make_uow()
    writer = ApprovalStateWriter(uow_factory=factory)

    await writer.write_audit_only(
        user_id="alice",
        session_id="s1",
        tool_name="shell_execute",
        tool_args={"command": "ls /"},
        risk_level="medium",
        action="approve",
        scope="once",
        approved_by="user",
    )

    uow.tool_approval_log.create.assert_awaited_once()
    kwargs = uow.tool_approval_log.create.await_args.kwargs
    assert kwargs["user_id"] == "alice"
    assert kwargs["session_id"] == "s1"
    assert kwargs["tool_name"] == "shell_execute"
    assert kwargs["tool_args"] == {"command": "ls /"}
    assert kwargs["risk_level"] == "medium"
    assert kwargs["action"] == "approve"
    assert kwargs["scope"] == "once"
    assert kwargs["approved_by"] == "user"


@pytest.mark.anyio
async def test_write_audit_only_does_not_touch_grants() -> None:
    """once scope 不建 grant：approval_grants repo 的所有方法都不能被调。

    合同面锁死 —— 确保未来 refactor 不会意外把 grant 写入塞进 audit-only 路径。
    """
    factory, uow = _make_uow()
    writer = ApprovalStateWriter(uow_factory=factory)

    await writer.write_audit_only(
        user_id="u1",
        session_id="s1",
        tool_name="shell_execute",
        tool_args={},
        risk_level="low",
        action="deny",
        scope="once",
    )

    uow.approval_grants.create.assert_not_awaited()
    uow.approval_grants.find_by_confirmation_id.assert_not_awaited()
    uow.approval_grants.find_smart_approve_dedup.assert_not_awaited()
    uow.approval_grants.delete.assert_not_awaited()


@pytest.mark.anyio
async def test_write_audit_only_default_approved_by_is_user() -> None:
    """approved_by 默认 'user'（once scope 只由用户点击触发，不会被 SmartApprove 调）。"""
    factory, uow = _make_uow()
    writer = ApprovalStateWriter(uow_factory=factory)

    await writer.write_audit_only(
        user_id="u1",
        session_id="s1",
        tool_name="shell_execute",
        tool_args={},
        risk_level="low",
        action="approve",
        scope="once",
    )

    assert uow.tool_approval_log.create.await_args.kwargs["approved_by"] == "user"


@pytest.mark.anyio
async def test_writer_has_no_delete_audit_only_method() -> None:
    """合同面锁死：write_audit_only 不提供镜像 delete——once audit 是用户决策事实，
    kickoff failure 不回滚。参见 write_audit_only docstring。"""
    writer = ApprovalStateWriter(uow_factory=lambda: None)  # factory 不被调
    assert not hasattr(writer, "delete_audit_only"), (
        "write_audit_only 不应有对应的 delete_audit_only 方法——once audit 是"
        "用户决策事实记录，不是可回滚的 claim 产物。加镜像 delete 会误导维护者。"
    )


@pytest.mark.anyio
@pytest.mark.parametrize("bad_scope", ["session", "always", "", "PERSISTENT", "Once"])
async def test_write_audit_only_rejects_non_once_scope(bad_scope: str) -> None:
    """合同面锁死：write_audit_only 运行时拒绝非 ``once`` scope。

    Why：若允许透传任意 scope，caller 误传 ``session`` / ``always`` 就会绕过
    ``write()`` 的 UNIQUE(confirmation_id) 原子 claim，写出没有 grant 的
    orphan audit 行，破坏 "persistent scope 必有 grant" 的 CS4 合同。AST
    Rule 2 也抓不到（audit 在 writer 内部合法），只能靠运行时守卫。
    """
    factory, uow = _make_uow()
    writer = ApprovalStateWriter(uow_factory=factory)

    with pytest.raises(ValueError, match=r"scope='once'"):
        await writer.write_audit_only(
            user_id="u1",
            session_id="s1",
            tool_name="shell_execute",
            tool_args={},
            risk_level="low",
            action="approve",
            scope=bad_scope,
        )
    # 关键：拒绝发生在 UoW 打开前，底层 repo 不应被触到
    uow.tool_approval_log.create.assert_not_awaited()
