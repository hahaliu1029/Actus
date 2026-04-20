"""R5 CS4 ApprovalState adapter 单元测试（纯 mock，无 DB 依赖）。

覆盖：
- ``UowApprovalGrantQuery`` 正确把入参透传给 ``uow.approval_grants.find_active_grants``
- ``SessionLegacyRuleQuery`` always_deny 优先级 / always_allow / no_match 三条路径
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.approval_state_adapters import (
    SessionLegacyRuleQuery,
    UowApprovalGrantQuery,
)
from app.domain.models.tool_approval_rule import ToolApprovalRule

pytestmark = pytest.mark.anyio


def _make_uow_with_grants(return_value):
    uow = AsyncMock()
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)
    uow.approval_grants = AsyncMock()
    uow.approval_grants.find_active_grants = AsyncMock(return_value=return_value)
    factory = MagicMock(return_value=uow)
    return factory, uow


# ---------------- UowApprovalGrantQuery ----------------


async def test_grant_query_delegates_to_uow_with_named_kwargs() -> None:
    """find_active_grants 应 keyword-arg 透传到 uow.approval_grants，返值原样返回。"""
    sentinel: list = []
    factory, uow = _make_uow_with_grants(sentinel)
    q = UowApprovalGrantQuery(uow_factory=factory)

    out = await q.find_active_grants(
        user_id="u1",
        session_id="s1",
        tool_name="shell_execute",
    )

    assert out is sentinel  # 原样返
    uow.approval_grants.find_active_grants.assert_awaited_once_with(
        user_id="u1",
        session_id="s1",
        tool_name="shell_execute",
    )
    # 每次调用开 + 关一次 UoW
    uow.__aenter__.assert_awaited_once()
    uow.__aexit__.assert_awaited_once()


async def test_grant_query_allows_none_session_id() -> None:
    """session_id=None（always-only 查询）透传给 uow 不报错。"""
    factory, uow = _make_uow_with_grants([])
    q = UowApprovalGrantQuery(uow_factory=factory)

    out = await q.find_active_grants(
        user_id="u1", session_id=None, tool_name="shell_execute"
    )
    assert out == []
    uow.approval_grants.find_active_grants.assert_awaited_once_with(
        user_id="u1", session_id=None, tool_name="shell_execute"
    )


# ---------------- SessionLegacyRuleQuery ----------------


class _FakeSessionCtx:
    """async_sessionmaker() 返值的最小 async context manager mock。"""

    async def __aenter__(self):
        return MagicMock(name="session")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


def _fake_session_factory():
    return MagicMock(return_value=_FakeSessionCtx())


def _rule(
    *,
    rule: str,
    command_pattern: str = "ls *",
    dir_pattern: str = "",
    tool_name: str = "shell_execute",
) -> ToolApprovalRule:
    return ToolApprovalRule(
        user_id="u1",
        tool_name=tool_name,
        rule=rule,
        command_pattern=command_pattern,
        dir_pattern=dir_pattern,
    )


async def test_legacy_query_returns_no_match_when_no_rules(monkeypatch) -> None:
    """空规则集 → no_match。"""

    class _StubRepo:
        def __init__(self, _session):
            pass

        async def find_by_user_and_tool(self, user_id, tool_name):
            return []

    monkeypatch.setattr(
        "app.infrastructure.repositories.db_tool_approval_rule_repository."
        "DBToolApprovalRuleRepository",
        _StubRepo,
    )

    q = SessionLegacyRuleQuery(session_factory=_fake_session_factory())
    out = await q.check("u1", "shell_execute", "ls /", "")
    assert out == "no_match"


async def test_legacy_query_always_deny_wins_over_allow(monkeypatch) -> None:
    """同工具同时匹配 allow 和 deny → deny 优先（Priority 5a）。"""
    rules = [
        _rule(rule="always_allow", command_pattern="ls *"),
        _rule(rule="always_deny", command_pattern="ls *"),
    ]

    class _StubRepo:
        def __init__(self, _session):
            pass

        async def find_by_user_and_tool(self, user_id, tool_name):
            return rules

    monkeypatch.setattr(
        "app.infrastructure.repositories.db_tool_approval_rule_repository."
        "DBToolApprovalRuleRepository",
        _StubRepo,
    )

    q = SessionLegacyRuleQuery(session_factory=_fake_session_factory())
    out = await q.check("u1", "shell_execute", "ls /", "")
    assert out == "deny"


async def test_legacy_query_always_allow_when_only_allow_matches(monkeypatch) -> None:
    """仅 always_allow 命中 → allow。"""
    rules = [_rule(rule="always_allow", command_pattern="ls *")]

    class _StubRepo:
        def __init__(self, _session):
            pass

        async def find_by_user_and_tool(self, user_id, tool_name):
            return rules

    monkeypatch.setattr(
        "app.infrastructure.repositories.db_tool_approval_rule_repository."
        "DBToolApprovalRuleRepository",
        _StubRepo,
    )

    q = SessionLegacyRuleQuery(session_factory=_fake_session_factory())
    out = await q.check("u1", "shell_execute", "ls /", "")
    assert out == "allow"


async def test_legacy_query_dir_pattern_filters_non_matching(monkeypatch) -> None:
    """``dir_pattern`` 非空且 dir_arg 不匹配 → 规则不命中，返 no_match。"""
    rules = [
        _rule(
            rule="always_allow",
            command_pattern="rm *",
            dir_pattern="/tmp/*",
        )
    ]

    class _StubRepo:
        def __init__(self, _session):
            pass

        async def find_by_user_and_tool(self, user_id, tool_name):
            return rules

    monkeypatch.setattr(
        "app.infrastructure.repositories.db_tool_approval_rule_repository."
        "DBToolApprovalRuleRepository",
        _StubRepo,
    )

    q = SessionLegacyRuleQuery(session_factory=_fake_session_factory())
    out: Optional[str] = await q.check("u1", "shell_execute", "rm file.txt", "/etc/x")
    assert out == "no_match"
