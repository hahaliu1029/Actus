"""R5 CS4 ApprovalState adapter 单元测试（纯 mock，无 DB 依赖）。

覆盖：
- ``UowApprovalGrantQuery`` 正确把入参透传给 ``uow.approval_grants.find_active_grants``

PE-4d1：``SessionLegacyRuleQuery`` 适配器已删除（legacy tool_approval_rules
read path 退役）。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.approval_state_adapters import (
    UowApprovalGrantQuery,
)

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
