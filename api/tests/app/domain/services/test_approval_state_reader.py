"""R5 CS4 Reader 单元测试（mock ApprovalGrantQuery）。"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from app.domain.models.approval_grant import ApprovalGrant
from app.domain.services.approval_state_reader import ApprovalStateReader


def _grant(
    *,
    decision_id: str = "d",
    scope: str = "always",
    effect: str = "approve",
    session_id: str | None = None,
    arg_digest: str = "",
    primary_arg: str = "*",
    dir_arg: str = "",
    expires_at: datetime | None = None,
) -> ApprovalGrant:
    return ApprovalGrant(
        decision_id=decision_id,
        user_id="u1",
        session_id=session_id,
        tool_name="shell_execute",
        tool_source="native",
        arg_digest=arg_digest,
        primary_arg=primary_arg,
        dir_arg=dir_arg,
        scope=scope,
        effect=effect,
        source_type="user_click",
        confirmation_id=None,
        expires_at=expires_at,
    )


def _query(returns: list[ApprovalGrant]):
    q = AsyncMock()
    q.find_active_grants = AsyncMock(return_value=returns)
    return q


@pytest.mark.anyio
async def test_reader_priority_always_deny_wins() -> None:
    grants = [
        _grant(decision_id="deny", scope="always", effect="deny", primary_arg="rm *"),
        _grant(decision_id="allow", scope="always", effect="approve", primary_arg="rm *"),
    ]
    reader = ApprovalStateReader(query=_query(grants))
    result = await reader.check("u1", "s1", "shell_execute", "d1", "rm foo", None)
    assert result == "deny"


@pytest.mark.anyio
async def test_reader_priority_always_allow_before_session_allow() -> None:
    grants = [
        _grant(scope="always", effect="approve", primary_arg="*"),
        _grant(
            scope="session",
            effect="approve",
            session_id="s1",
            arg_digest="d1",
            primary_arg="*",
        ),
    ]
    reader = ApprovalStateReader(query=_query(grants))
    result = await reader.check("u1", "s1", "shell_execute", "d1", "anything", None)
    assert result == "allow"


@pytest.mark.anyio
async def test_reader_session_allow_exact_digest_match() -> None:
    grants = [
        _grant(
            scope="session",
            effect="approve",
            session_id="s1",
            arg_digest="d1",
            primary_arg="*",
        ),
    ]
    reader = ApprovalStateReader(query=_query(grants))
    # digest 对得上 → allow
    assert await reader.check("u1", "s1", "shell_execute", "d1", "x", None) == "allow"
    # digest 对不上 → no_match
    assert await reader.check("u1", "s1", "shell_execute", "d2", "x", None) == "no_match"


@pytest.mark.anyio
async def test_reader_ignores_session_deny_phase1() -> None:
    """Phase 1 Reader 不 surface session_deny，审计仍保留在 DB 由 writer 落地。"""
    grants = [
        _grant(
            scope="session",
            effect="deny",
            session_id="s1",
            arg_digest="d1",
            primary_arg="*",
        ),
    ]
    reader = ApprovalStateReader(query=_query(grants))
    assert await reader.check("u1", "s1", "shell_execute", "d1", "x", None) == "no_match"


def test_reader_no_longer_accepts_legacy_rule_query() -> None:
    """PE-4d1: the legacy tool_approval_rules fallback is retired — the
    ctor must reject any `legacy_rule_query` kwarg."""
    with pytest.raises(TypeError):
        ApprovalStateReader(query=_query([]), legacy_rule_query=object())  # type: ignore[call-arg]


@pytest.mark.anyio
async def test_reader_grants_miss_returns_no_match() -> None:
    """PE-4d1: grants miss → no_match directly (no legacy fallback branch)."""
    reader = ApprovalStateReader(query=_query([]))
    assert await reader.check("u1", "s1", "shell_execute", "d1", "x", None) == "no_match"


@pytest.mark.anyio
async def test_reader_always_allow_respects_primary_pattern_match() -> None:
    """always_allow 的 primary_arg pattern 不命中输入时，不 surface。"""
    grants = [_grant(scope="always", effect="approve", primary_arg="ls *")]
    reader = ApprovalStateReader(query=_query(grants))
    # pattern=ls * 不匹配 rm foo
    assert await reader.check("u1", "s1", "shell_execute", "d1", "rm foo", None) == "no_match"
    # pattern=ls * 匹配 ls /tmp
    assert await reader.check("u1", "s1", "shell_execute", "d1", "ls /tmp", None) == "allow"
