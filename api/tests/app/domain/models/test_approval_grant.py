"""R5 CS4 domain model 不变式测试：ApprovalGrant + ApprovalDecision。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.domain.models.approval_grant import ApprovalDecision, ApprovalGrant


# -------- ApprovalDecision 不变式 --------


def _base_kwargs(**overrides):
    kwargs = dict(
        user_id="u1",
        session_id="s1",
        tool_name="shell_execute",
        tool_source="native",
        arg_digest="d1",
        primary_arg="ls *",
        dir_arg="",
        scope="session",
        effect="approve",
        source_type="user_click",
        confirmation_id="c1",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        risk_level="medium",
    )
    kwargs.update(overrides)
    return kwargs


def test_decision_session_scope_requires_session_id_and_expires_at() -> None:
    """session scope 必须同时带 session_id + expires_at。"""
    # 缺 session_id
    with pytest.raises(ValueError, match="session scope"):
        ApprovalDecision(**_base_kwargs(session_id=None))
    # 缺 expires_at
    with pytest.raises(ValueError, match="session scope"):
        ApprovalDecision(**_base_kwargs(expires_at=None))


def test_decision_always_scope_forbids_session_id_and_expires_at() -> None:
    """always scope 禁止 session_id 和 expires_at。"""
    with pytest.raises(ValueError, match="always scope"):
        ApprovalDecision(**_base_kwargs(scope="always", session_id="s1", expires_at=None))
    with pytest.raises(ValueError, match="always scope"):
        ApprovalDecision(
            **_base_kwargs(
                scope="always",
                session_id=None,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            )
        )


def test_decision_always_scope_valid_shape() -> None:
    d = ApprovalDecision(**_base_kwargs(scope="always", session_id=None, expires_at=None))
    assert d.scope == "always"
    assert d.session_id is None
    assert d.expires_at is None


# -------- ApprovalGrant.matches --------


def _grant(*, primary: str = "*", dir_pat: str = "") -> ApprovalGrant:
    return ApprovalGrant(
        decision_id="d",
        user_id="u1",
        session_id=None,
        tool_name="shell_execute",
        tool_source="native",
        arg_digest="",
        primary_arg=primary,
        dir_arg=dir_pat,
        scope="always",
        effect="approve",
        source_type="user_click",
        confirmation_id=None,
        expires_at=None,
    )


def test_grant_matches_empty_dir_pattern_not_constrained() -> None:
    g = _grant(primary="ls *", dir_pat="")
    assert g.matches("ls /tmp", "/tmp") is True
    assert g.matches("ls /tmp", None) is True


def test_grant_matches_nonempty_dir_pattern_requires_dir_arg() -> None:
    g = _grant(primary="ls *", dir_pat="/tmp/*")
    # dir_arg 命中
    assert g.matches("ls /tmp/abc", "/tmp/abc") is True
    # dir_arg=None 直接拒
    assert g.matches("ls /tmp/abc", None) is False
    # dir_arg 不命中
    assert g.matches("ls /tmp/abc", "/etc") is False


def test_grant_matches_primary_must_fnmatch() -> None:
    g = _grant(primary="ls *", dir_pat="")
    assert g.matches("rm foo", "/tmp") is False
