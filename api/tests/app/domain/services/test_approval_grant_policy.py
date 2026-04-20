"""R5 CS4 policy helper 单元测试（纯函数、无 I/O）。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.domain.services.approval_grant_policy import (
    SESSION_GRANT_TTL,
    match_command_and_dir,
    session_grant_expires_at,
)


def test_session_grant_expires_at_24h_utc() -> None:
    """过期时间 = now(UTC) + 24h。"""
    base = datetime(2026, 4, 20, 9, 30, tzinfo=timezone.utc)
    out = session_grant_expires_at(now=base)
    assert out - base == SESSION_GRANT_TTL
    assert SESSION_GRANT_TTL == timedelta(hours=24)
    # 默认路径（now=None）返回的是现在 +24h（tolerance 数秒）
    out_default = session_grant_expires_at()
    diff = out_default - datetime.now(timezone.utc)
    assert timedelta(hours=23, minutes=59) <= diff <= timedelta(hours=24, minutes=1)


@pytest.mark.parametrize(
    "primary_arg,dir_arg,pattern,dir_pattern,expected",
    [
        # primary 不命中
        ("rm foo", "/tmp", "ls *", "", False),
        # dir_pattern 空串不约束
        ("ls foo", "/tmp", "ls *", "", True),
        ("ls foo", None, "ls *", "", True),
        # dir_pattern 非空 + dir_arg 命中
        ("ls foo", "/tmp/abc", "ls *", "/tmp/*", True),
        # dir_pattern 非空 + dir_arg=None → 拒
        ("ls foo", None, "ls *", "/tmp/*", False),
        # dir_pattern 非空 + dir_arg 不命中
        ("ls foo", "/etc", "ls *", "/tmp/*", False),
    ],
)
def test_match_command_and_dir_table(
    primary_arg: str,
    dir_arg: str | None,
    pattern: str,
    dir_pattern: str,
    expected: bool,
) -> None:
    assert match_command_and_dir(primary_arg, dir_arg, pattern, dir_pattern) is expected
