"""C4.1a PR-2 — get_subagent_run_repository repo-or-None（flag OFF 纯 unit）（spec §5.0 + §7 PR-2）。"""
from __future__ import annotations

from core.config import Settings
from app.interfaces.service_dependencies import get_subagent_run_repository


def test_returns_none_when_flag_off() -> None:
    # flag OFF（default）→ 不构造 repo、不 touch DB（repo-or-None，INV-C4.1-1）。
    assert get_subagent_run_repository(Settings(env="test")) is None
