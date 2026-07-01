"""C4.1a PR-1 — subagent_run_record_enabled flag 测试（spec §5.0 + §7 PR-1）。"""
from __future__ import annotations

from core.config import Settings


def test_flag_defaults_false() -> None:
    # Default-OFF dark-launch：裸 test Settings flag OFF。
    assert Settings(env="test").subagent_run_record_enabled is False


def test_flag_parses_via_field_name() -> None:
    s = Settings(env="test", subagent_run_record_enabled=True)
    assert s.subagent_run_record_enabled is True


def test_flag_parses_via_screaming_alias() -> None:
    s = Settings(env="test", SUBAGENT_RUN_RECORD_ENABLED=True)
    assert s.subagent_run_record_enabled is True


def test_flag_parses_via_actus_alias() -> None:
    s = Settings(env="test", ACTUS_C4_SUBAGENT_RUN_RECORD_ENABLED=True)
    assert s.subagent_run_record_enabled is True
