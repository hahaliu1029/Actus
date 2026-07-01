"""C4.1a PR-2 — migration graph 检查（no-DB；spec §4 + §7 PR-2）。

只读 alembic 脚本图，不连 DB：验证新 migration 是唯一 head 且正确锚定
s3pr1。实际 up/down + 表结构由集成测试（CI）覆盖。
"""
from __future__ import annotations

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

_API_ROOT = Path(__file__).resolve().parents[2]  # api/


def _script() -> ScriptDirectory:
    cfg = Config(str(_API_ROOT / "alembic.ini"))
    return ScriptDirectory.from_config(cfg)


def test_c41a_is_single_head() -> None:
    heads = _script().get_heads()
    assert heads == ["c41a_add_subagent_runs"], heads


def test_c41a_down_revision_anchors_s3pr1() -> None:
    rev = _script().get_revision("c41a_add_subagent_runs")
    assert rev.down_revision == "s3pr1_add_session_depth_lineage"
