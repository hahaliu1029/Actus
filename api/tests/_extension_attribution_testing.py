"""Test-only reset helper for the B9 extension attribution registry.

与 tests/_tool_source_testing.py 同模式：刻意放在 api/tests/ 下，
prod import path 无法触达。若 prod 路径误调此函数，进程内所有已注册的
MCP/Skill 工具归因会中途蒸发，统计热路径就无法把调用归因回扩展。
"""
from __future__ import annotations

from app.domain.services.tools.extension_attribution import (
    _PER_EXTENSION_CAP_WARNED,
    _PER_EXTENSION_COUNTS,
    _REGISTRY,
)


def reset_extension_attribution() -> None:
    """清空归因注册表及其上限计数/告警簿记。

    从测试 fixture（推荐 autouse）调用，保证每个测试从空归因表开始：上一个测试注册
    的 mcp_/skill_ 条目不会污染下一个。
    """
    _REGISTRY.clear()
    _PER_EXTENSION_COUNTS.clear()
    _PER_EXTENSION_CAP_WARNED.clear()
