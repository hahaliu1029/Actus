"""B10 §7.5c — _translate_outcome 生产调用守卫 (R3#1/R10#5).

AST 扫描: react_graph 内每个 `await _translate_outcome(` 生产调用必须显式
携带 display_metadata_enabled= 实参, 防未来新增调用点静默丢元数据.
镜像 B5 executor AST gate 风格 (test_executor_no_skill_context_writeback.py).
"""
from __future__ import annotations

import ast
from pathlib import Path

import app.domain.services.graphs.react_graph as react_graph_module


def test_every_translate_outcome_call_passes_display_flag():
    src = Path(react_graph_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_translate_outcome"
    ]
    # 生产调用点 3 个 (R3#1): _translate_to_result / legacy sibling /
    # _legacy_interrupt_helper_resume deny 路径
    assert len(calls) >= 3, f"expected >=3 call sites, found {len(calls)}"
    for call in calls:
        kw_names = {kw.arg for kw in call.keywords}
        assert "display_metadata_enabled" in kw_names, (
            f"line {call.lineno}: _translate_outcome call missing explicit "
            "display_metadata_enabled= kwarg (B10 guard, spec §7.5c)"
        )
