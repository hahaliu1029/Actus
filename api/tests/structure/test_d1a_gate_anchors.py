"""D1-2 行为面的静态半：runner 源码在锚点区调用 gate helpers（AST 计数）。"""
from __future__ import annotations

from pathlib import Path

RUNNER = (Path(__file__).resolve().parent.parent.parent.parent
          / "api" / "app" / "domain" / "services" / "agent_task_runner.py")


def test_g1_wired_before_initialize():
    src = RUNNER.read_text()
    g1 = src.index("filter_mcp_config(")
    init = src.index("._mcp_tool.initialize(")
    assert g1 < init, "G1 必须在 mcp initialize 之前"


def test_g4_wraps_every_floor_product():
    src = RUNNER.read_text()
    floor_calls = src.count("_apply_member_skill_floor(")
    gate_calls = src.count("filter_skills(")
    # def 定义占 1 次 floor 计数；每个产出点 + session 池 1 处都要有 gate
    assert gate_calls >= (floor_calls - 1) + 1, (
        f"filter_skills 调用数 {gate_calls} < floor 产出点 {floor_calls - 1} + 池 1")
