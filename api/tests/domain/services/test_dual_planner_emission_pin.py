"""C7 PR2 — F11 pin：plan lifecycle 双路径 / step lifecycle 单路径的静态锚。

main_graph 发 PlanEvent+StepEvent；planner_react 只发 PlanEvent。两处 plan 源
都经咽喉天然获得 lifecycle 覆盖（spec §4.1）；若未来 planner_react 开始发
StepEvent（或 main_graph 停发），本 pin 变红提示重审 §4.2 单路径假设。
"""
from pathlib import Path

# parents[3] == api/（services→domain→tests→api）；指向真源码树 api/app/domain/services。
# 注：brief verbatim 给的 parents[2] 会落到 api/tests/app/... 这一不存在的路径（off-by-one），
# 已按内容锚定修正——断言与 F11 事实均未改（deviation logged in task-8-report）。
GRAPHS = Path(__file__).resolve().parents[3] / "app" / "domain" / "services"


def test_main_graph_emits_plan_and_step_events():
    src = (GRAPHS / "graphs" / "main_graph.py").read_text(encoding="utf-8")
    assert "PlanEvent(" in src
    assert "StepEvent(" in src


def test_planner_react_emits_plan_but_never_step():
    src = (GRAPHS / "flows" / "planner_react.py").read_text(encoding="utf-8")
    assert "PlanEvent(" in src
    assert "StepEvent" not in src   # import 都不该有（F11/R1#6 grep 实证）
