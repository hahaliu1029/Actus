"""spec R2#1: coordinator child runner must inherit the root's ToolRuntimeConfig —
otherwise B1 flags flip on root but child graphs silently stay default-OFF."""
from __future__ import annotations

import ast
import inspect


def test_every_agent_task_runner_construction_passes_tool_runtime():
    """AST 缝测试：service_dependencies 内所有 AgentTaskRunner 构造调用
    都必须显式传 tool_runtime=（根构造点 :507 已传；child builder 补齐后
    本断言对未来新增构造点同样生效）。直接构造 DI 图需要 app 生命周期——
    静态断言锁接线，agent_task_runner 侧默认回退（:539）是运行时防线。"""
    import app.interfaces.service_dependencies as deps

    tree = ast.parse(inspect.getsource(deps))
    ctor_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and (
            (isinstance(node.func, ast.Name) and node.func.id == "AgentTaskRunner")
            or (isinstance(node.func, ast.Attribute)
                and node.func.attr == "AgentTaskRunner")
        )
    ]
    assert ctor_calls, "no AgentTaskRunner constructions found — anchor drifted"
    missing = [
        call.lineno for call in ctor_calls
        if not any(kw.arg == "tool_runtime" for kw in call.keywords)
    ]
    assert not missing, (
        f"AgentTaskRunner constructed WITHOUT tool_runtime= at "
        f"service_dependencies.py lines {missing} (spec R2#1 — child/root "
        "B1 flag drift)"
    )
