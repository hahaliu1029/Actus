"""B9 runner liveness 接线：fail-open + finally 兜底（spec §4/§13）。

不起完整 runner——直接测 AgentTaskRunner 上新增的两个瘦 helper：
_liveness_acquire_connected() 与 _liveness_release_all()（实现须把接线逻辑收进
这两个方法，使其可独立单测；invoke()/_cleanup_tools 只调 helper）。
"""
from unittest.mock import MagicMock

import pytest

from app.domain.services.agent_task_runner import AgentTaskRunner


def _bare_runner():
    """绕过 ctor 的最小实例（对齐仓库既有 runner 单测的构造惯例；
    若已有 fixture/builder 直接复用）。"""
    runner = object.__new__(AgentTaskRunner)
    runner._session_id = "sess-1"
    runner._liveness_acquired = []
    runner._mcp_tool = MagicMock()
    runner._a2a_tool = MagicMock()
    return runner


def test_acquire_records_connected_ids(monkeypatch):
    import app.domain.services.agent_task_runner as runner_mod
    reg = MagicMock()
    monkeypatch.setattr(runner_mod, "runtime_liveness_registry", reg)
    runner = _bare_runner()
    runner._mcp_tool.connected_server_ids.return_value = ["srv-a"]
    runner._a2a_tool.connected_server_ids.return_value = ["a2a-1"]
    runner._liveness_acquire_connected()
    reg.acquire.assert_any_call("mcp", "srv-a", "sess-1")
    reg.acquire.assert_any_call("a2a", "a2a-1", "sess-1")
    assert ("mcp", "srv-a") in runner._liveness_acquired


def test_acquire_failure_marks_degraded_and_does_not_raise(monkeypatch):
    """R12#2：connected_server_ids 抛错 → mark_degraded，绝不冒泡。"""
    import app.domain.services.agent_task_runner as runner_mod
    reg = MagicMock()
    monkeypatch.setattr(runner_mod, "runtime_liveness_registry", reg)
    runner = _bare_runner()
    runner._mcp_tool.connected_server_ids.side_effect = RuntimeError("boom")
    runner._liveness_acquire_connected()          # 不抛
    reg.mark_degraded.assert_called_once_with("sess-1")


def test_release_all_always_ends_run(monkeypatch):
    """R14#1：release 抛错被吞后 end_run 仍无条件执行。"""
    import app.domain.services.agent_task_runner as runner_mod
    reg = MagicMock()
    reg.release.side_effect = RuntimeError("boom")
    monkeypatch.setattr(runner_mod, "runtime_liveness_registry", reg)
    runner = _bare_runner()
    runner._liveness_acquired = [("mcp", "srv-a")]
    runner._liveness_release_all()                # 不抛
    reg.end_run.assert_called_once_with("sess-1")
    assert runner._liveness_acquired == []


def test_invoke_calls_begin_run_before_first_try():
    """R14#1 全序（R2#8 加强）：begin_run 在 invoke() 帧内且先于首个 try 块——
    锁定"开始处"而非任意后置位置。"""
    import ast
    import inspect
    import textwrap

    from app.domain.services.agent_task_runner import AgentTaskRunner

    src = textwrap.dedent(inspect.getsource(AgentTaskRunner.invoke))
    # 剥掉 docstring，防止 docstring 中若含字面 "try:" 误判位置
    tree = ast.parse(src)
    func = tree.body[0]
    docstring = ast.get_docstring(func)
    assert "_liveness_begin_run" in src
    body_src = src
    if docstring is not None:
        body_src = src.replace(docstring, "", 1)
    assert "_liveness_begin_run" in body_src
    assert body_src.index("_liveness_begin_run") < body_src.index("try:")
