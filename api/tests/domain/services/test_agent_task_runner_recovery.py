"""T19d / T19m — Runner forwards profile; step graph uses Flow._llm.

Uses inspect.getsource on the imported module so the test works in any
worktree / CI checkout (no hardcoded absolute paths).
"""
import ast
import inspect


def _runner_source() -> str:
    from app.domain.services import agent_task_runner
    return inspect.getsource(agent_task_runner)


def test_T19d_runner_forwards_profile_to_flow():
    source = _runner_source()
    # The constructor call should pass profile=self.profile explicitly.
    assert "profile=self.profile" in source, (
        "AgentTaskRunner must pass profile=self.profile when constructing "
        "PlannerReActFlow (spec §4.5)"
    )


def test_T19m_step_react_graph_uses_flow_wrapped_llm():
    source = _runner_source()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.AsyncFunctionDef)
            and node.name == "_build_step_react_graph"
        ):
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            func = sub.func
            if isinstance(func, ast.Name) and func.id == "build_react_graph":
                llm_arg = next(
                    (kw for kw in sub.keywords if kw.arg == "llm"), None,
                )
                assert llm_arg is not None, "build_react_graph must receive llm="
                val = ast.unparse(llm_arg.value)
                assert "self._flow._llm" in val, (
                    f"_build_step_react_graph must use self._flow._llm "
                    f"(Recovery-wrapped), got {val!r}"
                )
                return
    raise AssertionError("_build_step_react_graph not found in agent_task_runner.py")
