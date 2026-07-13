"""INV-D1-2/D1-0：注入拓扑结构门（AST/签名扫描，无需起 app）。"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MAIN = REPO_ROOT / "api" / "app" / "main.py"


def _kwargs_of_calls(source: str, func_name: str) -> list[set[str]]:
    tree = ast.parse(source)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", getattr(node.func, "attr", None))
            if name == func_name:
                out.append({kw.arg for kw in node.keywords if kw.arg})
    return out


SERVICE_DEPS = REPO_ROOT / "api" / "app" / "interfaces" / "service_dependencies.py"


def test_admission_port_wired_through_real_composition_roots():
    # R2#F12：main.py 不直接构造 AgentService——真实链=main `_build_agent_service(...)` →
    # service_dependencies.py:1892 `AgentService(...)`。两跳都断言。
    main_src = MAIN.read_text()
    build_calls = _kwargs_of_calls(main_src, "_build_agent_service")
    assert any("extension_admission_port" in kws for kws in build_calls), \
        "main.py _build_agent_service(...) 未透传 extension_admission_port"
    deps_src = SERVICE_DEPS.read_text()
    agent_service_calls = _kwargs_of_calls(deps_src, "AgentService")
    assert any("extension_admission_port" in kws for kws in agent_service_calls), \
        "service_dependencies.py AgentService(...) 未传 extension_admission_port"
    child_deps_calls = _kwargs_of_calls(main_src, "ChildRunnerSharedDeps")
    assert any("extension_admission_port" in kws for kws in child_deps_calls), \
        "main.py ChildRunnerSharedDeps(...) 未填 extension_admission_port（root/child 必须同源实例）"


def test_child_deps_dataclass_has_field_default_none():
    from app.interfaces.service_dependencies import ChildRunnerSharedDeps
    field = ChildRunnerSharedDeps.__dataclass_fields__["extension_admission_port"]
    assert field.default is None   # 向后兼容（B12 模式）


def test_runner_and_tools_accept_port_default_none():
    from app.domain.services.agent_task_runner import AgentTaskRunner
    from app.domain.services.tools.skill import SkillTool
    from app.domain.services.tools.skill_bundle_sync import SkillBundleSyncManager
    for cls in (AgentTaskRunner, SkillTool, SkillBundleSyncManager):
        sig = inspect.signature(cls.__init__)
        assert "extension_admission_port" in sig.parameters or "admission_port" in sig.parameters, cls
        p = sig.parameters.get("extension_admission_port") or sig.parameters["admission_port"]
        assert p.default is None, cls


def test_runner_module_never_imports_registry_impl():
    # INV-D1-2 结构：runner/SkillTool/sync manager 仅可 import admission protocol
    for rel in ("domain/services/agent_task_runner.py",
                "domain/services/tools/skill.py",
                "domain/services/tools/skill_bundle_sync.py",
                "domain/services/extension_admission_gates.py"):
        source = (REPO_ROOT / "api" / "app" / rel).read_text()
        assert "db_extension_registry" not in source, rel
        assert "db_extension_admission" not in source, rel
