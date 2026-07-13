"""INV-D1-3：admission 实现禁触 WritePort/pin 写；WritePort 变更方法调用点白名单。"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
API_APP = REPO_ROOT / "api" / "app"
ADMISSION = API_APP / "infrastructure" / "external" / "governance" / "db_extension_admission.py"
REGISTRY_IMPL = API_APP / "infrastructure" / "external" / "governance" / "db_extension_registry.py"

# WritePort 变更方法（T4 合同）——调用点只允许出现在白名单模块
MUTATING_METHODS = {
    "record_install", "record_delete", "record_reconciled_seen",
    "mark_source_missing", "mark_source_restored", "record_reconciled_missing",
    "reset_pins_after_config_drift", "quarantine", "reapprove",
    "set_governance_enabled", "approve_pin",
}
# 允许调用 WritePort 变更方法的模块（相对 api/app；后续任务落地这些文件——
# 文件不存在=天然合规；新增调用点必须先进本白名单并说明归属）
ALLOWED_CALLERS = {
    "application/services/extension_reconciler.py",       # T15/T17
    "application/services/extension_governance_service.py",  # T20
    "application/services/extension_install_service.py",  # T18/T19（mcp/a2a 管道）
    "application/services/plugin_install_service.py",     # T21-T23（saga）
    "application/services/skill_service.py",               # T16（skill hooks）
    "infrastructure/external/governance/db_extension_registry.py",  # 实现自身
    # T20：治理路由调用的是 ExtensionGovernanceService facade（同名 quarantine/reapprove
    # 方法与 WritePort 撞名，AST gate 按方法名匹配无法区分接收者）——非 WritePort 直触，
    # 名归属显式登记（gate 注释「新增调用点必须先进本白名单并说明归属」）
    "interfaces/endpoints/extension_governance_routes.py",  # T20（service facade，非 WritePort）
}


def _calls_in(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def test_admission_module_never_touches_write_port():
    tree = ast.parse(ADMISSION.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "db_extension_registry" not in node.module, "admission imports WritePort impl"
    bad = _calls_in(ADMISSION) & MUTATING_METHODS
    assert not bad, f"admission 实现调用了治理写方法: {bad}"


def test_write_method_callers_whitelisted():
    violations: list[str] = []
    for py in API_APP.rglob("*.py"):
        rel = py.relative_to(API_APP).as_posix()
        if rel in ALLOWED_CALLERS:
            continue
        called = _calls_in(py) & MUTATING_METHODS
        if called:
            violations.append(f"{rel}: {sorted(called)}")
    assert not violations, "WritePort 变更方法出现在白名单外:\n" + "\n".join(violations)
