"""B9 分层门：application 不 import interfaces；application/domain 不 import infrastructure.runtime_stats."""
import ast
from pathlib import Path

import pytest

API_ROOT = Path(__file__).resolve().parents[2] / "app"


def _imports_of(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_runtime_extension_service_no_interfaces_import():
    mods = _imports_of(API_ROOT / "application/services/runtime_extension_service.py")
    assert not any(m.startswith("app.interfaces") for m in mods)


def test_runtime_extension_service_no_network_imports():
    """INV-B9-4 结构门：GET 聚合器源码禁 import 任何网络/探测栈（httpx / mcp /
    a2a / extension_probe_service）——GET 纯读注入的 view 快照，绝不构造网络客户端。"""
    mods = _imports_of(API_ROOT / "application/services/runtime_extension_service.py")
    forbidden = ("httpx", "app.domain.services.tools.mcp", "app.domain.services.tools.a2a")
    for m in mods:
        assert not any(m == f or m.startswith(f + ".") for f in forbidden), m
    # extension_probe_service 仅允许 TYPE_CHECKING 下的类型注解（运行期不导入具体网络实现）；
    # ProbeRecord 类型钉子经字符串注解/TYPE_CHECKING 引用，源码顶层不得运行期 import。
    assert not any(
        m == "app.application.services.extension_probe_service" for m in mods
    ), "GET 聚合器不得运行期 import extension_probe_service（含 httpx/mcp 传递依赖）"


def test_probe_service_no_interfaces_import():
    path = API_ROOT / "application/services/extension_probe_service.py"
    if not path.exists():  # PR-1 时尚未创建；PR-2 起生效
        pytest.skip(
            "extension_probe_service.py 尚未落地（PR-2 Task 12）——落地后本门自动生效"
        )
    assert not any(m.startswith("app.interfaces") for m in _imports_of(path))


def test_no_runtime_stats_import_outside_infra():
    for base in ("application", "domain"):
        for py in (API_ROOT / base).rglob("*.py"):
            mods = _imports_of(py)
            assert not any(
                "infrastructure.external.runtime_stats" in m for m in mods
            ), str(py)
