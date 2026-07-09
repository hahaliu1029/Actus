"""INV-C7-2 — LifecycleEvent 只能经 build_lifecycle_event 构造（spec §9）。

AST 扫描 api/app/ 全部生产代码：
- 直接调用 LifecycleEvent(...)（含 import alias 与 from-import 别名）
- LifecycleEvent.model_construct / model_validate / model_validate_json / model_copy
- class X(LifecycleEvent) 子类化
豁免清单见 EXEMPT_FILES。测试代码（tests/）天然不在扫描范围。
"""
import ast
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2] / "app"

EXEMPT_FILES = {
    # helper 自身（唯一合法构造点）
    APP_ROOT / "domain" / "services" / "lifecycle_emit.py",
    # 类定义所在地（定义不是构造；此文件内也禁止实例化，但豁免以防未来 default 工厂误报）
    APP_ROOT / "domain" / "models" / "event.py",
}
EXEMPT_DIRS = {
    # Redis recovery 经 TypeAdapter(Event) 反序列化重建——合法路径（防御性豁免）
    APP_ROOT / "infrastructure" / "external" / "event_recovery",
}

FORBIDDEN_METHODS = {"model_construct", "model_validate", "model_validate_json", "model_copy"}


def _lifecycle_aliases(tree: ast.AST) -> set[str]:
    """收集该文件中指向 LifecycleEvent 的全部名字（含 alias）。"""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "LifecycleEvent":
                    names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            # import app.domain.models.event [as m] → 属性访问 m.LifecycleEvent 由下方 Attribute 分支覆盖
            pass
    return names


def _is_lifecycle_ref(node: ast.expr, aliases: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in aliases
    if isinstance(node, ast.Attribute):
        return node.attr == "LifecycleEvent"  # m.LifecycleEvent / event.LifecycleEvent
    return False


def _scan_file(path: Path) -> list[str]:
    violations: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases = _lifecycle_aliases(tree)
    for node in ast.walk(tree):
        # 1) 直接构造 LifecycleEvent(...) / alias(...) / m.LifecycleEvent(...)
        if isinstance(node, ast.Call) and _is_lifecycle_ref(node.func, aliases):
            violations.append(f"{path}:{node.lineno} direct LifecycleEvent(...) construction")
        # 2) 逃逸口方法：LifecycleEvent.model_construct(...) 等
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in FORBIDDEN_METHODS
            and _is_lifecycle_ref(node.func.value, aliases | {"LifecycleEvent"})
        ):
            violations.append(f"{path}:{node.lineno} LifecycleEvent.{node.func.attr}(...) escape hatch")
        # 3) 子类化 class X(LifecycleEvent)
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if _is_lifecycle_ref(base, aliases):
                    violations.append(f"{path}:{node.lineno} subclassing LifecycleEvent")
    return violations


def test_lifecycle_event_single_constructor():
    violations: list[str] = []
    for path in sorted(APP_ROOT.rglob("*.py")):
        if path in EXEMPT_FILES:
            continue
        if any(parent in EXEMPT_DIRS for parent in path.parents):
            continue
        violations.extend(_scan_file(path))
    assert violations == [], "INV-C7-2 violations:\n" + "\n".join(violations)


def test_gate_detects_violation_negative_control(tmp_path):
    """反向自证：gate 能抓到三类违规（防 gate 本身空转）。"""
    sample = tmp_path / "bad.py"
    sample.write_text(
        "from app.domain.models.event import LifecycleEvent as LE\n"
        "x = LE(unit_id='u')\n"
        "y = LE.model_construct()\n"
        "class Evil(LE):\n    pass\n"
    )
    found = _scan_file(sample)
    assert len(found) == 3
