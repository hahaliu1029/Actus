"""R4 CS3 envelope invariant enforcement (CI AST scan).

5 rules (Round 2b/c refined):
- Rule 1: domain typed 类 import 白名单
- Rule 2: function_result.success 直读禁令 (仅 projector 允许)
- Rule 3: interfaces/frontend 禁 domain typed 字符串
- Rule 4: ToolSSEEvent wire 序列化 callsite by_alias=True
- Rule 5: projector 唯一 ToolEventEnvelopeV1 构造点
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
APP_DIR = REPO_ROOT / "api" / "app"
TESTS_DIR = REPO_ROOT / "api" / "tests"
UI_DIR = REPO_ROOT / "ui" / "src"


# ============================================================
# Rule 1: domain typed 类 import 白名单
# ============================================================

TYPED_CLASSES = {
    "ToolArtifact", "ToolOutcome", "AllowSuccess", "AllowError",
    "Denied", "Asked", "Passthrough",
    "MultimodalPayload", "TextBlock", "ImageUrlBlock", "FileBlock", "MultimodalBlock",
}

RULE_1_WHITELIST = {
    "api/app/domain/models/tool_result.py",
    "api/app/domain/services/graphs/",
    "api/app/domain/services/tools/",
    "api/app/application/services/tool_event_envelope_v1.py",
    "api/app/infrastructure/external/llm/_error_prefix.py",
}


def _is_in_whitelist(rel_path: str, whitelist: set[str]) -> bool:
    return any(rel_path.startswith(wl) or rel_path == wl for wl in whitelist)


def _iter_app_py_files():
    for py in APP_DIR.rglob("*.py"):
        rel = py.relative_to(REPO_ROOT).as_posix()
        yield py, rel


def _is_test_file(rel: str) -> bool:
    return "/tests/" in rel or rel.startswith("api/tests/")


def test_rule_1_domain_typed_import_whitelist() -> None:
    """Whitelist-only modules may import typed ToolOutcome / ToolArtifact / MultimodalPayload classes."""
    violations: list[str] = []
    for py_path, rel in _iter_app_py_files():
        if _is_in_whitelist(rel, RULE_1_WHITELIST):
            continue
        try:
            tree = ast.parse(py_path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "app.domain.models.tool_result":
                for alias in node.names:
                    if alias.name in TYPED_CLASSES:
                        violations.append(f"{rel}: illegal import `{alias.name}`")
    assert not violations, (
        "Rule 1 violation: typed class import outside whitelist:\n"
        + "\n".join(violations)
    )


# ============================================================
# Rule 2: .success 直读禁令
# ============================================================

RULE_2_WHITELIST = {
    "api/app/application/services/tool_event_envelope_v1.py",
    "api/app/domain/models/tool_result.py",
}


def test_rule_2_no_direct_success_read() -> None:
    """禁止 event.function_result.success / msg.function_result.success 类直读."""
    violations: list[str] = []
    for py_path, rel in _iter_app_py_files():
        if _is_test_file(rel) or _is_in_whitelist(rel, RULE_2_WHITELIST):
            continue
        try:
            tree = ast.parse(py_path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "success"
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "function_result"
            ):
                violations.append(f"{rel}:{node.lineno}: `.function_result.success` direct read")
    assert not violations, (
        "Rule 2 violation: direct `.function_result.success` read outside projector:\n"
        + "\n".join(violations)
    )


# ============================================================
# Rule 3: interfaces / frontend 禁 domain typed 字符串
# ============================================================

FORBIDDEN_STRINGS_RULE_3 = {
    "ToolArtifact", "ToolOutcome", "AllowSuccess", "AllowError",
    "Denied", "Asked", "Passthrough",
    "MultimodalPayload", "TextBlock", "ImageUrlBlock", "FileBlock",
    # Note: 不扫 DecisionReason (允许 DecisionReasonWire), 不扫 ToolSource (CS1 允许穿透)
}


def test_rule_3_no_typed_string_in_interfaces_or_frontend() -> None:
    violations: list[str] = []
    # Backend interfaces 扫文本
    interfaces_dir = APP_DIR / "interfaces"
    for py_path in interfaces_dir.rglob("*.py"):
        rel = py_path.relative_to(REPO_ROOT).as_posix()
        content = py_path.read_text()
        for fstr in FORBIDDEN_STRINGS_RULE_3:
            if fstr in content:
                violations.append(f"{rel}: forbidden typed string `{fstr}`")
    # Frontend TS/TSX 扫
    for ext in ("*.ts", "*.tsx"):
        for ts_path in UI_DIR.rglob(ext):
            rel = ts_path.relative_to(REPO_ROOT).as_posix()
            # Skip node_modules / build artifacts
            if "node_modules" in rel or "/dist/" in rel or "/.next/" in rel:
                continue
            content = ts_path.read_text()
            for fstr in FORBIDDEN_STRINGS_RULE_3:
                if fstr in content:
                    violations.append(f"{rel}: forbidden typed string `{fstr}`")

    assert not violations, (
        "Rule 3 violation: domain typed class name in interfaces/frontend:\n"
        + "\n".join(violations)
    )


# ============================================================
# Rule 4: SSE wire 序列化 by_alias=True 强制
# 两处扫描源:
# 1. api/app/interfaces/schemas/event.py — ToolSSEEvent class body
# 2. api/app/interfaces/endpoints/session_routes.py — ServerSentEvent(data=...) callsite
# ============================================================


def test_rule_4_sse_wire_serialization_by_alias() -> None:
    """ToolSSEEvent 类 或 session_routes 写 ServerSentEvent(data=...) 时,
    所有 model_dump / model_dump_json 调用必须带 by_alias=True 或走 to_sse_data_json()."""
    violations: list[str] = []

    # --- 源 1: interfaces/schemas/event.py 的 ToolSSEEvent 类体内 ---
    event_py = APP_DIR / "interfaces" / "schemas" / "event.py"
    tree = ast.parse(event_py.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ToolSSEEvent":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                    if sub.func.attr in ("model_dump", "model_dump_json"):
                        has_by_alias = any(
                            kw.arg == "by_alias"
                            and isinstance(kw.value, ast.Constant)
                            and kw.value.value is True
                            for kw in sub.keywords
                        )
                        if not has_by_alias:
                            violations.append(
                                f"event.py:{sub.lineno}: ToolSSEEvent.{sub.func.attr}() missing by_alias=True"
                            )

    # --- 源 2: session_routes.py 的 SSE frame 写出点 ---
    # 仅扫描 sse_event.* 类调用（agent chat 路径），跳过非 sse_event 对象的 model_dump_json 调用。
    # 原因：session list endpoint 的 ListSessionResponse.model_dump_json() 不涉及 ToolSSEEvent wire 格式，
    # 不在本 rule 管辖范围内。
    session_routes_py = APP_DIR / "interfaces" / "endpoints" / "session_routes.py"
    tree = ast.parse(session_routes_py.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ServerSentEvent"
        ):
            for kw in node.keywords:
                if kw.arg != "data":
                    continue
                if not isinstance(kw.value, ast.Call):
                    continue
                call = kw.value
                if not isinstance(call.func, ast.Attribute):
                    continue

                # 只扫 sse_event 变量上的方法调用（agent chat 路径）
                # 走 attribute chain 根: 既覆盖 sse_event.x() 也覆盖 sse_event.data.x()
                # (Round 2f 的回归向量是 `sse_event.data.model_dump_json()`, 必须捕获)
                root = call.func.value
                while isinstance(root, ast.Attribute):
                    root = root.value
                if not (isinstance(root, ast.Name) and "sse_event" in root.id):
                    continue

                method_name = call.func.attr
                if method_name == "to_sse_data_json":
                    # OK — 走规定的 wire 路径
                    continue
                if method_name in ("model_dump_json", "model_dump"):
                    has_by_alias = any(
                        k.arg == "by_alias"
                        and isinstance(k.value, ast.Constant)
                        and k.value.value is True
                        for k in call.keywords
                    )
                    if not has_by_alias:
                        violations.append(
                            f"session_routes.py:{call.lineno}: "
                            f"ServerSentEvent(data=sse_event.{method_name}()) missing by_alias=True. "
                            f"Use sse_event.to_sse_data_json() (preferred) or pass by_alias=True."
                        )

    assert not violations, (
        "Rule 4 violation: SSE wire serialization without by_alias=True:\n"
        + "\n".join(violations)
    )


# ============================================================
# Rule 5: projector 唯一 ToolEventEnvelopeV1 构造点
# ============================================================

RULE_5_WHITELIST = {
    "api/app/application/services/tool_event_envelope_v1.py",
}


def test_rule_5_projector_is_unique_envelope_constructor() -> None:
    """api/app/ 下除 projector 外禁止 `ToolEventEnvelopeV1(...)` Call node."""
    violations: list[str] = []
    for py_path, rel in _iter_app_py_files():
        if _is_test_file(rel) or _is_in_whitelist(rel, RULE_5_WHITELIST):
            continue
        try:
            tree = ast.parse(py_path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "ToolEventEnvelopeV1"
            ):
                violations.append(f"{rel}:{node.lineno}: illegal ToolEventEnvelopeV1(...) construction")
    assert not violations, (
        "Rule 5 violation: ToolEventEnvelopeV1 constructed outside projector:\n"
        + "\n".join(violations)
    )


# ============================================================
# Rule 6: _translate_outcome 所有 caller 必须透传 enabled_outcome_variants
# (Round 2g P1 — 防止死配置 regression)
# ============================================================


def test_rule_6_translate_outcome_callers_pass_enabled_variants() -> None:
    """react_graph.py 内所有 _translate_outcome(...) Call 必须带
    enabled_outcome_variants kwarg. 阻止未来 contributor 新加 caller 时漏掉 kwarg
    让 enabled_outcome_variants=None 悄悄走默认 (= runtime enforcement 失效)."""
    react_graph_py = APP_DIR / "domain" / "services" / "graphs" / "react_graph.py"
    tree = ast.parse(react_graph_py.read_text())

    violations: list[str] = []
    for node in ast.walk(tree):
        is_translate = (
            isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "_translate_outcome")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "_translate_outcome")
            )
        )
        if not is_translate:
            continue
        has_kwarg = any(kw.arg == "enabled_outcome_variants" for kw in node.keywords)
        if not has_kwarg:
            violations.append(
                f"react_graph.py:{node.lineno}: _translate_outcome(...) missing "
                f"enabled_outcome_variants kwarg"
            )

    assert not violations, (
        "Rule 6 violation: _translate_outcome caller 未透传 enabled_outcome_variants:\n"
        + "\n".join(violations)
    )
