"""N2 AST guard: chat SSE endpoint 函数体内直接出现的 ServerSentEvent 构造点必带 id=.

用 decorator-based selector 定位 @router.post(...) 装饰、路径为 "/{session_id}/chat"
的 async 函数. 兼容 kwarg 形式 `@router.post(path=...)` 与 positional 形式
`@router.post("...")`. 不扫 stream_sessions / create_skill_ai.

**已知局限**: 若未来把 ServerSentEvent 构造抽到 helper 函数里, 此 guard 的
_find_server_sent_event_calls 只 walk chat_fn 子树, 不追 helper 调用链 — 此类
重构需主动 re-evaluate 并更新此 guard 的 selector (或改为追调用链的版本).

Spec: docs/superpowers/specs/2026-04-17-n2-sse-transport-repair-design.md §2, §7
"""

from __future__ import annotations

import ast
from pathlib import Path


SESSION_ROUTES = (
    Path(__file__).resolve().parents[2]
    / "app"
    / "interfaces"
    / "endpoints"
    / "session_routes.py"
)

_CHAT_PATH = "/{session_id}/chat"


def _decorator_path_literal(deco: ast.Call) -> str | None:
    """Return the path string literal in @router.post(...) or None."""
    # kwarg form: @router.post(path="...")
    for kw in deco.keywords:
        if kw.arg == "path" and isinstance(kw.value, ast.Constant):
            return kw.value.value
    # positional form: @router.post("...")
    if deco.args:
        first = deco.args[0]
        if isinstance(first, ast.Constant):
            return first.value
    return None


def _find_chat_endpoint(module: ast.Module) -> ast.AsyncFunctionDef:
    """Locate @router.post(...) decorated async function whose path == chat."""
    for node in ast.walk(module):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        for deco in node.decorator_list:
            if not isinstance(deco, ast.Call):
                continue
            if not (
                isinstance(deco.func, ast.Attribute)
                and deco.func.attr == "post"
            ):
                continue
            if _decorator_path_literal(deco) == _CHAT_PATH:
                return node
    raise AssertionError(
        f"chat endpoint not found (expected @router.post(...) with path={_CHAT_PATH!r})"
    )


def _find_server_sent_event_calls(fn: ast.AsyncFunctionDef) -> list[ast.Call]:
    found: list[ast.Call] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "ServerSentEvent":
                found.append(node)
    return found


def test_chat_sse_yield_includes_id_kwarg() -> None:
    source = SESSION_ROUTES.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(SESSION_ROUTES))
    chat_fn = _find_chat_endpoint(module)
    calls = _find_server_sent_event_calls(chat_fn)

    assert calls, "expected at least one ServerSentEvent(...) in chat endpoint body"
    missing: list[int] = []
    for call in calls:
        kwarg_names = {kw.arg for kw in call.keywords if kw.arg}
        if "id" not in kwarg_names:
            missing.append(call.lineno)
    assert not missing, (
        f"ServerSentEvent(...) calls at lines {missing} in chat endpoint body "
        "missing `id=` kwarg (N2 wire contract; CS3 ADR)"
    )
