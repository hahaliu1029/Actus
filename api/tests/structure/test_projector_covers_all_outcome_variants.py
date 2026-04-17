"""R4 P2.1 executable guard 2: projector 必须显式处理所有 ToolOutcome variant.

扫 `app.domain.models.tool_result.ToolOutcome` union 的所有 variant 类,
断言每个 variant 都在 `_function_result_from_outcome` 里被 isinstance 显式处理.
新 R2 variant 引入但 projector 未跟上时 CI 自动红灯.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import get_args


def _collect_outcome_variants() -> set[str]:
    """从 ToolOutcome 类型注解收集所有 variant 类名."""
    from app.domain.models.tool_result import ToolOutcome
    union_args = get_args(ToolOutcome)
    # Annotated[Union[...], Field(discriminator=...)] — first arg is the actual union.
    # If ToolOutcome is plain Union, union_args IS the tuple of variants.
    if len(union_args) == 1 or not inspect.isclass(union_args[0]):
        # Annotated case — unpack inner union
        inner = union_args[0]
        variant_classes = get_args(inner)
    else:
        variant_classes = union_args
    return {cls.__name__ for cls in variant_classes if inspect.isclass(cls)}


def _collect_handled_variants_in_projector() -> set[str]:
    """AST 扫 _function_result_from_outcome, 提取所有 isinstance(outcome, XYZ) 的 XYZ."""
    from app.application.services import tool_event_envelope_v1
    src = Path(inspect.getfile(tool_event_envelope_v1)).read_text()
    tree = ast.parse(src)
    handled: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_function_result_from_outcome":
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id == "isinstance"
                    and len(sub.args) >= 2
                    and isinstance(sub.args[1], ast.Name)
                ):
                    handled.add(sub.args[1].id)
    return handled


def test_projector_covers_all_outcome_variants() -> None:
    """R2 ToolOutcome union 里所有 variant 必须在 projector 显式处理."""
    declared = _collect_outcome_variants()
    handled = _collect_handled_variants_in_projector()
    missing = declared - handled
    assert not missing, (
        f"R4 P2.1 executable guard: projector 未处理 ToolOutcome variant: {missing}. "
        f"新 variant 引入必须同 PR 补 _function_result_from_outcome 的 isinstance 分支."
    )
