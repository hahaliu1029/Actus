"""C4 PR-1 — SubagentWorker port ABC + domain 纯度测试（spec §4/§8 + §7 PR-1）。"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest

import app.domain.external.subagent_worker as _c4_port_mod
import app.domain.models.subagent_worker as _c4_models_mod
from app.domain.external.subagent_worker import SubagentWorker
from app.domain.models.subagent_worker import (
    SubagentRunResult,
    WorkerLifecycleState,
    WorkerRuntimeType,
    WorkerTerminalOutcome,
)


def test_port_is_abstract() -> None:
    with pytest.raises(TypeError):
        SubagentWorker()  # type: ignore[abstract]


def test_run_is_coroutine_abstractmethod() -> None:
    assert getattr(SubagentWorker.run, "__isabstractmethod__", False) is True
    assert inspect.iscoroutinefunction(SubagentWorker.run)


@pytest.mark.asyncio
async def test_concrete_subclass_instantiable_and_runs() -> None:
    class _Concrete(SubagentWorker):
        async def run(self, spec):  # type: ignore[override]
            return SubagentRunResult(
                worker_runtime_type=WorkerRuntimeType.LOCAL,
                lifecycle_state=WorkerLifecycleState.TERMINAL,
                terminal_outcome=WorkerTerminalOutcome.SUCCESS,
            )

    worker = _Concrete()
    assert isinstance(worker, SubagentWorker)
    result = await worker.run(None)
    assert result.terminal_outcome == WorkerTerminalOutcome.SUCCESS


# ── domain 纯度守门（spec §8 — R1#P1 修）──────────────────────────────
# 既有 tests/domain/external/test_validate_attributes_domain_purity.py 经
# _resolve_observability_source() 硬编码只扫 observability.py 单文件，**不**覆盖
# C4 新文件。这里为两个新 domain 文件加专用 AST 扫描（直接对账 CLAUDE.md/§8 约束）。
_ALLOWED_TOP_LEVEL = frozenset({"__future__", "pydantic", "langchain_core", "langgraph"})


def _import_allowed(module: str) -> bool:
    if not module:
        return False
    root = module.split(".", 1)[0]
    if root in sys.stdlib_module_names:
        return True
    if root in _ALLOWED_TOP_LEVEL:
        return True
    # 同层 domain only（禁 app.application / infrastructure / interfaces）
    return module.startswith("app.domain.")


@pytest.mark.parametrize("module", [_c4_models_mod, _c4_port_mod])
def test_c4_domain_file_is_pure(module) -> None:
    """C4 domain 文件顶层 import 仅许 stdlib + pydantic + langchain/langgraph +
    同层 app.domain.*；且全文件无 fastapi/sqlalchemy（含 lazy import）。
    AST + 子串双查，新增禁项即红（mutation-proof）。"""
    text = Path(module.__file__).read_text(encoding="utf-8")
    assert "fastapi" not in text.lower()
    assert "sqlalchemy" not in text.lower()
    violations: list[str] = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _import_allowed(alias.name):
                    violations.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if not _import_allowed(node.module or ""):
                violations.append(f"from {node.module}")
    assert not violations, f"{module.__name__} 顶层 import 违反 domain 纯度: {violations}"
