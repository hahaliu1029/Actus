"""B5 PR-S1-1 acceptance: ``domain/external/observability.py`` import purity.

Locks the domain-purity invariant via AST scan. POST reviewer-concern-1
in the B5 design doc: the canonical attribute contract module must not
take a top-level dependency on infrastructure / FastAPI / SQLAlchemy /
``app.*`` packages so the contract stays portable across consumers and
tests can be run without spinning up the full app graph.

The ``build_canonical_attributes`` helper does need to read contextvars
from ``app.infrastructure.observability.context`` at runtime; it does so
via a *lazy* import inside the function body. Function-body imports are
nested under ``ast.FunctionDef`` and therefore not visible in
``tree.body`` — this AST scan only walks the module's top-level
statement list, so the lazy import is allowed by construction.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

_ALLOWED_TOP_LEVEL_PREFIXES = frozenset(
    {
        "__future__",
        "langchain_core",
        "langgraph",
    }
)


def _module_root(module: str) -> str:
    return module.split(".", 1)[0] if module else ""


def _is_allowed(module: str) -> bool:
    root = _module_root(module)
    if not root:
        return False
    if root in sys.stdlib_module_names:
        return True
    if root in _ALLOWED_TOP_LEVEL_PREFIXES:
        return True
    return False


def _resolve_observability_source() -> Path:
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor / "app" / "domain" / "external" / "observability.py"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "could not locate api/app/domain/external/observability.py from "
        f"{here}"
    )


class TestObservabilityModuleTopLevelImports:
    def test_top_level_imports_allowlisted(self):
        src = _resolve_observability_source()
        tree = ast.parse(src.read_text(encoding="utf-8"))

        violations: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not _is_allowed(alias.name):
                        violations.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if not _is_allowed(module):
                    violations.append(f"from {module}")

        assert not violations, (
            "domain/external/observability.py must keep top-level imports "
            "stdlib-only (per Clean Architecture domain purity). "
            f"Violations: {violations}"
        )

    def test_no_fastapi_anywhere(self):
        src = _resolve_observability_source()
        contents = src.read_text(encoding="utf-8")
        assert "fastapi" not in contents.lower()

    def test_no_sqlalchemy_anywhere(self):
        src = _resolve_observability_source()
        contents = src.read_text(encoding="utf-8")
        assert "sqlalchemy" not in contents.lower()

    def test_lazy_infrastructure_import_present(self):
        src = _resolve_observability_source()
        tree = ast.parse(src.read_text(encoding="utf-8"))

        found_lazy = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "build_canonical_attributes":
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.ImportFrom):
                    if (child.module or "").startswith(
                        "app.infrastructure.observability"
                    ):
                        found_lazy = True
                        break
        assert found_lazy, (
            "build_canonical_attributes is expected to do a lazy "
            "`from app.infrastructure.observability.context import "
            "get_trace_context` inside its function body; refactoring "
            "this away would either break domain purity (top-level "
            "import) or break runtime context reads (no import at all)."
        )
