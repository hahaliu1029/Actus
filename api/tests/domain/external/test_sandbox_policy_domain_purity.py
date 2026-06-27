"""INV-3 (§8.10): the three C5a pure files import no FastAPI / SQLAlchemy /
app.infrastructure.* and never call get_settings(). Top-level scan only —
TYPE_CHECKING / function-body imports are nested and intentionally allowed.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_TARGETS = [
    ("app", "domain", "models", "sandbox_policy.py"),
    ("app", "domain", "services", "safety", "sandbox_policy_compiler.py"),
    ("app", "domain", "external", "policy_snapshot_sink.py"),
    ("app", "domain", "services", "safety", "command_policy_evaluator.py"),
]
_FORBIDDEN_IMPORT_PREFIXES = ("fastapi", "sqlalchemy", "app.infrastructure")


def _resolve(parts: tuple[str, ...]) -> Path:
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor.joinpath(*parts)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"could not locate {'/'.join(parts)} from {here}")


def _top_level_imports(tree: ast.Module) -> list[str]:
    mods: list[str] = []
    for node in tree.body:  # top-level only
        if isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mods.append(node.module or "")
    return mods


@pytest.mark.parametrize("parts", _TARGETS, ids=lambda p: p[-1])
def test_pure_file_imports(parts):
    src = _resolve(parts)
    tree = ast.parse(src.read_text(encoding="utf-8"))

    # (a) no top-level import of a forbidden package
    for mod in _top_level_imports(tree):
        assert not mod.startswith(_FORBIDDEN_IMPORT_PREFIXES), (
            f"{parts[-1]} must not import {mod} (INV-3 domain purity)"
        )

    # (b) get_settings is never imported or referenced. AST-ONLY — a docstring
    #     that merely NAMES get_settings/FastAPI/SQLAlchemy (the pure files'
    #     module docstrings do exactly that) must NOT trip the gate, so we never
    #     substring-scan raw text. [codex planR2 P1 — the substring scan was self-failing.]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert all(a.name != "get_settings" for a in node.names), (
                f"{parts[-1]} must not import get_settings (INV-3)"
            )
        elif isinstance(node, ast.Name):
            assert node.id != "get_settings", (
                f"{parts[-1]} must not reference get_settings (INV-3)"
            )
        elif isinstance(node, ast.Attribute):
            assert node.attr != "get_settings", (
                f"{parts[-1]} must not reference get_settings (INV-3)"
            )
