"""D1a 新增 domain 文件禁 FastAPI/SQLAlchemy import（项目硬约束）。"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
D1A_DOMAIN_FILES = [
    REPO_ROOT / "api" / "app" / "domain" / "models" / "extension_governance.py",
    REPO_ROOT / "api" / "app" / "domain" / "external" / "extension_admission.py",
]
FORBIDDEN_ROOTS = {"fastapi", "sqlalchemy", "starlette", "alembic"}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_d1a_domain_files_pure():
    for f in D1A_DOMAIN_FILES:
        assert f.exists(), f
        bad = _imported_roots(f) & FORBIDDEN_ROOTS
        assert not bad, f"{f.name} imports forbidden: {bad}"
