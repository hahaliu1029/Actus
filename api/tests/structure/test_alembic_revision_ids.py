"""Alembic revision id shape checks."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VERSIONS_DIR = REPO_ROOT / "api" / "alembic" / "versions"
MAX_VERSION_NUM_LENGTH = 32


def _get_revision(tree: ast.AST) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "revision"
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    return node.value.value
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "revision"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    return None


def test_revision_ids_fit_default_alembic_version_table() -> None:
    """Alembic's default version table stores version_num as varchar(32)."""
    violations: list[str] = []
    for migration in sorted(VERSIONS_DIR.glob("*.py")):
        if migration.name == "__init__.py":
            continue
        revision = _get_revision(ast.parse(migration.read_text()))
        if revision is None:
            violations.append(f"{migration.name}: missing revision")
            continue
        if len(revision) > MAX_VERSION_NUM_LENGTH:
            violations.append(
                f"{migration.name}: revision {revision!r} has length {len(revision)}"
            )

    assert not violations, (
        "Alembic revision ids must fit alembic_version.version_num varchar(32):\n"
        + "\n".join(violations)
    )
