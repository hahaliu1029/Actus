"""R5 CS4 migration 形状检查（无 DB 依赖）。

覆盖 test plan §1/§5 里不需要真实 DB 的 3 条：
- ``test_r5_migration_only_ddl``：r5 migration 只含 DDL，**无 INSERT / bulk_insert / op.execute 写路径**
- ``test_r5_alembic_chain_head``：r5 挂到 memory chain 的当前 head（``m3_memory_system_notifications``）
- ``test_backfill_not_in_alembic_chain``：``alembic/versions/`` 下无 ``backfill`` 名字的 migration（脱离 alembic chain 验证）
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VERSIONS_DIR = REPO_ROOT / "api" / "alembic" / "versions"
R5_MIGRATION = VERSIONS_DIR / "r5_add_tool_approval_grants.py"


def test_r5_migration_only_ddl() -> None:
    """r5 migration 只含 DDL：create_table / add_column / create_index /
    create_foreign_key / drop_* 等。禁 ``op.execute`` / ``op.bulk_insert`` /
    文本 ``INSERT`` SQL 字面量（这些是 data migration，设计明确脱离 alembic）。"""
    tree = ast.parse(R5_MIGRATION.read_text())

    # 禁用的 alembic op 函数名
    FORBIDDEN_OP_CALLS = {"execute", "bulk_insert"}
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # op.execute / op.bulk_insert
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "op"
            and node.func.attr in FORBIDDEN_OP_CALLS
        ):
            violations.append(
                f"{R5_MIGRATION.name}:{node.lineno}: op.{node.func.attr}() is data migration"
            )

    # 文本 INSERT 字面量（sa.text("INSERT ...")）
    content = R5_MIGRATION.read_text()
    for match in re.finditer(r"""['"]\s*INSERT\s+INTO""", content, re.IGNORECASE):
        violations.append(
            f"{R5_MIGRATION.name}: INSERT literal at offset {match.start()} is data migration"
        )

    assert not violations, (
        "r5 migration must be DDL-only (backfill is a separate CLI, not alembic):\n"
        + "\n".join(violations)
    )


def test_r5_alembic_chain_head() -> None:
    """r5 migration 的 ``down_revision`` 锚定到 memory chain 当前 head。

    r5 挂在 ``m3_memory_system_notifications`` 后：migration 自动 upgrade 时
    r5 位于正确的 chain 位置，不会与未来的 memory migration 分叉。
    """
    tree = ast.parse(R5_MIGRATION.read_text())
    revision: str | None = None
    down_revision: object = object()  # sentinel
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign):
            continue
        if not isinstance(node.target, ast.Name):
            continue
        if node.target.id == "revision" and isinstance(node.value, ast.Constant):
            revision = node.value.value
        elif node.target.id == "down_revision" and isinstance(node.value, ast.Constant):
            down_revision = node.value.value

    assert revision == "r5_add_tool_approval_grants"
    assert down_revision == "m3_memory_system_notifications", (
        f"r5 must chain after m3_memory_system_notifications, got {down_revision!r}"
    )


def test_backfill_not_in_alembic_chain() -> None:
    """``alembic/versions/`` 下无 backfill 相关 migration（R5 backfill 脱离 chain）。"""
    offenders = [
        p.name
        for p in VERSIONS_DIR.glob("*.py")
        if "backfill" in p.name.lower() and p.name != "__init__.py"
    ]
    assert not offenders, (
        "Backfill must live in app/cli/, NOT in alembic/versions/ (design doc §Distribution Plan step 3):\n"
        + "\n".join(offenders)
    )


def test_backfill_cli_exists_at_expected_path() -> None:
    """Backfill CLI 在 ``api/app/cli/backfill_approval_grants.py``。"""
    cli_path = REPO_ROOT / "api" / "app" / "cli" / "backfill_approval_grants.py"
    assert cli_path.exists(), (
        "Backfill CLI expected at api/app/cli/backfill_approval_grants.py "
        "(Actus 约定 app.cli/ 放独立运维脚本)"
    )
    # Docstring 必须包含正确的调用命令（防 R5a 自 review LOW-1 回潮）
    content = cli_path.read_text()
    assert "uv run python -m app.cli.backfill_approval_grants" in content
    # 不应该继续用 app.scripts 名字
    assert "app.scripts.backfill_approval_grants" not in content
