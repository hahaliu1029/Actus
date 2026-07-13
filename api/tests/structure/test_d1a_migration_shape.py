"""D1a migration 形状检查（无 DB）：revision 链、四表、关键约束、只 DDL，
以及 migration DDL ↔ ORM 交叉门（列集/nullability 一致 + INV-D1-4 内容列负检）。

导入 ORM 模型只注册 SQLAlchemy metadata，不建连接，故仍属「无 DB」。
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from app.infrastructure.models.extension_governance import (
    ExtensionAuditLogModel,
    ExtensionInstallOperationModel,
    ExtensionModel,
    PluginMembershipModel,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MIGRATION = REPO_ROOT / "api" / "alembic" / "versions" / "d1a_add_extension_registry.py"


def test_revision_chain():
    content = MIGRATION.read_text()
    assert 'revision = "d1a_add_extension_registry"' in content
    assert 'down_revision = "c41a_add_subagent_runs"' in content   # F23


def test_four_tables_created():
    content = MIGRATION.read_text()
    for table in ("extensions", "plugin_memberships",
                  "extension_install_operations", "extension_audit_log"):
        assert f'"{table}"' in content, table
    assert content.count("op.create_table(") == 4


def test_key_constraints_present():
    content = MIGRATION.read_text()
    # 部分唯一 + CHECK 双向 + saga 互斥 + audit 分页索引
    assert "uq_extensions_kind_ext_id_live" in content
    assert "deleted_at IS NULL" in content
    assert "(status = 'quarantined') = (quarantine_reason IS NOT NULL)" in content
    assert "uq_extension_install_ops_inflight" in content
    assert "state = 'in_progress'" in content
    assert "ix_extension_audit_log_created_id" in content


def test_only_ddl_no_data_migration():
    tree = ast.parse(MIGRATION.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "op"):
            assert node.func.attr not in {"execute", "bulk_insert"}, (
                f"line {node.lineno}: op.{node.func.attr}() is data migration")
    assert not re.search(r"""['"]\s*INSERT\s+INTO""", MIGRATION.read_text(), re.IGNORECASE)


# ===========================================================================
# Finding 5 (P2) — migration DDL ↔ ORM cross-gate.
# The tests above only prove the 4 tables + a handful of named constraints are
# textually present. They do NOT prove the per-table COLUMN SET or per-column
# NULLABILITY agree with the ORM models — a drift that would ship a schema the
# app can't map (or a silently-nullable pin column). This AST-vs-ORM diff locks
# both, and re-asserts INV-D1-4 (no content-authority columns) on the DDL side.
# ===========================================================================

_ORM_BY_TABLE = {
    "extensions": ExtensionModel,
    "plugin_memberships": PluginMembershipModel,
    "extension_install_operations": ExtensionInstallOperationModel,
    "extension_audit_log": ExtensionAuditLogModel,
}

# INV-D1-4: the governance overlay stores no content authority (config/content/
# manifest/entry/bundle/source_code live in config.yaml / skill store, never here).
_FORBIDDEN_CONTENT_COLS = {"config", "content", "manifest", "entry", "bundle", "source_code"}


def _is_call(node: ast.AST, value_id: str, attr: str) -> bool:
    """True if ``node`` is a call ``value_id.attr(...)`` (e.g. ``sa.Column(...)``)."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == value_id
        and node.func.attr == attr
    )


def _col_nullable(col_call: ast.Call) -> bool:
    """Mirror how the migration writes nullability.

    ``nullable=False`` → False; a primary-key column (``primary_key=True`` with
    no explicit ``nullable``) → False; ``nullable=True`` or absent → True.
    """
    kw = {k.arg: k.value for k in col_call.keywords}
    pk = kw.get("primary_key")
    if isinstance(pk, ast.Constant) and pk.value is True:
        return False
    nullable = kw.get("nullable")
    if isinstance(nullable, ast.Constant):
        return bool(nullable.value)
    return True


def _parse_migration_tables() -> dict[str, dict[str, bool]]:
    """AST-extract {table_name: {column_name: nullable}} from upgrade()."""
    tree = ast.parse(MIGRATION.read_text())
    upgrade = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "upgrade"
    )
    tables: dict[str, dict[str, bool]] = {}
    for node in ast.walk(upgrade):
        if not _is_call(node, "op", "create_table"):
            continue
        table_name = node.args[0].value  # first positional str arg
        cols: dict[str, bool] = {}
        for arg in node.args[1:]:  # remaining args are Column/CheckConstraint/... calls
            if _is_call(arg, "sa", "Column"):
                cols[arg.args[0].value] = _col_nullable(arg)
        tables[table_name] = cols
    return tables


_MIGRATION_TABLES = _parse_migration_tables()


def test_migration_ast_extracts_four_tables():
    # sanity: the AST extractor found exactly the 4 governance tables
    assert set(_MIGRATION_TABLES) == set(_ORM_BY_TABLE)


def test_migration_columns_match_orm_columns():
    for table, orm in _ORM_BY_TABLE.items():
        mig_cols = _MIGRATION_TABLES[table]
        orm_cols = orm.__table__.columns
        assert set(mig_cols) == set(orm_cols.keys()), f"{table}: column-name set drift"


def test_migration_nullability_matches_orm():
    for table, orm in _ORM_BY_TABLE.items():
        mig_cols = _MIGRATION_TABLES[table]
        orm_cols = orm.__table__.columns
        for name, mig_nullable in mig_cols.items():
            assert mig_nullable == orm_cols[name].nullable, (
                f"{table}.{name}: nullability drift "
                f"(migration={mig_nullable}, orm={orm_cols[name].nullable})")


def test_no_content_authority_columns_in_ddl():
    # INV-D1-4 DDL-side negative: not one governance table declares a content col
    all_cols = {c for cols in _MIGRATION_TABLES.values() for c in cols}
    leaked = all_cols & _FORBIDDEN_CONTENT_COLS
    assert not leaked, f"INV-D1-4 DDL leak: forbidden content-authority column(s) {leaked}"
