"""S3 PR-1: structural checks on the lineage migration + its ORM mirror."""
from __future__ import annotations

import ast
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[2]  # api/
MIGRATION = (
    API_ROOT / "alembic" / "versions" / "s3pr1_add_session_depth_lineage.py"
)


def _assigns(tree: ast.AST) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (
                    isinstance(t, ast.Name)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    out[t.id] = node.value.value
    return out


def test_migration_revision_and_down_revision():
    assert MIGRATION.exists(), "s3pr1 migration file missing"
    a = _assigns(ast.parse(MIGRATION.read_text(encoding="utf-8")))
    assert a.get("revision") == "s3pr1_add_session_depth_lineage"
    assert a.get("down_revision") == "c2b1_mailbox_running_child_idx"
    assert len(a["revision"]) <= 32  # alembic_version.version_num varchar(32)


def test_migration_adds_depth_and_root_columns_and_backfills():
    src = MIGRATION.read_text(encoding="utf-8")
    assert "add_column" in src and '"depth"' in src and '"root_session_id"' in src
    # Backfill scopes the write to subagent rows only (roots use the default).
    assert "parent_session_id IS NOT NULL" in src
    # No index / FK / CHECK in PR-1 (design §4.1 steps 4-6).
    assert "create_index" not in src
    assert "ForeignKey" not in src and "foreign" not in src.lower()


def test_orm_lineage_columns_mirror_migration():
    from app.infrastructure.models.session import SessionModel

    depth_col = SessionModel.__table__.c.depth
    assert depth_col.nullable is False
    assert depth_col.server_default is not None  # mirrors DEFAULT 0

    root_col = SessionModel.__table__.c.root_session_id
    assert root_col.nullable is True
    assert root_col.type.length == 255  # VARCHAR(255), matches sessions.id width
