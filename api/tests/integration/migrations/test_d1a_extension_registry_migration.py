"""[D1a §3] 真 DB 约束验证：部分唯一 / CHECK 双向 / saga 互斥索引。CI-only。"""
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

pytestmark = pytest.mark.anyio


def _extension_row(**overrides):
    row = dict(
        id=str(uuid.uuid4()), kind="mcp", ext_id="srv-a", status="active",
        quarantine_reason=None, trust_origin="user_installed", source_type="config",
        hash_schema_version=1, row_revision=0,
    )
    row.update(overrides)
    return row


INSERT_EXT = text(
    "INSERT INTO extensions (id, kind, ext_id, status, quarantine_reason, trust_origin,"
    " source_type, hash_schema_version, row_revision)"
    " VALUES (:id, :kind, :ext_id, :status, :quarantine_reason, :trust_origin,"
    " :source_type, :hash_schema_version, :row_revision)"
)


async def test_partial_unique_allows_reinstall_after_soft_delete(db_session):
    await db_session.execute(INSERT_EXT, _extension_row())
    await db_session.flush()
    # 同名 live 行 → IntegrityError
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            await db_session.execute(INSERT_EXT, _extension_row())
    # 软删旧行后同名新行合法（§3.2）
    await db_session.execute(text("UPDATE extensions SET deleted_at = now() WHERE ext_id = 'srv-a'"))
    await db_session.execute(INSERT_EXT, _extension_row())
    await db_session.flush()


async def test_check_quarantine_reason_pairing_bidirectional(db_session):
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            await db_session.execute(
                INSERT_EXT, _extension_row(ext_id="q1", status="quarantined", quarantine_reason=None))
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            await db_session.execute(
                INSERT_EXT, _extension_row(ext_id="q2", status="active", quarantine_reason="admin_manual"))
    await db_session.execute(
        INSERT_EXT, _extension_row(ext_id="q3", status="quarantined", quarantine_reason="pin_mismatch"))
    await db_session.flush()


async def test_operations_inflight_mutex(db_session):
    plugin_id = str(uuid.uuid4())
    await db_session.execute(
        INSERT_EXT, _extension_row(id=plugin_id, kind="plugin", ext_id="org.example.pack",
                                   status="disabled", source_type="local"))
    ins_op = text(
        "INSERT INTO extension_install_operations"
        " (id, operation_type, plugin_extension_id, initiated_by, state, steps)"
        " VALUES (:id, 'plugin_install', :pid, 'admin-1', :state, '[]'::jsonb)")
    await db_session.execute(ins_op, {"id": str(uuid.uuid4()), "pid": plugin_id, "state": "in_progress"})
    await db_session.flush()
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            await db_session.execute(
                ins_op, {"id": str(uuid.uuid4()), "pid": plugin_id, "state": "in_progress"})
    # 终态并存合法（failed 可多个）
    await db_session.execute(ins_op, {"id": str(uuid.uuid4()), "pid": plugin_id, "state": "failed"})
    await db_session.execute(ins_op, {"id": str(uuid.uuid4()), "pid": plugin_id, "state": "failed"})
    await db_session.flush()
