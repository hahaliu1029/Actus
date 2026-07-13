"""D1a: add extension governance registry (4 tables).
Revision ID: d1a_add_extension_registry
Revises: c41a_add_subagent_runs
Create Date: 2026-07-11
Spec ref: docs/superpowers/specs/2026-07-10-d1a-extension-governance-design.md §3.
治理 overlay 四表：extensions（注册表+pin+observed）/ plugin_memberships /
extension_install_operations（saga 态）/ extension_audit_log。
不存 config/content 本体（INV-D1-4）。单 migration 全 epic（后续任务禁新增）。
Downgrade（DESTRUCTIVE）：drop 四表——治理 overlay 数据丢失可接受（内容权威在
config.yaml/skill store，重扫 reconcile 可重建 unpinned 行；pins 丢失=回到未批准态）。
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "d1a_add_extension_registry"
down_revision = "c41a_add_subagent_runs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "extensions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("ext_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default=sa.text("'active'")),
        sa.Column("quarantine_reason", sa.Text(), nullable=True),
        sa.Column("trust_origin", sa.Text(), nullable=False),
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("source_ref", sa.Text(), nullable=True),
        sa.Column("version", sa.Text(), nullable=True),
        sa.Column("artifact_hash", sa.Text(), nullable=True),
        sa.Column("surface_hash", sa.Text(), nullable=True),
        sa.Column("config_fingerprint", sa.Text(), nullable=True),
        sa.Column("hash_schema_version", sa.Integer(), nullable=False),
        sa.Column("observed_surface_hash", sa.Text(), nullable=True),
        sa.Column("observed_artifact_hash", sa.Text(), nullable=True),
        sa.Column("observed_config_fingerprint", sa.Text(), nullable=True),
        sa.Column("observed_hash_schema_version", sa.Integer(), nullable=True),
        sa.Column("last_observed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_mismatch_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("pinned_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("pinned_by", sa.String(length=255), nullable=True),
        sa.Column("scan_verdict", sa.Text(), nullable=True),
        sa.Column("scan_report", postgresql.JSONB(), nullable=True),
        sa.Column("source_missing_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("installed_by", sa.String(length=255), nullable=True),
        sa.Column("registry_source", sa.Text(), nullable=True),
        sa.Column("registry_key", sa.Text(), nullable=True),
        sa.Column("registry_version", sa.Text(), nullable=True),
        sa.Column("row_revision", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "(status = 'quarantined') = (quarantine_reason IS NOT NULL)",
            name="ck_extensions_quarantine_reason_pairing",
        ),
    )
    op.create_index(
        "uq_extensions_kind_ext_id_live", "extensions", ["kind", "ext_id"],
        unique=True, postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.create_index(
        "ix_extensions_status_updated", "extensions",
        ["status", sa.text("updated_at DESC")],
        postgresql_where=sa.text("deleted_at IS NULL"),
    )

    op.create_table(
        "plugin_memberships",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("plugin_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("extensions.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("child_extension_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("extensions.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("declared_component_id", sa.Text(), nullable=False),
        sa.Column("expected_hash", sa.Text(), nullable=True),
        sa.Column("managed_by_plugin", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("installed_version", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("plugin_id", "child_extension_id", name="uq_plugin_memberships_pair"),
        sa.UniqueConstraint("child_extension_id", name="uq_plugin_memberships_child"),
    )

    op.create_table(
        "extension_install_operations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("operation_type", sa.Text(), nullable=False),
        sa.Column("plugin_extension_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("extensions.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("initiated_by", sa.String(length=255), nullable=False),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column("steps", postgresql.JSONB(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index(
        "uq_extension_install_ops_inflight", "extension_install_operations",
        ["plugin_extension_id"], unique=True,
        postgresql_where=sa.text("state = 'in_progress'"),
    )

    op.create_table(
        "extension_audit_log",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("extension_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("ext_id", sa.Text(), nullable=True),
        sa.Column("actor_user_id", sa.String(length=255), nullable=True),
        sa.Column("event", sa.Text(), nullable=False),
        sa.Column("before", postgresql.JSONB(), nullable=True),
        sa.Column("after", postgresql.JSONB(), nullable=True),
        sa.Column("details", postgresql.JSONB(), nullable=True),
        sa.Column("correlation_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index(
        "ix_extension_audit_log_created_id", "extension_audit_log", ["created_at", "id"],
    )


def downgrade() -> None:
    op.drop_index("ix_extension_audit_log_created_id", table_name="extension_audit_log")
    op.drop_table("extension_audit_log")
    op.drop_index("uq_extension_install_ops_inflight", table_name="extension_install_operations")
    op.drop_table("extension_install_operations")
    op.drop_table("plugin_memberships")
    op.drop_index("ix_extensions_status_updated", table_name="extensions")
    op.drop_index("uq_extensions_kind_ext_id_live", table_name="extensions")
    op.drop_table("extensions")
