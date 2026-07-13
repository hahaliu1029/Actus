"""D1a §3 扩展治理四表 ORM（治理 overlay——不存 config/content 本体，INV-D1-4）。

§3.6-pre 合同：未标 nullable 的列 NOT NULL；全时间列 timestamptz；
created_at/updated_at server_default=now()，updated_at 更新责任在 repo 层（onupdate）；
id 由应用侧 uuid4() 生成（非 DB 生成）。
"""
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class ExtensionModel(Base):
    """治理注册表（§3.1）。"""

    __tablename__ = "extensions"
    __table_args__ = (
        # 部分唯一：同名重装新 UUID，旧行审计留存（§3.1）
        Index(
            "uq_extensions_kind_ext_id_live",
            "kind", "ext_id",
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
        ),
        # 管理台热查询
        Index(
            "ix_extensions_status_updated",
            "status", text("updated_at DESC"),
            postgresql_where=text("deleted_at IS NULL"),
        ),
        # CHECK 双向（R8#2）：quarantined ⇔ reason 非空
        CheckConstraint(
            "(status = 'quarantined') = (quarantine_reason IS NOT NULL)",
            name="ck_extensions_quarantine_reason_pairing",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    ext_id: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'active'"))
    quarantine_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    trust_origin: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(Text, nullable=False)
    source_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    surface_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    config_fingerprint: Mapped[str | None] = mapped_column(Text, nullable=True)
    hash_schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    observed_surface_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    observed_artifact_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    observed_config_fingerprint: Mapped[str | None] = mapped_column(Text, nullable=True)
    observed_hash_schema_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_mismatch_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    pinned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    pinned_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    scan_verdict: Mapped[str | None] = mapped_column(Text, nullable=True)
    scan_report: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    source_missing_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    installed_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    registry_source: Mapped[str | None] = mapped_column(Text, nullable=True)   # D1b 预留，本期恒 NULL
    registry_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    registry_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    row_revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()"))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class PluginMembershipModel(Base):
    """plugin → 成员关系（§3.3）。删除语义=物理删除（R13#7，无软删列）。"""

    __tablename__ = "plugin_memberships"
    __table_args__ = (
        UniqueConstraint("plugin_id", "child_extension_id", name="uq_plugin_memberships_pair"),
        UniqueConstraint("child_extension_id", name="uq_plugin_memberships_child"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    plugin_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("extensions.id", ondelete="RESTRICT"), nullable=False)
    child_extension_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("extensions.id", ondelete="RESTRICT"), nullable=False)
    declared_component_id: Mapped[str] = mapped_column(Text, nullable=False)
    expected_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    managed_by_plugin: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=text("true"))
    installed_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()"))


class ExtensionInstallOperationModel(Base):
    """plugin saga 操作态（§3.4）。steps=write-ahead intent 唯一 schema（R7#1）。"""

    __tablename__ = "extension_install_operations"
    __table_args__ = (
        # 同一 plugin 并发 install/uninstall 数据库层互斥
        Index(
            "uq_extension_install_ops_inflight",
            "plugin_extension_id",
            unique=True,
            postgresql_where=text("state = 'in_progress'"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    operation_type: Mapped[str] = mapped_column(Text, nullable=False)   # {plugin_install, plugin_uninstall}
    plugin_extension_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("extensions.id", ondelete="RESTRICT"), nullable=False)
    initiated_by: Mapped[str] = mapped_column(String(255), nullable=False)
    state: Mapped[str] = mapped_column(Text, nullable=False)            # {in_progress, completed, failed, compensated}
    steps: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()"))


class ExtensionAuditLogModel(Base):
    """扩展治理审计（§3.5）。identity snapshot 自持（行可能已软删）。"""

    __tablename__ = "extension_audit_log"
    __table_args__ = (
        Index("ix_extension_audit_log_created_id", "created_at", "id"),   # cursor 分页
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    extension_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    ext_id: Mapped[str | None] = mapped_column(Text, nullable=True)     # R13#5：仅 install_rejected 身份解析前
    actor_user_id: Mapped[str | None] = mapped_column(String(255), nullable=True)   # NULL=系统
    event: Mapped[str] = mapped_column(Text, nullable=False)
    before: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    after: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    correlation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()"))
