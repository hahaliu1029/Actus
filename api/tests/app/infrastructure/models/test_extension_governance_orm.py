"""D1a §3 四表 ORM metadata snapshot（本地无 DB；INV-D1-4 列封闭——registry 不存 config/content 本体）。"""
from sqlalchemy.dialects.postgresql import JSONB, UUID

from app.infrastructure.models.extension_governance import (
    ExtensionAuditLogModel,
    ExtensionInstallOperationModel,
    ExtensionModel,
    PluginMembershipModel,
)

# spec §3.1 列字面复刻（列封闭 snapshot：多列/少列/改名即红——INV-D1-4 的 ORM 侧）
EXPECTED_EXTENSIONS_COLUMNS = {
    "id", "kind", "ext_id", "status", "quarantine_reason", "trust_origin",
    "source_type", "source_ref", "version",
    "artifact_hash", "surface_hash", "config_fingerprint", "hash_schema_version",
    "observed_surface_hash", "observed_artifact_hash", "observed_config_fingerprint",
    "observed_hash_schema_version",
    "last_observed_at", "last_verified_at", "last_mismatch_at",
    "pinned_at", "pinned_by", "scan_verdict", "scan_report",
    "source_missing_at", "installed_by",
    "registry_source", "registry_key", "registry_version",
    "row_revision", "created_at", "updated_at", "deleted_at",
}
EXPECTED_MEMBERSHIP_COLUMNS = {
    "id", "plugin_id", "child_extension_id", "declared_component_id",
    "expected_hash", "managed_by_plugin", "installed_version", "created_at",
}
EXPECTED_OPERATION_COLUMNS = {
    "id", "operation_type", "plugin_extension_id", "initiated_by",
    "state", "steps", "error", "created_at", "updated_at",
}
EXPECTED_AUDIT_COLUMNS = {
    "id", "extension_id", "kind", "ext_id", "actor_user_id", "event",
    "before", "after", "details", "correlation_id", "created_at",
}

NULLABLE_EXTENSIONS = {
    "quarantine_reason", "source_ref", "version",
    "artifact_hash", "surface_hash", "config_fingerprint",
    "observed_surface_hash", "observed_artifact_hash", "observed_config_fingerprint",
    "observed_hash_schema_version",
    "last_observed_at", "last_verified_at", "last_mismatch_at",
    "pinned_at", "pinned_by", "scan_verdict", "scan_report",
    "source_missing_at", "installed_by",
    "registry_source", "registry_key", "registry_version", "deleted_at",
}


def _cols(model):
    return {c.name: c for c in model.__table__.columns}


class TestExtensionsTable:
    def test_table_name_and_columns_closed(self):
        assert ExtensionModel.__tablename__ == "extensions"
        assert set(_cols(ExtensionModel)) == EXPECTED_EXTENSIONS_COLUMNS

    def test_nullability_contract(self):
        # §3.6-pre 默认约定：未标 NULL 一律 NOT NULL；显式例外=deleted_at
        cols = _cols(ExtensionModel)
        for name, col in cols.items():
            assert col.nullable == (name in NULLABLE_EXTENSIONS), name

    def test_no_content_columns(self):
        # INV-D1-4：治理 overlay 不复制 config/content 本体
        forbidden = {"config", "content", "manifest", "bundle", "command", "url", "env"}
        assert forbidden.isdisjoint(set(_cols(ExtensionModel)))

    def test_key_types(self):
        cols = _cols(ExtensionModel)
        assert isinstance(cols["id"].type, UUID)
        assert isinstance(cols["scan_report"].type, JSONB)
        assert cols["row_revision"].default.arg == 0
        assert cols["pinned_by"].type.length == 255      # R23：对齐 users.id str
        assert cols["installed_by"].type.length == 255


class TestMembershipTable:
    def test_columns_closed(self):
        assert PluginMembershipModel.__tablename__ == "plugin_memberships"
        assert set(_cols(PluginMembershipModel)) == EXPECTED_MEMBERSHIP_COLUMNS

    def test_fks_restrict(self):
        for col_name in ("plugin_id", "child_extension_id"):
            fks = list(_cols(PluginMembershipModel)[col_name].foreign_keys)
            assert len(fks) == 1
            assert fks[0].ondelete == "RESTRICT"
            assert fks[0].column.table.name == "extensions"


class TestOperationTable:
    def test_columns_closed(self):
        assert ExtensionInstallOperationModel.__tablename__ == "extension_install_operations"
        assert set(_cols(ExtensionInstallOperationModel)) == EXPECTED_OPERATION_COLUMNS

    def test_plugin_fk_not_null_restrict(self):
        col = _cols(ExtensionInstallOperationModel)["plugin_extension_id"]
        assert col.nullable is False
        assert list(col.foreign_keys)[0].ondelete == "RESTRICT"

    def test_steps_jsonb_not_null(self):
        col = _cols(ExtensionInstallOperationModel)["steps"]
        assert isinstance(col.type, JSONB) and col.nullable is False


class TestAuditTable:
    def test_columns_closed(self):
        assert ExtensionAuditLogModel.__tablename__ == "extension_audit_log"
        assert set(_cols(ExtensionAuditLogModel)) == EXPECTED_AUDIT_COLUMNS

    def test_identity_snapshot_nullability(self):
        cols = _cols(ExtensionAuditLogModel)
        assert cols["extension_id"].nullable is True    # 行可能已软删/身份解析前拒绝
        assert cols["ext_id"].nullable is True          # R13#5 唯一例外场景
        assert cols["kind"].nullable is False
        assert cols["event"].nullable is False
        assert cols["actor_user_id"].nullable is True   # NULL=系统动作
