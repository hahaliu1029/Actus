"""D1a — INV-D1-1 治理词表单权威契约测试（spec §2/§11.2 D1-1）。"""
from typing import get_args

from app.domain.models.extension_governance import (
    ADMINISTRATIVE_REASONS,
    DETECTION_REASONS,
    HASH_SCHEMA_VERSION,
    NEUTRAL_REASONS,
    SOURCE_TYPES,
    TRUST_ORIGINS,
    AdmissionDecisionReason,
    ExtensionAuditEvent,
    ExtensionStatus,
    GovernanceMode,
    GovernanceScanFinding,
    GovernanceScanSummary,
    GovernedExtensionKind,
    PinApprovalOutcome,
    QuarantineReason,
    SyncOutcome,
)
from app.domain.models.runtime_extension import ExtensionKind

# spec §2 词表字面复刻（测试与实现双写同一张表 = 抄错即红）
EXPECTED_STATUS = {"active", "quarantined", "disabled", "deleted"}
EXPECTED_QUARANTINE_REASON = {"pin_mismatch", "admin_manual"}
EXPECTED_MODE = {"off", "shadow", "enforce"}
EXPECTED_AUDIT_EVENTS = {
    "installed", "install_rejected", "uninstalled", "config_changed",
    "enabled", "disabled", "quarantined", "reapproved",
    "pin_established", "pin_mismatch", "config_drift_detected",
    "observed_first", "acknowledged",
    "scan_recorded", "force_installed",
    "reconciled_seen", "source_missing", "source_restored", "reconciled_missing",
    "plugin_expand_started", "plugin_expand_completed", "plugin_expand_compensated",
}
EXPECTED_REASONS = {
    "ok", "unknown", "unpinned", "pin_stale", "quarantined", "disabled",
    "deleted", "parent_blocked", "config_drift", "pin_mismatch",
    "registry_unavailable", "mode_off",
}
EXPECTED_SYNC_OUTCOME = {
    "uploaded", "already_current", "no_bundle",
    "r3_rejected_old_bundle", "governance_rejected",
}
EXPECTED_PIN_APPROVAL = {
    "pinned", "skipped_no_observation", "skipped_invalid_state", "conflict",
}


class TestKindVocabulary:
    def test_extension_kind_four_values(self):
        # spec §9.1：ExtensionKind 扩为四值（runtime_extension.py 权威）
        assert set(get_args(ExtensionKind)) == {"mcp", "a2a", "skill", "plugin"}

    def test_governed_kind_is_alias_not_second_definition(self):
        # INV-D1-1：别名同一性——不是第二个 Literal 定义
        assert GovernedExtensionKind is ExtensionKind


class TestGovernanceVocabularies:
    def test_status(self):
        assert set(get_args(ExtensionStatus)) == EXPECTED_STATUS

    def test_quarantine_reason(self):
        # R46#7：scan_dangerous 已删——两值封闭
        assert set(get_args(QuarantineReason)) == EXPECTED_QUARANTINE_REASON

    def test_mode(self):
        assert set(get_args(GovernanceMode)) == EXPECTED_MODE

    def test_audit_events(self):
        assert set(get_args(ExtensionAuditEvent)) == EXPECTED_AUDIT_EVENTS
        assert len(get_args(ExtensionAuditEvent)) == 22

    def test_admission_reasons(self):
        assert set(get_args(AdmissionDecisionReason)) == EXPECTED_REASONS
        assert len(get_args(AdmissionDecisionReason)) == 12

    def test_sync_outcome(self):
        assert set(get_args(SyncOutcome)) == EXPECTED_SYNC_OUTCOME

    def test_pin_approval_outcome(self):
        assert set(get_args(PinApprovalOutcome)) == EXPECTED_PIN_APPROVAL

    def test_hash_schema_version(self):
        assert HASH_SCHEMA_VERSION == 1


class TestReasonTrichotomy:
    """spec §2 R33#1：12 值三分归类穷尽且两两不交（INV-D1-1 snapshot）。"""

    def test_partition_is_exhaustive_and_disjoint(self):
        assert NEUTRAL_REASONS == frozenset({"ok", "mode_off"})
        assert DETECTION_REASONS == frozenset({
            "unknown", "unpinned", "pin_stale", "config_drift",
            "pin_mismatch", "registry_unavailable",
        })
        assert ADMINISTRATIVE_REASONS == frozenset({
            "quarantined", "disabled", "deleted", "parent_blocked",
        })
        union = NEUTRAL_REASONS | DETECTION_REASONS | ADMINISTRATIVE_REASONS
        assert union == set(get_args(AdmissionDecisionReason))
        assert NEUTRAL_REASONS.isdisjoint(DETECTION_REASONS)
        assert NEUTRAL_REASONS.isdisjoint(ADMINISTRATIVE_REASONS)
        assert DETECTION_REASONS.isdisjoint(ADMINISTRATIVE_REASONS)


class TestConventionSets:
    def test_trust_origins(self):
        # F13 惯例三值（str 域非 Literal——skill 侧自由 str 现状不变）
        assert TRUST_ORIGINS == frozenset({"builtin", "user_installed", "agent_created"})

    def test_source_types(self):
        # §3.1：含 mcp_registry deprecated 历史值（skill.py:13-18 直透传，R47#9）
        assert SOURCE_TYPES == frozenset({
            "local", "github", "mcp_registry", "config", "generated", "plugin",
        })


class TestGovernanceScanSummary:
    """§3.1 R6#8：固定字段、findings ≤50、字段长度 ≤256、禁原始 match 文本（结构层）。"""

    def test_valid_summary_roundtrip(self):
        s = GovernanceScanSummary(
            verdict="caution",
            finding_count=1,
            findings=[GovernanceScanFinding(
                category="injection", severity="medium",
                pattern_id="inj-001", path="tools/search", line=12,
            )],
        )
        assert s.model_dump()["findings"][0]["category"] == "injection"

    def test_findings_capped_at_50(self):
        import pytest
        from pydantic import ValidationError
        f = GovernanceScanFinding(
            category="c", severity="s", pattern_id="p", path="x", line=None,
        )
        with pytest.raises(ValidationError):
            GovernanceScanSummary(verdict="safe", finding_count=51, findings=[f] * 51)

    def test_field_length_capped_at_256(self):
        import pytest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            GovernanceScanFinding(
                category="x" * 257, severity="s", pattern_id="p", path="x", line=None,
            )

    def test_no_raw_match_field_exists(self):
        # 禁存原始 match 文本：schema 层根本没有该字段
        assert "match" not in GovernanceScanFinding.model_fields
        assert "matched_text" not in GovernanceScanFinding.model_fields
