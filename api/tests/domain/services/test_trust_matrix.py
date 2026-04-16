import pytest
from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType


def test_skill_trust_origin_defaults_to_user_installed():
    skill = Skill(
        id="test-id",
        slug="test",
        name="Test",
        description="test",
        source_type=SkillSourceType.LOCAL,
        source_ref="local:/tmp/test",
        runtime_type=SkillRuntimeType.NATIVE,
    )
    assert skill.trust_origin == "user_installed"
    assert skill.scan_report is None


def test_base_floor_native_user_installed():
    from app.domain.services.trust_matrix import compute_base_floor
    from app.domain.services.risk_assessor import RiskLevel
    assert compute_base_floor(SkillRuntimeType.NATIVE, "user_installed") == RiskLevel.LOW


def test_base_floor_a2a_agent_created():
    from app.domain.services.trust_matrix import compute_base_floor
    from app.domain.services.risk_assessor import RiskLevel
    assert compute_base_floor(SkillRuntimeType.A2A, "agent_created") == RiskLevel.HIGH


def test_final_risk_only_escalates():
    from app.domain.services.trust_matrix import compute_final_risk
    from app.domain.services.risk_assessor import RiskLevel
    result = compute_final_risk(
        base_floor=RiskLevel.LOW,
        scan_verdict="dangerous",
        manifest_risk_level=None,
    )
    assert result == RiskLevel.HIGH


def test_manifest_cannot_lower_risk():
    from app.domain.services.trust_matrix import compute_final_risk
    from app.domain.services.risk_assessor import RiskLevel
    result = compute_final_risk(
        base_floor=RiskLevel.HIGH,
        scan_verdict="safe",
        manifest_risk_level="low",
    )
    assert result == RiskLevel.HIGH


def test_install_policy_user_dangerous_is_block():
    from app.domain.services.trust_matrix import get_install_decision
    assert get_install_decision("user_installed", "dangerous") == "block"


def test_install_policy_user_caution_is_warn():
    from app.domain.services.trust_matrix import get_install_decision
    assert get_install_decision("user_installed", "caution") == "warn"


def test_install_policy_builtin_dangerous_is_allow():
    from app.domain.services.trust_matrix import get_install_decision
    assert get_install_decision("builtin", "dangerous") == "allow"


def test_missing_scan_report_yields_dangerous_final_risk():
    """scan_report=None → _compute_final_risk treats as dangerous (fail-closed)"""
    from app.domain.services.tools.skill import SkillTool
    skill = Skill(
        id="legacy", slug="legacy", name="Legacy",
        source_type=SkillSourceType.LOCAL, source_ref="local:/tmp",
        runtime_type=SkillRuntimeType.NATIVE,
        scan_report=None,
    )
    risk = SkillTool._compute_final_risk(skill)
    assert risk == "high", f"expected 'high' for missing scan_report, got '{risk}'"


import json
from pathlib import Path


def test_meta_json_roundtrip_with_scan_fields(tmp_path):
    """meta.json 写入 trust_origin + scan_report 后能正确读回"""
    from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType

    meta = {
        "id": "test-id", "slug": "test", "name": "Test",
        "description": "test", "version": "1.0.0",
        "source_type": "local", "source_ref": "",
        "runtime_type": "native", "enabled": True,
        "installed_by": "admin", "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "trust_origin": "agent_created",
        "scan_report": {
            "verdict": "caution",
            "content_hash": "sha256:abc123",
            "scanned_at": "2026-04-16T10:00:00",
            "findings": [{"pattern_id": "test", "category": "execution",
                          "severity": "high", "file": "test.py",
                          "line": 1, "match": "os.system()"}],
        },
    }
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps(meta))
    (tmp_path / "manifest.json").write_text("{}")

    skill = Skill(
        id=meta["id"], slug=meta["slug"], name=meta["name"],
        description=meta["description"],
        source_type=SkillSourceType.LOCAL, source_ref="",
        runtime_type=SkillRuntimeType.NATIVE,
        trust_origin=meta["trust_origin"],
        scan_report=meta["scan_report"],
    )
    assert skill.trust_origin == "agent_created"
    assert skill.scan_report["verdict"] == "caution"
