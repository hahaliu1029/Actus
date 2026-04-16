import pytest
from app.domain.models.skill import SkillRuntimeType
from app.domain.services.risk_assessor import RiskLevel


def test_assess_skill_returns_risk_assessment():
    from app.domain.services.skill_risk_assessor import SkillRiskAssessor
    assessor = SkillRiskAssessor()
    result = assessor.assess(
        tool_name="skill_deploy_run",
        tool_args={"env": "staging", "force": True},
        risk_level=RiskLevel.HIGH,
        runtime_type=SkillRuntimeType.NATIVE,
        trust_origin="user_installed",
    )
    assert result.final_level == RiskLevel.HIGH
    assert result.arg_digest  # non-empty


def test_arg_digest_hash_all_for_high():
    from app.domain.services.skill_risk_assessor import SkillRiskAssessor
    assessor = SkillRiskAssessor()
    r1 = assessor.assess("t", {"a": "x"}, RiskLevel.HIGH, SkillRuntimeType.NATIVE, "user_installed")
    r2 = assessor.assess("t", {"a": "y"}, RiskLevel.HIGH, SkillRuntimeType.NATIVE, "user_installed")
    assert r1.arg_digest != r2.arg_digest  # hash-all → different args → different digest


def test_arg_digest_tool_level_for_native_medium():
    from app.domain.services.skill_risk_assessor import SkillRiskAssessor
    assessor = SkillRiskAssessor()
    r1 = assessor.assess("t", {"a": "x"}, RiskLevel.MEDIUM, SkillRuntimeType.NATIVE, "user_installed")
    r2 = assessor.assess("t", {"a": "y"}, RiskLevel.MEDIUM, SkillRuntimeType.NATIVE, "user_installed")
    assert r1.arg_digest == r2.arg_digest  # tool-level → same tool → same digest


def test_primary_arg_hashes_all_strings():
    from app.domain.services.skill_risk_assessor import _extract_skill_primary_arg
    result = _extract_skill_primary_arg({"z_param": "dangerous", "a_param": "safe", "count": 5})
    assert "dangerous" in result
    assert "safe" in result
