"""PE-1 §3.1 — SkillRiskRefreshResult typed status + Redis payload codec."""

from __future__ import annotations

import json

import pytest

from app.domain.services.permission.sources.skill_metadata import (
    SkillRiskRefreshResult,
)
from app.domain.services.risk_assessor import RiskLevel


class TestSkillRiskRefreshResultConstruction:
    def test_fresh_minimal(self):
        r = SkillRiskRefreshResult(status="fresh")
        assert r.status == "fresh"
        assert r.risk_level is None
        assert r.error is None

    def test_refreshed_with_risk_level(self):
        r = SkillRiskRefreshResult(status="refreshed", risk_level=RiskLevel.HIGH)
        assert r.status == "refreshed"
        assert r.risk_level == RiskLevel.HIGH

    def test_failed_with_error(self):
        r = SkillRiskRefreshResult(status="failed", error="OSError: ENOENT")
        assert r.status == "failed"
        assert r.error == "OSError: ENOENT"

    def test_unknown_with_error(self):
        r = SkillRiskRefreshResult(status="unknown", error="skill_dir_missing")
        assert r.status == "unknown"
        assert r.error == "skill_dir_missing"


class TestToRedisValueRoundtrip:
    def test_fresh_roundtrip(self):
        original = SkillRiskRefreshResult(status="fresh")
        payload = original.to_redis_value()
        parsed = json.loads(payload)
        assert parsed["status"] == "fresh"
        assert parsed["risk_level"] is None
        assert parsed["error"] is None
        assert parsed["version"] == 1

        restored = SkillRiskRefreshResult.from_redis_value(payload)
        assert restored.status == original.status
        assert restored.risk_level == original.risk_level
        assert restored.error == original.error

    def test_refreshed_with_risk_roundtrip(self):
        original = SkillRiskRefreshResult(
            status="refreshed", risk_level=RiskLevel.MEDIUM
        )
        payload = original.to_redis_value()
        restored = SkillRiskRefreshResult.from_redis_value(payload)
        assert restored.status == "refreshed"
        assert restored.risk_level == RiskLevel.MEDIUM

    def test_failed_roundtrip(self):
        original = SkillRiskRefreshResult(status="failed", error="redis_outage")
        payload = original.to_redis_value()
        restored = SkillRiskRefreshResult.from_redis_value(payload)
        assert restored.status == "failed"
        assert restored.error == "redis_outage"

    def test_unknown_roundtrip(self):
        original = SkillRiskRefreshResult(status="unknown", error="hash_unresolvable")
        payload = original.to_redis_value()
        restored = SkillRiskRefreshResult.from_redis_value(payload)
        assert restored.status == "unknown"
        assert restored.error == "hash_unresolvable"

    def test_from_redis_value_accepts_bytes(self):
        original = SkillRiskRefreshResult(status="fresh")
        payload = original.to_redis_value()
        restored = SkillRiskRefreshResult.from_redis_value(payload.encode("utf-8"))
        assert restored.status == "fresh"


class TestFromRedisValueMalformedFailClosed:
    """Round 3 P2#7 hard rule: never raise on garbage Redis data;
    return status="failed" so caller goes force-HIGH (defense-in-depth)."""

    def test_invalid_json_returns_failed(self):
        r = SkillRiskRefreshResult.from_redis_value("not-json{{")
        assert r.status == "failed"
        assert r.error and r.error.startswith("malformed_redis_payload:")

    def test_missing_status_returns_failed(self):
        bad = json.dumps({"risk_level": "HIGH", "version": 1})
        r = SkillRiskRefreshResult.from_redis_value(bad)
        assert r.status == "failed"
        assert "malformed_redis_payload" in (r.error or "")

    def test_invalid_status_string_returns_failed(self):
        bad = json.dumps({"status": "approved", "version": 1})
        r = SkillRiskRefreshResult.from_redis_value(bad)
        assert r.status == "failed"
        assert r.error and "invalid_status:approved" in r.error

    def test_unsupported_version_returns_failed(self):
        bad = json.dumps({"status": "fresh", "version": 99})
        r = SkillRiskRefreshResult.from_redis_value(bad)
        assert r.status == "failed"
        assert r.error and "unsupported_version:99" in r.error

    def test_missing_version_returns_failed(self):
        bad = json.dumps({"status": "fresh"})
        r = SkillRiskRefreshResult.from_redis_value(bad)
        assert r.status == "failed"
        # missing version → version=None → unsupported_version
        assert r.error and "unsupported_version:None" in r.error

    def test_invalid_risk_level_name_returns_failed(self):
        bad = json.dumps({"status": "refreshed", "risk_level": "WAT", "version": 1})
        r = SkillRiskRefreshResult.from_redis_value(bad)
        assert r.status == "failed"
        assert "malformed_redis_payload" in (r.error or "")


# ---------- T7: build_skill_call_metadata helper ----------

from unittest.mock import MagicMock

from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType
from app.domain.services.permission.source_metadata import SkillCallMetadata
from app.domain.services.permission.sources.skill_metadata import (
    build_skill_call_metadata,
)


def _make_skill(skill_id: str, content_hash: str | None = "sha256:deadbeef") -> Skill:
    return Skill(
        id=skill_id,
        slug="t7-fixture",
        name=skill_id,
        source_type=SkillSourceType.LOCAL,
        source_ref="local:t7",
        manifest={"tools": [], "policy": {}},
        runtime_type=SkillRuntimeType.NATIVE,
        scan_report=(
            {"verdict": "safe", "content_hash": content_hash}
            if content_hash is not None
            else {"verdict": "safe"}
        ),
        trust_origin="user_installed",
        enabled=True,
    )


def _make_binding(skill: Skill, final_risk: str = "high", scan_verdict: str = "safe"):
    return {
        "skill": skill,
        "runtime_type": SkillRuntimeType.NATIVE,
        "manifest_tool": {},
        "final_risk": final_risk,
        "trust_origin": skill.trust_origin,
        "scan_verdict": scan_verdict,
    }


class TestBuildSkillCallMetadata:
    def test_extracts_all_canonical_fields(self):
        skill = _make_skill("sk_abc")
        binding = _make_binding(skill, final_risk="high", scan_verdict="safe")

        skill_tool = MagicMock()
        skill_tool._tool_bindings = {"myskill_run": binding}

        tool_fn = MagicMock()
        tool_fn.metadata = {"risk_level": "stale_value", "runtime_type": "native"}

        meta = build_skill_call_metadata(
            tool_name="myskill_run",
            tool_fn=tool_fn,
            skill_tool=skill_tool,
        )

        assert isinstance(meta, SkillCallMetadata)
        assert meta.tool_name == "myskill_run"
        assert meta.skill_id == "sk_abc"
        assert meta.content_hash == "sha256:deadbeef"
        # canonical rule: risk_level from binding final_risk, NOT tool_fn.metadata
        assert meta.risk_level == RiskLevel.HIGH
        assert meta.runtime_type == SkillRuntimeType.NATIVE
        assert meta.trust_origin == "user_installed"
        assert meta.scan_verdict == "safe"

    def test_handles_missing_content_hash(self):
        skill = _make_skill("sk_no_hash", content_hash=None)
        binding = _make_binding(skill, final_risk="low")
        skill_tool = MagicMock()
        skill_tool._tool_bindings = {"t1": binding}
        tool_fn = MagicMock()
        tool_fn.metadata = {}

        meta = build_skill_call_metadata(
            tool_name="t1", tool_fn=tool_fn, skill_tool=skill_tool,
        )
        assert meta.content_hash is None
        assert meta.risk_level == RiskLevel.LOW

    def test_does_not_read_risk_from_tool_fn_metadata(self):
        """Canonical rule (spec Round 2 P0#3): risk MUST come from
        _tool_bindings final_risk, NEVER tool_fn.metadata. If binding says
        HIGH and tool_fn.metadata says LOW, we get HIGH."""
        skill = _make_skill("sk")
        binding = _make_binding(skill, final_risk="high", scan_verdict="dangerous")
        skill_tool = MagicMock()
        skill_tool._tool_bindings = {"t1": binding}
        tool_fn = MagicMock()
        tool_fn.metadata = {"risk_level": "low"}  # ← tries to mislead

        meta = build_skill_call_metadata(
            tool_name="t1", tool_fn=tool_fn, skill_tool=skill_tool,
        )
        assert meta.risk_level == RiskLevel.HIGH
        assert meta.scan_verdict == "dangerous"

    def test_raises_on_missing_binding(self):
        skill_tool = MagicMock()
        skill_tool._tool_bindings = {}
        tool_fn = MagicMock()

        with pytest.raises(KeyError):
            build_skill_call_metadata(
                tool_name="absent",
                tool_fn=tool_fn,
                skill_tool=skill_tool,
            )

    def test_raises_on_unknown_risk_level_string(self):
        skill = _make_skill("sk")
        binding = _make_binding(skill, final_risk="catastrophic")
        skill_tool = MagicMock()
        skill_tool._tool_bindings = {"t1": binding}
        tool_fn = MagicMock()

        with pytest.raises(KeyError):
            build_skill_call_metadata(
                tool_name="t1",
                tool_fn=tool_fn,
                skill_tool=skill_tool,
            )
