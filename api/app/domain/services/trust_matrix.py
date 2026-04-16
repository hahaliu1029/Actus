"""三层风险界定模型 + scan_skill_source() + INSTALL_POLICY"""

from __future__ import annotations

import logging
from pathlib import Path

from app.domain.models.skill import SkillRuntimeType
from app.domain.services.risk_assessor import RiskLevel
from app.domain.services.skills_guard import SkillsGuard, ScanReport

logger = logging.getLogger(__name__)

# ---------- Layer 1: base_floor matrix ----------

_RUNTIME_ORIGIN_MATRIX: dict[SkillRuntimeType, dict[str, RiskLevel]] = {
    SkillRuntimeType.NATIVE: {
        "builtin": RiskLevel.NONE,
        "user_installed": RiskLevel.LOW,
        "agent_created": RiskLevel.MEDIUM,
    },
    SkillRuntimeType.MCP: {
        "builtin": RiskLevel.LOW,
        "user_installed": RiskLevel.MEDIUM,
        "agent_created": RiskLevel.MEDIUM,
    },
    SkillRuntimeType.A2A: {
        "builtin": RiskLevel.MEDIUM,
        "user_installed": RiskLevel.MEDIUM,
        "agent_created": RiskLevel.HIGH,
    },
}


def compute_base_floor(
    runtime_type: SkillRuntimeType,
    trust_origin: str,
) -> RiskLevel:
    origin_map = _RUNTIME_ORIGIN_MATRIX.get(runtime_type, {})
    return origin_map.get(trust_origin, RiskLevel.MEDIUM)


# ---------- Layer 2: scan_floor ----------

_SCAN_VERDICT_MAP: dict[str, RiskLevel] = {
    "safe": RiskLevel.NONE,
    "caution": RiskLevel.MEDIUM,
    "dangerous": RiskLevel.HIGH,
}


# ---------- Layer 3: manifest_floor ----------

_MANIFEST_RISK_MAP: dict[str, RiskLevel] = {
    "none": RiskLevel.NONE,
    "low": RiskLevel.LOW,
    "medium": RiskLevel.MEDIUM,
    "high": RiskLevel.HIGH,
}


def compute_final_risk(
    base_floor: RiskLevel,
    scan_verdict: str | None,
    manifest_risk_level: str | None,
) -> RiskLevel:
    scan_floor = _SCAN_VERDICT_MAP.get(scan_verdict or "safe", RiskLevel.NONE)
    manifest_floor = _MANIFEST_RISK_MAP.get(
        (manifest_risk_level or "none").strip().lower(), RiskLevel.NONE
    )
    return max(base_floor, scan_floor, manifest_floor)


# ---------- INSTALL_POLICY ----------

INSTALL_POLICY: dict[str, tuple[str, str, str]] = {
    #                    safe      caution    dangerous
    "builtin":         ("allow",  "allow",   "allow"),
    "user_installed":  ("allow",  "warn",    "block"),
    "agent_created":   ("allow",  "allow",   "block"),
}

_VERDICT_INDEX = {"safe": 0, "caution": 1, "dangerous": 2}


def get_install_decision(trust_origin: str, verdict: str) -> str:
    policy = INSTALL_POLICY.get(trust_origin, INSTALL_POLICY["user_installed"])
    idx = _VERDICT_INDEX.get(verdict, 2)  # unknown → dangerous
    return policy[idx]


# ---------- scan_skill_source() unified entry ----------

_default_guard = SkillsGuard()


def scan_skill_source(
    runtime_type: SkillRuntimeType,
    bundle_dir: Path,
    guard: SkillsGuard | None = None,
) -> ScanReport:
    g = guard or _default_guard
    if runtime_type == SkillRuntimeType.A2A:
        return g.scan_limited(bundle_dir)
    return g.scan(bundle_dir)
