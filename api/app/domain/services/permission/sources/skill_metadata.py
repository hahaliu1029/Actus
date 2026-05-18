"""SkillRiskRefreshResult + (in T7) build_skill_call_metadata helper.

This module also depends on LangChain types in T7 (StructuredTool) — kept
separate from no-cycle ``permission/source_metadata.py`` so source_metadata
can stay LangChain-free for static analysis.

Spec ref: §3.1 (SkillRiskRefreshResult typed status + Redis payload codec).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from app.domain.services.permission.source_metadata import (
    SkillCallMetadata,
    SourceMetadata,
)
from app.domain.services.risk_assessor import RiskLevel

__all__ = [
    "SkillCallMetadata",
    "SourceMetadata",
    "SkillRiskRefreshResult",
    "build_skill_call_metadata",
]

# Source of truth for the Redis payload version. Bump when adding fields
# to to_redis_value(). from_redis_value() rejects mismatched versions for
# back-compat safety (force-HIGH path).
_REDIS_PAYLOAD_VERSION = 1
_VALID_STATUSES = frozenset({"fresh", "refreshed", "unknown", "failed"})


@dataclass(frozen=True)
class SkillRiskRefreshResult:
    """Typed result with hash-aware "fresh" status.

    Status meaning:
      - "fresh":     on-disk hash == expected; cached metadata risk_level valid.
      - "refreshed": on-disk hash differs; ``risk_level`` carries NEW (rescanned) value.
      - "unknown":   binding / bundle / skill_dir state prevents proving freshness.
                     Cannot guarantee cached risk_level is valid → force-HIGH path.
      - "failed":    IO / scan / Redis exception, or loser-poll timeout.
                     Caller treats as force-HIGH.

    Spec ref: §3.1 SkillRiskRefreshResult; Round 1 P0#1 hash-aware status;
              Round 3 P2#7 strict from_redis_value validation.
    """

    status: Literal["fresh", "refreshed", "unknown", "failed"]
    risk_level: RiskLevel | None = None
    error: str | None = None

    def to_redis_value(self) -> str:
        """Round 2 P1#8 — Redis result_key payload (JSON, version-tagged).

        Forward compat: bump _REDIS_PAYLOAD_VERSION when adding fields;
        old readers will treat new payloads as "failed" (force-HIGH) instead
        of silently mis-parsing.
        """
        return json.dumps(
            {
                "status": self.status,
                "risk_level": self.risk_level.name if self.risk_level else None,
                "error": self.error,
                "version": _REDIS_PAYLOAD_VERSION,
            }
        )

    @classmethod
    def from_redis_value(cls, raw: str | bytes) -> "SkillRiskRefreshResult":
        """Parse Redis payload back to typed result. Fail-closed on malformed.

        Round 3 P2#7 strict validation:
          - Unsupported / missing version → failed.
          - status not in _VALID_STATUSES → failed.
          - JSON / Key / Value / Type errors → failed (NEVER raise).
          - Invalid risk_level name → failed (KeyError caught).

        Caller is on the fast-path (singleflight loser poll) and must not
        crash on garbage Redis data; force-HIGH is the safe fallback.
        """
        try:
            data = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
            version = data.get("version")
            if version != _REDIS_PAYLOAD_VERSION:
                return cls(status="failed", error=f"unsupported_version:{version}")
            status = data["status"]
            if status not in _VALID_STATUSES:
                return cls(status="failed", error=f"invalid_status:{status}")
            rl_name = data.get("risk_level")
            risk_level = RiskLevel[rl_name] if rl_name else None
            return cls(
                status=status,
                risk_level=risk_level,
                error=data.get("error"),
            )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            return cls(status="failed", error=f"malformed_redis_payload:{exc}")


def build_skill_call_metadata(
    tool_name: str,
    tool_fn: Any,           # LangChain StructuredTool — ONLY this helper touches it
    skill_tool: Any,        # SkillTool — typed as Any to avoid heavy import in PE layer
) -> SkillCallMetadata:
    """Single extraction point for LangChain ``tool.metadata`` + ``_tool_bindings``.

    Canonical risk source rule (spec Round 2 P0#3, HARD):
      - risk_level    → ``skill_tool._tool_bindings[tool_name]["final_risk"]``
                        (in-place updated by ``refresh_risk_if_stale``)
      - runtime_type  → ``skill_tool._tool_bindings[tool_name]["runtime_type"]``
      - trust_origin  → ``skill_tool._tool_bindings[tool_name]["trust_origin"]``
      - scan_verdict  → ``skill_tool._tool_bindings[tool_name]["scan_verdict"]``
      - skill_id      → ``skill_tool._tool_bindings[tool_name]["skill"].id``
      - content_hash  → ``skill_tool._tool_bindings[tool_name]["skill"].scan_report.get("content_hash")``
      - tool_name     → input param

    ``tool_fn.metadata`` is a build-time SNAPSHOT — it goes stale after
    refresh. DO NOT read risk_level / runtime_type / trust_origin /
    scan_verdict from ``tool_fn.metadata``. The grep gate (INV-6) treats
    direct reads outside this helper as bypass.

    Raises KeyError if binding missing (caller bug — gate helper should
    have prevented routing to skill source for unknown tool_name).
    """
    binding = skill_tool._tool_bindings[tool_name]  # KeyError if absent (intentional)
    skill = binding["skill"]
    scan_report = skill.scan_report or {}
    return SkillCallMetadata(
        tool_name=tool_name,
        skill_id=skill.id,
        content_hash=scan_report.get("content_hash"),
        risk_level=RiskLevel[binding["final_risk"].upper()],
        runtime_type=binding["runtime_type"],
        trust_origin=binding["trust_origin"],
        scan_verdict=binding["scan_verdict"],
    )
