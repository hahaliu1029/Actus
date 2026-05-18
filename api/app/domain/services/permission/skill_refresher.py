"""SkillRiskRefresher — hash-aware wrapper around SkillTool.refresh_risk_if_stale.

Owns content_hash comparison itself so unchanged skills return "fresh"
(cached risk_level valid) instead of being force-HIGH on every PE call
(Round 1 P0#1).

Uses ``SkillTool.resolve_skill_dir(tool_name)`` public port (Round 3 P1#5)
instead of reaching across two classes' private attrs.

Spec §3.1.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from app.domain.services.permission.sources.skill_metadata import (
    SkillRiskRefreshResult,
)
from app.domain.services.risk_assessor import RiskLevel
from app.domain.services.skills_guard import SkillsGuard

if TYPE_CHECKING:
    from app.domain.services.tools.skill import SkillTool

logger = logging.getLogger(__name__)


class SkillRiskRefresher:
    """Hash-aware refresh with typed result.

    Status semantics:
      - "fresh":     on-disk hash == expected_content_hash; cached risk_level is valid.
      - "refreshed": on-disk hash differs; ``risk_level`` carries the NEW (rescanned) value.
      - "unknown":   binding / bundle_sync_manager / skills_root / skill_dir state
                     prevents proving freshness — caller treats as force-HIGH.
      - "failed":    IO / scan exception or other thrown error during refresh.

    NOT thread-safe at the SkillTool binding level (in-place mutation in
    ``refresh_risk_if_stale``); singleflight in ``SkillSource`` ensures only
    one refresh per (skill_id, content_hash) tuple runs at a time.
    """

    def __init__(self, skill_tool: "SkillTool"):
        self._skill_tool = skill_tool

    async def refresh(
        self,
        tool_name: str,
        expected_content_hash: str | None,
    ) -> SkillRiskRefreshResult:
        try:
            skill_dir = self._skill_tool.resolve_skill_dir(tool_name)
            if skill_dir is None:
                return SkillRiskRefreshResult(
                    status="unknown", error="skill_dir_unresolvable"
                )
            if not skill_dir.exists():
                return SkillRiskRefreshResult(
                    status="unknown", error="skill_dir_missing"
                )

            current_hash = SkillsGuard.compute_content_hash(skill_dir)
            if current_hash is None:
                return SkillRiskRefreshResult(
                    status="unknown", error="cannot_compute_hash"
                )

            if current_hash == expected_content_hash:
                return SkillRiskRefreshResult(status="fresh")

            # Hash changed — delegate to existing refresh_risk_if_stale,
            # which rescans + updates binding["final_risk"] in place
            # (api/app/domain/services/tools/skill.py:206).
            raw = self._skill_tool.refresh_risk_if_stale(tool_name)
            if raw is None:
                return SkillRiskRefreshResult(
                    status="unknown",
                    error="refresh_returned_none_after_hash_diff",
                )
            return SkillRiskRefreshResult(
                status="refreshed",
                risk_level=RiskLevel[raw.upper()],
            )
        except (OSError, ValueError, KeyError, AttributeError) as exc:
            return SkillRiskRefreshResult(status="failed", error=str(exc))
        except Exception as exc:  # noqa: BLE001 — defense-in-depth boundary
            logger.warning(
                "SkillRiskRefresher.refresh unexpected exception "
                "for tool=%s expected_hash=%s: %r",
                tool_name, expected_content_hash, exc,
            )
            return SkillRiskRefreshResult(
                status="failed", error=f"unexpected:{exc!r}"
            )
