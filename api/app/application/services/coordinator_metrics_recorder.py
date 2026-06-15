"""[C2b rollout WS1b] Application-layer recorder for the 3 dead CoordinatorMetrics
instruments (run_cost_usd / duration_seconds / tool_calls).

Exists because the DOMAIN reducer must NOT hold the user_id_hash salt or read
Settings. Wraps the OTel CoordinatorMetrics bundle + the salt and records on
behalf of the reducer (run-level) and the invoke-adapter (per-child tool calls).
budget_exhaustion stays recorded directly in coordinator_child_runner (NG6).

Every record is best-effort (INV-B9 mirror): a telemetry failure logs at WARN
and never propagates — metrics must not break a coordinator run.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


def hash_user_id(user_id: str | None, salt: str) -> str | None:
    """[F12] sha256(salt + user_id).hexdigest()[:16]; empty salt OR empty
    user_id → None (the user_id_hash attribute is omitted — contract-legal)."""
    if not salt or not user_id:
        return None
    return hashlib.sha256((salt + user_id).encode("utf-8")).hexdigest()[:16]


class CoordinatorMetricsRecorder:
    """Thin app-layer wrapper around the CoordinatorMetrics OTel bundle.

    ``metrics`` is the CoordinatorMetrics instance (or None — every method
    no-ops when absent). ``user_id_hash_salt`` comes from Settings at the
    composition root.
    """

    def __init__(self, *, metrics: Any, user_id_hash_salt: str = "") -> None:
        self._metrics = metrics
        self._salt = user_id_hash_salt

    def record_run_terminal(
        self,
        *,
        coordinator_run_id: str,
        user_id: str | None,
        cost_usd: float,
        outcome: str,
        cost_authoritative: bool,
        duration_s: float | None,
    ) -> None:
        """Record run_cost_usd (only when cost is authoritative — never add(0),
        which would mask 'unknown' as 'zero', §3.6) + duration_seconds (when
        duration is known). Each best-effort."""
        if self._metrics is None:
            return
        if cost_authoritative:
            uid_hash = hash_user_id(user_id, self._salt)
            attrs: dict[str, Any] = {
                "coordinator_run_id": coordinator_run_id,
                "outcome": outcome,
            }
            if uid_hash is not None:
                attrs["user_id_hash"] = uid_hash
            try:
                self._metrics.run_cost_usd.add(cost_usd, attributes=attrs)
            except Exception:  # noqa: BLE001 — INV-B9 best-effort
                logger.warning(
                    "coordinator run_cost_usd.add failed (best-effort)", exc_info=True
                )
        if duration_s is not None:
            try:
                self._metrics.duration_seconds.record(
                    duration_s,
                    attributes={
                        "coordinator_run_id": coordinator_run_id,
                        "group_outcome": outcome,
                    },
                )
            except Exception:  # noqa: BLE001 — INV-B9 best-effort
                logger.warning(
                    "coordinator duration_seconds.record failed (best-effort)",
                    exc_info=True,
                )

    def record_tool_call(
        self, *, coordinator_run_id: str, work_unit_id: str, function_name: str
    ) -> None:
        """Record one tool_calls increment for a child CALLING ToolEvent."""
        if self._metrics is None:
            return
        try:
            self._metrics.tool_calls.add(
                1,
                attributes={
                    "coordinator_run_id": coordinator_run_id,
                    "work_unit_id": work_unit_id,
                    "tool_name": function_name,
                },
            )
        except Exception:  # noqa: BLE001 — INV-B9 best-effort
            logger.warning(
                "coordinator tool_calls.add failed (best-effort)", exc_info=True
            )
