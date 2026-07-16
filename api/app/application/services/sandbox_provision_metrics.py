"""SPM PR-1c Task 17: in-process sandbox-provision metrics.

Mirrors the lightweight style of ``domain/services/execution_metrics.py`` —
plain in-process counters + structured log lines, NO prometheus dependency
(spec §5.9; observability is exported later from the logs / snapshot). A single
instance is pinned on ``app.state.sandbox_provision_metrics`` (main.py) and
injected into:

* the four NON-provisioner trigger surfaces — ``AgentService`` (``run_start``),
  ``SessionService`` (``vnc`` / ``takeover``), the coordinator child-runner
  starter (``child_spawn``);
* the ``SandboxProvisioner`` on_demand provision flow (``tool_call`` /
  ``skill_sync``) and the ``SandboxAttachmentFlusher`` (attachment skips).

Side-effect-only: every ``record_*`` method MUST NEVER raise — a metrics hiccup
must not mask the real provisioning error (the provisioner emits from inside its
exception handlers).

Four indicators are retained (§5.9, unchanged by the r10/R9-F6 "no independent
completion counter" decision): ``sandbox_provision_total{mode,trigger,outcome}``,
``sandbox_provision_latency_seconds``, ``attachment_skipped{reason}``,
``lingering_after_off`` (Task 28 off-startup residual). "Containers saved" is
derived elsewhere from the ``sandbox_lifecycle_log`` CREATING→ACTIVE rows, NOT
from these counters.
"""
from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class SandboxProvisionMetrics:
    """进程内 provision-flow 计数 + 结构化日志（§5.9；无 prometheus）。"""

    def __init__(self) -> None:
        # sandbox_provision_total{mode,trigger,outcome}
        self._provision_total: dict[tuple[str, str, str], int] = defaultdict(int)
        # sandbox_provision_latency_seconds — running sum + count per key
        self._latency_sum: dict[tuple[str, str, str], float] = defaultdict(float)
        self._latency_count: dict[tuple[str, str, str], int] = defaultdict(int)
        # attachment_skipped{reason}
        self._attachment_skipped: dict[str, int] = defaultdict(int)
        # last-recorded off-startup residual sandbox count (Task 28)
        self._lingering_after_off: int = 0

    def record_provision(
        self,
        *,
        mode: str,
        trigger: str,
        outcome: str,
        latency_seconds: float | None = None,
        latency: float | None = None,
    ) -> None:
        """Record one provision-flow outcome.

        ``latency_seconds`` is the canonical parameter (spec §5.9). ``latency``
        is accepted as a back-compat alias because the frozen Task 14
        ``SandboxProvisioner`` (its ``SandboxProvisionMetrics`` Protocol +
        ``_emit`` call site) passes ``latency=``; both name the same value.
        Side-effect-only — never raises.
        """
        lat = latency_seconds if latency_seconds is not None else latency
        try:
            key = (mode, trigger, outcome)
            self._provision_total[key] += 1
            if lat is not None:
                self._latency_sum[key] += float(lat)
                self._latency_count[key] += 1
            logger.info(
                "sandbox_provision mode=%s trigger=%s outcome=%s latency_s=%s",
                mode,
                trigger,
                outcome,
                round(float(lat), 4) if lat is not None else "-",
            )
        except Exception:  # noqa: BLE001 — observability must not break the flow
            logger.warning("record_provision emit failed", exc_info=True)

    def record_attachment_skipped(self, *, reason: str) -> None:
        try:
            self._attachment_skipped[reason] += 1
            logger.info("sandbox_provision attachment_skipped reason=%s", reason)
        except Exception:  # noqa: BLE001
            logger.warning("record_attachment_skipped emit failed", exc_info=True)

    def record_lingering_after_off(self, *, count: int) -> None:
        """r9/codex R8-G3: off-startup residual sandbox count (Task 28 caller)."""
        try:
            self._lingering_after_off = int(count)
            logger.info("sandbox_provision lingering_after_off count=%d", int(count))
        except Exception:  # noqa: BLE001
            logger.warning("record_lingering_after_off emit failed", exc_info=True)

    def snapshot(self) -> dict:
        """诊断/测试快照——含全部计数器（singleton 断言用）。

        Keys are stringified ``mode|trigger|outcome`` so the dict is trivially
        assertable and JSON-safe.
        """
        return {
            "provision_total": {
                "|".join(k): v for k, v in self._provision_total.items()
            },
            "latency_sum": {
                "|".join(k): round(v, 4) for k, v in self._latency_sum.items()
            },
            "latency_count": {
                "|".join(k): v for k, v in self._latency_count.items()
            },
            "attachment_skipped": dict(self._attachment_skipped),
            "lingering_after_off": self._lingering_after_off,
        }
