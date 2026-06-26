"""C5a infra sink — one canonical-JSON INFO line per snapshot.

Best-effort + non-suspending (no await / no I/O offload), protecting INV-0
(§6 R7#1, §8.9). Never raises into the caller. Mirrors the `mailbox.telemetry`
precedent (log directly with a stable greppable prefix).
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.domain.models.sandbox_policy import SandboxPolicySnapshot

logger = logging.getLogger("sandbox.policy")


class LoggingPolicySnapshotSink:
    async def record(self, snapshot: "SandboxPolicySnapshot") -> None:  # noqa: RUF029
        try:
            payload = json.dumps(
                snapshot.model_dump(mode="json"), ensure_ascii=False, sort_keys=True
            )
            logger.info("sandbox.policy %s", payload)
        except Exception as exc:  # noqa: BLE001 — observe must never break the hot path
            logger.warning(
                "sandbox.policy observe render failed surface=%s exc=%s",
                getattr(snapshot, "surface", "?"),
                type(exc).__name__,
            )
