"""RecoveryEvent telemetry record (PR-3)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.domain.services.provider_profiles._base import ErrorClass


Outcome = Literal["retry_sent", "rule_missed", "budget_exhausted", "success"]


@dataclass(frozen=True)
class RecoveryEvent:
    call_id: str
    attempt_index: int
    provider_id: str
    api_mode: str
    model_name: str
    error_class: ErrorClass | None
    fingerprint_code: str | None
    action_code: str | None
    rewrite_applied_keys: tuple[str, ...]
    outcome: Outcome
    latency_ms: int
