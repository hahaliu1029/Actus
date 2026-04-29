"""JSONL-backed implementation of PromptTelemetryPort.

B5 C1: writes telemetry events as JSON lines to a configurable directory.
Failures are logged and swallowed — telemetry must never propagate errors
to the prompt assembly hot path.

C1 only ships the implementation. DI wiring (constructing the instance and
passing it to PromptAssembler) happens in C5a.

B5 PR-S1-6 (A6): every emitted JSONL row now carries the canonical
join keys ``trace_id`` / ``request_id`` / ``session_id`` read from
the per-request ``TraceContext`` carrier. Schema is backward
compatible — old keys keep their position and value; the three new
keys are appended at the tail of every payload. Sprint 2 ships the
real OTel spans with the same join keys, so an analyst can ``LEFT
JOIN`` the JSONL files against span attributes via ``trace_id``.

After Sprint 2, this writer schema is **FROZEN** — any new field
goes on OTel attributes, not into the JSONL files (avoid divergent
sources of truth on the same metric).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from app.domain.external.telemetry import PromptTelemetryPort

if TYPE_CHECKING:
    from app.domain.services.recovery._event import RecoveryEvent

logger = logging.getLogger(__name__)


def _trace_fields() -> dict[str, str | None]:
    """Read canonical trace/request/session join keys for a JSONL row.

    Per the v1 canonical attribute contract (PR-S1-1) ``trace_id``
    and ``request_id`` are **NEVER null** on any emit — null values
    would silently fail downstream ``validate_attributes`` checks
    and leave 100% of CLI / scheduler emissions un-joinable against
    future OTel span attributes.

    Delegates to ``build_canonical_attributes()`` so the "no
    request scope" fallback (``uuid4().hex`` for trace_id, ``str(
    uuid4())`` for request_id) is shared with every other canonical
    emit site — single source of truth for fallback semantics.
    ``session_id`` may be ``None`` here (the canonical contract
    marks it optional).

    Defensive: telemetry must never propagate errors to the
    prompt-assembly hot path (port contract). The outer
    ``try/except`` guards against boot-time circular import or any
    future change to ``build_canonical_attributes`` that could
    raise — last-resort path generates the same uuid4 shapes
    locally so the contract still holds.
    """
    try:
        from app.domain.external.observability import build_canonical_attributes

        attrs = build_canonical_attributes()
        return {
            "trace_id": attrs["trace_id"],
            "request_id": attrs["request_id"],
            "session_id": attrs.get("session_id"),
        }
    except Exception:
        import uuid as _uuid

        return {
            "trace_id": _uuid.uuid4().hex,
            "request_id": str(_uuid.uuid4()),
            "session_id": None,
        }


class JsonlPromptTelemetry(PromptTelemetryPort):
    """Append-only JSONL writer for prompt assembly, LLM invocation, and
    B2 Recovery telemetry.

    Three log files (created on demand under ``log_dir``):
    - ``assembly.jsonl`` — one record per ``PromptAssembler.assemble`` call
    - ``llm_invocation.jsonl`` — one record per LLM adapter invoke (C11)
    - ``recovery_event.jsonl`` — one record per B2 ``RecoveryEvent`` emit (Round 22 P1 #1)

    All writes are best-effort. On failure the error is logged at WARN
    level and the call returns normally — never raises.
    """

    def __init__(self, log_dir: Path | str):
        self._log_dir = Path(log_dir)
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "JsonlPromptTelemetry: failed to create log dir %s: %s",
                self._log_dir,
                exc,
            )

    # ---- PromptTelemetryPort implementation ------------------------------ #

    def record_assembly(
        self,
        *,
        sections_included: list[str],
        sections_dropped: list[str],
        tokens_used: int,
        lang: str,
        provider: str,
        mode: str,
        version_hash: str,
        fallback_used: bool = False,
    ) -> None:
        self._append(
            "assembly.jsonl",
            {
                "ts": _now_iso(),
                "sections_included": sections_included,
                "sections_dropped": sections_dropped,
                "tokens_used": tokens_used,
                "lang": lang,
                "provider": provider,
                "mode": mode,
                "version_hash": version_hash,
                "fallback_used": fallback_used,
                # B5 PR-S1-6: canonical join keys appended at tail —
                # old positions preserved for backward-compat consumers.
                **_trace_fields(),
            },
        )

    def record_llm_invocation(
        self,
        *,
        system_prompt_hash: str,
        system_prompt_bytes: int,
        tools_hash: str,
        lang: str,
        provider: str,
    ) -> None:
        self._append(
            "llm_invocation.jsonl",
            {
                "ts": _now_iso(),
                "system_prompt_hash": system_prompt_hash,
                "system_prompt_bytes": system_prompt_bytes,
                "tools_hash": tools_hash,
                "lang": lang,
                "provider": provider,
                **_trace_fields(),
            },
        )

    def emit_recovery_event(self, event: "RecoveryEvent") -> None:
        """Append a RecoveryEvent to recovery_event.jsonl. Best-effort.

        Round 22 P1 #1: implements the new PromptTelemetryPort hook so B2
        Recovery telemetry actually lands on disk in production.
        """
        self._append(
            "recovery_event.jsonl",
            {
                "ts": _now_iso(),
                "call_id": event.call_id,
                "attempt_index": event.attempt_index,
                "provider_id": event.provider_id,
                "api_mode": event.api_mode,
                "model_name": event.model_name,
                "error_class": (
                    event.error_class.value if event.error_class is not None else None
                ),
                "fingerprint_code": event.fingerprint_code,
                "action_code": event.action_code,
                "rewrite_applied_keys": list(event.rewrite_applied_keys),
                "outcome": event.outcome,
                "latency_ms": event.latency_ms,
                **_trace_fields(),
            },
        )

    # ---- Private --------------------------------------------------------- #

    def _append(self, filename: str, payload: dict[str, Any]) -> None:
        try:
            path = self._log_dir / filename
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning(
                "JsonlPromptTelemetry: failed to write %s: %s",
                filename,
                exc,
            )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
