"""JSONL-backed implementation of PromptTelemetryPort.

B5 C1: writes telemetry events as JSON lines to a configurable directory.
Failures are logged and swallowed — telemetry must never propagate errors
to the prompt assembly hot path.

C1 only ships the implementation. DI wiring (constructing the instance and
passing it to PromptAssembler) happens in C5a.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.domain.external.telemetry import PromptTelemetryPort

logger = logging.getLogger(__name__)


class JsonlPromptTelemetry(PromptTelemetryPort):
    """Append-only JSONL writer for prompt assembly + LLM invocation telemetry.

    Three log files (created on demand under ``log_dir``):
    - ``assembly.jsonl`` — one record per ``PromptAssembler.assemble`` call
    - ``llm_invocation.jsonl`` — one record per LLM adapter invoke (C11)
    - ``degradation.jsonl`` — one record per ``_build_lc_tools`` fallback

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
            },
        )

    def record_lc_tools_degradation(self, *, reason: str) -> None:
        self._append(
            "degradation.jsonl",
            {
                "ts": _now_iso(),
                "reason": reason,
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
