"""B5 PR-S1-6 acceptance: ``JsonlPromptTelemetry`` carries trace join keys.

Sprint 1 ships the canonical attribute contract (PR-S1-1) and the
contextvar carrier (PR-S1-2) plus the middleware that binds it
(PR-S1-5). PR-S1-6 closes the loop on the existing JSONL writer:
every ``assembly.jsonl`` / ``llm_invocation.jsonl`` /
``recovery_event.jsonl`` row now appends ``trace_id`` /
``request_id`` / ``session_id`` so an analyst can ``LEFT JOIN``
the JSONL files against future OTel span attributes via
``trace_id`` once Sprint 2 lands.

Schema invariants pinned here:

- Old fields are unchanged in name, position, type — backward
  compatible. Anything that read e.g. ``sections_included`` before
  Sprint 1 still reads the same shape.
- Three new tail fields exist on every row. ``trace_id`` and
  ``request_id`` are **NEVER null** (canonical contract from
  PR-S1-1) — when no request scope is bound, they fall back to
  ``uuid4().hex`` / ``str(uuid4())`` via
  ``build_canonical_attributes()`` so CLI / scheduler / background
  emissions still join cleanly against future OTel span
  attributes. ``session_id`` is allowed to be ``null`` (optional
  in the canonical contract).
- When a context IS bound, the values match the bound
  ``TraceContext`` exactly (no formatting / case mutation).

After Sprint 2, this writer's schema is FROZEN — any new field
goes on OTel attributes, not into the JSONL files.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from app.domain.external.observability import TraceContext


# Canonical formats from PR-S1-1 contract.
_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_REQUEST_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
from app.domain.services.provider_profiles._base import ErrorClass
from app.domain.services.recovery._event import RecoveryEvent
from app.infrastructure.observability.context import (
    reset_trace_context,
    set_trace_context,
)
from app.infrastructure.telemetry.prompt_telemetry import JsonlPromptTelemetry


_TRACE_ID = "0123456789abcdef0123456789abcdef"
_REQUEST_ID = "12345678-1234-4abc-8def-012345678901"
_EVENT_ID = "abcdef01-2345-4678-89ab-cdef01234567"
_SESSION_ID = "sess-jsonl-001"


def _read_one(path: Path) -> dict:
    """Read the single JSON line from ``path`` and return the parsed dict."""
    text = path.read_text(encoding="utf-8").strip()
    assert text, f"{path} is empty"
    lines = text.splitlines()
    assert len(lines) == 1, f"expected exactly 1 line, got {len(lines)}"
    return json.loads(lines[0])


def _bound_ctx() -> TraceContext:
    return TraceContext(
        trace_id=_TRACE_ID,
        request_id=_REQUEST_ID,
        event_id=_EVENT_ID,
        session_id=_SESSION_ID,
    )


def _sample_recovery_event() -> RecoveryEvent:
    return RecoveryEvent(
        call_id="test-call-uuid",
        attempt_index=1,
        provider_id="dashscope_qwen",
        api_mode="chat_completions",
        model_name="qwen-max-2025",
        error_class=ErrorClass.COMPAT_QUIRK,
        fingerprint_code="json_mode_with_thinking",
        action_code="strip_response_format",
        rewrite_applied_keys=("response_format",),
        outcome="retry_sent",
        latency_ms=147,
    )


# ---------------------------------------------------------------------------
# record_assembly
# ---------------------------------------------------------------------------
class TestRecordAssembly:
    def test_no_context_emits_canonical_fallback_trace_keys(
        self, tmp_path: Path
    ) -> None:
        """Review-found P1: ``trace_id`` / ``request_id`` are NEVER null.

        PR-S1-1 canonical contract: ``trace_id`` and ``request_id``
        are required join keys and must never be null. When no
        request scope is bound, ``_trace_fields()`` falls back to
        ``build_canonical_attributes()``'s uuid4 fallback so CLI /
        scheduler emissions still join cleanly against future OTel
        span attributes. ``session_id`` may stay ``None`` (optional
        in the contract).
        """
        telemetry = JsonlPromptTelemetry(log_dir=tmp_path)

        telemetry.record_assembly(
            sections_included=["identity", "behavior_core"],
            sections_dropped=[],
            tokens_used=1234,
            lang="en",
            provider="openai",
            mode="planner",
            version_hash="abc123",
            fallback_used=False,
        )

        row = _read_one(tmp_path / "assembly.jsonl")
        assert row["trace_id"] is not None, (
            "trace_id is null — violates v1 canonical contract"
        )
        assert _TRACE_ID_RE.match(row["trace_id"]), (
            f"trace_id {row['trace_id']!r} not 32-hex"
        )
        assert row["request_id"] is not None, (
            "request_id is null — violates v1 canonical contract"
        )
        assert _REQUEST_ID_RE.match(row["request_id"]), (
            f"request_id {row['request_id']!r} not UUIDv4"
        )
        # session_id is optional in the contract; with no scope
        # bound it stays None.
        assert row["session_id"] is None

    def test_bound_context_populates_trace_keys(self, tmp_path: Path) -> None:
        telemetry = JsonlPromptTelemetry(log_dir=tmp_path)

        token = set_trace_context(_bound_ctx())
        try:
            telemetry.record_assembly(
                sections_included=["identity"],
                sections_dropped=["sandbox_state"],
                tokens_used=2048,
                lang="zh",
                provider="anthropic",
                mode="executor",
                version_hash="def456",
                fallback_used=True,
            )
        finally:
            reset_trace_context(token)

        row = _read_one(tmp_path / "assembly.jsonl")
        assert row["trace_id"] == _TRACE_ID
        assert row["request_id"] == _REQUEST_ID
        assert row["session_id"] == _SESSION_ID

    def test_old_fields_preserved_unchanged(self, tmp_path: Path) -> None:
        """Backward compat: old keys still present with identical types."""
        telemetry = JsonlPromptTelemetry(log_dir=tmp_path)

        telemetry.record_assembly(
            sections_included=["identity", "behavior_core"],
            sections_dropped=["sandbox_state"],
            tokens_used=1234,
            lang="en",
            provider="openai",
            mode="planner",
            version_hash="abc123",
            fallback_used=False,
        )

        row = _read_one(tmp_path / "assembly.jsonl")
        assert row["sections_included"] == ["identity", "behavior_core"]
        assert row["sections_dropped"] == ["sandbox_state"]
        assert row["tokens_used"] == 1234
        assert row["lang"] == "en"
        assert row["provider"] == "openai"
        assert row["mode"] == "planner"
        assert row["version_hash"] == "abc123"
        assert row["fallback_used"] is False
        assert "ts" in row


# ---------------------------------------------------------------------------
# record_llm_invocation
# ---------------------------------------------------------------------------
class TestRecordLlmInvocation:
    def test_no_context_emits_canonical_fallback_trace_keys(
        self, tmp_path: Path
    ) -> None:
        """``trace_id`` / ``request_id`` are NEVER null on llm_invocation.jsonl."""
        telemetry = JsonlPromptTelemetry(log_dir=tmp_path)

        telemetry.record_llm_invocation(
            system_prompt_hash="sha256:abc",
            system_prompt_bytes=4096,
            tools_hash="sha256:def",
            lang="en",
            provider="openai",
        )

        row = _read_one(tmp_path / "llm_invocation.jsonl")
        assert row["trace_id"] is not None
        assert _TRACE_ID_RE.match(row["trace_id"]), (
            f"trace_id {row['trace_id']!r} not 32-hex"
        )
        assert row["request_id"] is not None
        assert _REQUEST_ID_RE.match(row["request_id"]), (
            f"request_id {row['request_id']!r} not UUIDv4"
        )
        assert row["session_id"] is None

    def test_bound_context_populates_trace_keys(self, tmp_path: Path) -> None:
        telemetry = JsonlPromptTelemetry(log_dir=tmp_path)

        token = set_trace_context(_bound_ctx())
        try:
            telemetry.record_llm_invocation(
                system_prompt_hash="sha256:abc",
                system_prompt_bytes=4096,
                tools_hash="sha256:def",
                lang="en",
                provider="openai",
            )
        finally:
            reset_trace_context(token)

        row = _read_one(tmp_path / "llm_invocation.jsonl")
        assert row["trace_id"] == _TRACE_ID
        assert row["request_id"] == _REQUEST_ID
        assert row["session_id"] == _SESSION_ID

    def test_old_fields_preserved_unchanged(self, tmp_path: Path) -> None:
        telemetry = JsonlPromptTelemetry(log_dir=tmp_path)

        telemetry.record_llm_invocation(
            system_prompt_hash="sha256:abc",
            system_prompt_bytes=4096,
            tools_hash="sha256:def",
            lang="en",
            provider="openai",
        )

        row = _read_one(tmp_path / "llm_invocation.jsonl")
        assert row["system_prompt_hash"] == "sha256:abc"
        assert row["system_prompt_bytes"] == 4096
        assert row["tools_hash"] == "sha256:def"
        assert row["lang"] == "en"
        assert row["provider"] == "openai"
        assert "ts" in row


# ---------------------------------------------------------------------------
# emit_recovery_event
# ---------------------------------------------------------------------------
class TestEmitRecoveryEvent:
    def test_no_context_emits_canonical_fallback_trace_keys(
        self, tmp_path: Path
    ) -> None:
        """``trace_id`` / ``request_id`` are NEVER null on recovery_event.jsonl."""
        telemetry = JsonlPromptTelemetry(log_dir=tmp_path)

        telemetry.emit_recovery_event(_sample_recovery_event())

        row = _read_one(tmp_path / "recovery_event.jsonl")
        assert row["trace_id"] is not None
        assert _TRACE_ID_RE.match(row["trace_id"]), (
            f"trace_id {row['trace_id']!r} not 32-hex"
        )
        assert row["request_id"] is not None
        assert _REQUEST_ID_RE.match(row["request_id"]), (
            f"request_id {row['request_id']!r} not UUIDv4"
        )
        assert row["session_id"] is None

    def test_bound_context_populates_trace_keys(self, tmp_path: Path) -> None:
        telemetry = JsonlPromptTelemetry(log_dir=tmp_path)

        token = set_trace_context(_bound_ctx())
        try:
            telemetry.emit_recovery_event(_sample_recovery_event())
        finally:
            reset_trace_context(token)

        row = _read_one(tmp_path / "recovery_event.jsonl")
        assert row["trace_id"] == _TRACE_ID
        assert row["request_id"] == _REQUEST_ID
        assert row["session_id"] == _SESSION_ID

    def test_old_fields_preserved_unchanged(self, tmp_path: Path) -> None:
        telemetry = JsonlPromptTelemetry(log_dir=tmp_path)

        telemetry.emit_recovery_event(_sample_recovery_event())

        row = _read_one(tmp_path / "recovery_event.jsonl")
        assert row["call_id"] == "test-call-uuid"
        assert row["attempt_index"] == 1
        assert row["provider_id"] == "dashscope_qwen"
        assert row["api_mode"] == "chat_completions"
        assert row["model_name"] == "qwen-max-2025"
        assert row["error_class"] == ErrorClass.COMPAT_QUIRK.value
        assert row["fingerprint_code"] == "json_mode_with_thinking"
        assert row["action_code"] == "strip_response_format"
        assert row["rewrite_applied_keys"] == ["response_format"]
        assert row["outcome"] == "retry_sent"
        assert row["latency_ms"] == 147
        assert "ts" in row


# ---------------------------------------------------------------------------
# Per-row uniformity
# ---------------------------------------------------------------------------
def test_every_row_carries_uniform_trace_key_set(tmp_path: Path) -> None:
    """Every JSONL row from any emit path has the same join-key column set.

    Lets downstream consumers union-select across the three files
    on ``trace_id`` / ``request_id`` / ``session_id`` without
    per-file column shimming.
    """
    telemetry = JsonlPromptTelemetry(log_dir=tmp_path)

    telemetry.record_assembly(
        sections_included=["x"],
        sections_dropped=[],
        tokens_used=1,
        lang="en",
        provider="openai",
        mode="planner",
        version_hash="v",
        fallback_used=False,
    )
    telemetry.record_llm_invocation(
        system_prompt_hash="h",
        system_prompt_bytes=1,
        tools_hash="t",
        lang="en",
        provider="openai",
    )
    telemetry.emit_recovery_event(_sample_recovery_event())

    expected = {"trace_id", "request_id", "session_id"}
    for filename in (
        "assembly.jsonl",
        "llm_invocation.jsonl",
        "recovery_event.jsonl",
    ):
        row = _read_one(tmp_path / filename)
        assert expected <= row.keys(), (
            f"{filename} missing join keys; got keys={sorted(row.keys())}"
        )


def test_no_emit_writes_null_required_join_keys(tmp_path: Path) -> None:
    """Review-found P1 contract: ``trace_id`` / ``request_id`` NEVER null.

    Drives every emit path with no bound ``TraceContext`` (the
    "CLI / scheduler / unit test" code path) and asserts both
    required canonical join keys are non-null on every JSONL row.
    Catches regressions where ``_trace_fields()`` reverts to
    emitting raw ``None`` for the absent-context case — which
    would silently break ``trace_id``-keyed log/span join the
    moment Sprint 2 wires OTel.
    """
    telemetry = JsonlPromptTelemetry(log_dir=tmp_path)

    telemetry.record_assembly(
        sections_included=["x"],
        sections_dropped=[],
        tokens_used=1,
        lang="en",
        provider="openai",
        mode="planner",
        version_hash="v",
        fallback_used=False,
    )
    telemetry.record_llm_invocation(
        system_prompt_hash="h",
        system_prompt_bytes=1,
        tools_hash="t",
        lang="en",
        provider="openai",
    )
    telemetry.emit_recovery_event(_sample_recovery_event())

    for filename in (
        "assembly.jsonl",
        "llm_invocation.jsonl",
        "recovery_event.jsonl",
    ):
        row = _read_one(tmp_path / filename)
        assert row["trace_id"] is not None, (
            f"{filename} wrote null trace_id — violates v1 canonical contract"
        )
        assert row["request_id"] is not None, (
            f"{filename} wrote null request_id — violates v1 canonical contract"
        )
