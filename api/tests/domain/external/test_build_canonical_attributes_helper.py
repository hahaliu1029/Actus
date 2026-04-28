"""B5 PR-S1-1 acceptance: ``build_canonical_attributes`` helper behavior.

POST Q3 decision in the design doc: every span emit / JSONL write goes
through ``build_canonical_attributes`` instead of hand-rolling a dict.
The helper must:

- Read trace_id / request_id / session_id / user_id_hash from the
  request-scoped ``TraceContext`` (via the lazy import of
  ``app.infrastructure.observability.context.get_trace_context``)
- Let caller-supplied ``graph_node`` / ``step_id`` override the
  contextvar-bound coarse defaults — emit-site precision wins
- Generate a fresh ``event_id`` per call (uuid4 per emit)
- Return a dict that already passes ``validate_attributes``
- Fall back to generated trace_id / request_id when no context is
  bound (CLI / startup / direct unit tests) — the contract requires
  these fields to be NEVER null

All ctx fixtures use spec-format-compliant IDs (32-hex trace_id,
UUIDv4 request_id / event_id) so the v1 format check inside
``validate_attributes`` accepts the helper's output.
"""
from __future__ import annotations

import re
import uuid

from app.domain.external.observability import (
    CANONICAL_ATTRIBUTES,
    REQUIRED_ATTRIBUTES,
    TraceContext,
    build_canonical_attributes,
)

_UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_HEX32_RE = re.compile(r"^[0-9a-f]{32}$")

# Canonical ctx fixture values. Each satisfies the v1 format constraint
# enforced by ``validate_attributes`` (trace_id = 32 hex; request_id /
# session_id / event_id / step_id = UUIDv4 with dashes; user_id_hash =
# 16 lowercase hex per the canonical contract).
_CTX_TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
_CTX_REQUEST_ID = "11111111-1111-4111-8111-111111111111"
_CTX_EVENT_ID = "22222222-2222-4222-8222-222222222222"
_CTX_SESSION_ID = "33333333-3333-4333-8333-333333333333"
_CTX_USER_HASH = "a1b2c3d4e5f60789"
_CTX_STEP_ID = "44444444-4444-4444-8444-444444444444"


def _minimal_ctx(**overrides) -> TraceContext:
    base: dict[str, object] = {
        "trace_id": _CTX_TRACE_ID,
        "request_id": _CTX_REQUEST_ID,
        "event_id": _CTX_EVENT_ID,
    }
    base.update(overrides)
    return TraceContext(**base)  # type: ignore[arg-type]


def _patch_get_trace_context(monkeypatch, ctx: TraceContext | None) -> None:
    """Inject a fake ``TraceContext`` into the lazy-import path.

    The helper does ``from app.infrastructure.observability.context
    import get_trace_context`` inside its function body, so we patch the
    real module attribute. Subsequent in-function imports re-resolve and
    pick up the patched callable.
    """
    monkeypatch.setattr(
        "app.infrastructure.observability.context.get_trace_context",
        lambda: ctx,
    )


class TestReadsContextvars:
    def test_reads_trace_id_from_ctx(self, monkeypatch):
        ctx = TraceContext(
            trace_id=_CTX_TRACE_ID,
            request_id=_CTX_REQUEST_ID,
            event_id=_CTX_EVENT_ID,
            session_id=_CTX_SESSION_ID,
            user_id_hash=_CTX_USER_HASH,
            graph_node="planner",
            step_id=_CTX_STEP_ID,
        )
        _patch_get_trace_context(monkeypatch, ctx)

        result = build_canonical_attributes()

        assert result["trace_id"] == _CTX_TRACE_ID
        assert result["request_id"] == _CTX_REQUEST_ID
        assert result["session_id"] == _CTX_SESSION_ID
        assert result["user_id_hash"] == _CTX_USER_HASH
        assert result["graph_node"] == "planner"
        assert result["step_id"] == _CTX_STEP_ID

    def test_event_id_is_fresh_per_call(self, monkeypatch):
        _patch_get_trace_context(monkeypatch, _minimal_ctx())

        first = build_canonical_attributes()
        second = build_canonical_attributes()

        assert first["event_id"] != second["event_id"]
        # Helper does NOT propagate ctx.event_id into the result —
        # fresh uuid4 per emit.
        assert first["event_id"] != _CTX_EVENT_ID
        assert _UUID4_RE.match(first["event_id"])
        assert _UUID4_RE.match(second["event_id"])


class TestLocalsOverrideContext:
    def test_graph_node_override(self, monkeypatch):
        _patch_get_trace_context(
            monkeypatch, _minimal_ctx(graph_node="planner")
        )

        result = build_canonical_attributes(graph_node="executor")

        assert result["graph_node"] == "executor"

    def test_step_id_override(self, monkeypatch):
        _patch_get_trace_context(
            monkeypatch, _minimal_ctx(step_id=_CTX_STEP_ID)
        )

        local_step_id = "55555555-5555-4555-8555-555555555555"
        result = build_canonical_attributes(step_id=local_step_id)

        assert result["step_id"] == local_step_id

    def test_no_local_falls_back_to_ctx(self, monkeypatch):
        _patch_get_trace_context(
            monkeypatch,
            _minimal_ctx(graph_node="planner", step_id=_CTX_STEP_ID),
        )

        result = build_canonical_attributes()

        assert result["graph_node"] == "planner"
        assert result["step_id"] == _CTX_STEP_ID


class TestEmitSiteOnlyFields:
    def test_tool_fields_pass_through(self, monkeypatch):
        _patch_get_trace_context(monkeypatch, _minimal_ctx())

        result = build_canonical_attributes(
            tool_name="file_read",
            tool_call_id="call_abc",
            tool_args_hash="deadbeef" * 2,
            tool_args_size=42,
        )

        assert result["tool_name"] == "file_read"
        assert result["tool_call_id"] == "call_abc"
        assert result["tool_args_hash"] == "deadbeef" * 2
        assert result["tool_args_size"] == 42

    def test_llm_fields_pass_through(self, monkeypatch):
        _patch_get_trace_context(monkeypatch, _minimal_ctx())

        result = build_canonical_attributes(
            llm_provider="openai",
            model="gpt-4o-2024-11-20",
            attempt_ix=0,
        )

        assert result["llm_provider"] == "openai"
        assert result["model"] == "gpt-4o-2024-11-20"
        assert result["attempt_ix"] == 0

    def test_decision_reason_pass_through(self, monkeypatch):
        _patch_get_trace_context(monkeypatch, _minimal_ctx())

        result = build_canonical_attributes(
            decision_reason="executor preferred file_read because plan step required code inspection"
        )

        assert (
            result["decision_reason"]
            == "executor preferred file_read because plan step required code inspection"
        )


class TestNoContextFallback:
    def test_no_context_generates_fallback_ids(self, monkeypatch):
        _patch_get_trace_context(monkeypatch, None)

        result = build_canonical_attributes()

        assert result["trace_id"]
        assert result["request_id"]
        assert result["event_id"]
        assert _HEX32_RE.match(result["trace_id"])
        assert _UUID4_RE.match(result["request_id"])
        assert _UUID4_RE.match(result["event_id"])

    def test_no_context_session_user_are_none(self, monkeypatch):
        _patch_get_trace_context(monkeypatch, None)

        result = build_canonical_attributes()

        assert result["session_id"] is None
        assert result["user_id_hash"] is None

    def test_fallback_each_call_distinct(self, monkeypatch):
        _patch_get_trace_context(monkeypatch, None)

        first = build_canonical_attributes()
        second = build_canonical_attributes()

        assert first["trace_id"] != second["trace_id"]
        assert first["request_id"] != second["request_id"]
        assert first["event_id"] != second["event_id"]


class TestValidationContract:
    def test_result_passes_validate_attributes(self, monkeypatch):
        _patch_get_trace_context(monkeypatch, _minimal_ctx())

        result = build_canonical_attributes()

        for key in REQUIRED_ATTRIBUTES:
            assert key in result

        assert set(result.keys()).issubset(set(CANONICAL_ATTRIBUTES))

    def test_result_has_all_canonical_keys(self, monkeypatch):
        _patch_get_trace_context(monkeypatch, _minimal_ctx())

        result = build_canonical_attributes()

        assert set(result.keys()) == set(CANONICAL_ATTRIBUTES)


class TestUuidGenerationStable:
    def test_event_id_is_string_not_uuid_object(self, monkeypatch):
        _patch_get_trace_context(monkeypatch, _minimal_ctx())

        result = build_canonical_attributes()

        assert isinstance(result["event_id"], str)
        uuid.UUID(result["event_id"])
