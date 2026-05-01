"""B5 PR-S3-2: ``record_decision`` writes span events on the active span.

Locks:

- ``record_decision(name, outcome, reason=...)`` adds a span event
  named ``decision.<name>`` with ``decision_outcome=<outcome>`` and
  ``decision_reason=<reason>`` attributes on the currently-active
  span.
- Canonical attributes ride through (``trace_id`` / ``request_id`` /
  ``event_id`` / ``graph_node`` / ``step_id`` / ``session_id``) via
  ``build_canonical_attributes``, joining the event with whatever
  parent span context the request had.
- Non-canonical caller-supplied attrs are silently dropped (same
  whitelist semantics as ``OtelToolSpanCallback``).
- ``None`` values dropped at the boundary.
- Empty ``name`` / ``outcome`` is a logged no-op.
- No active span: call is a no-op (no exception, no leak).
"""
from __future__ import annotations

from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from app.infrastructure.observability.decision_trace import record_decision


@pytest.fixture
def tracer_and_exporter() -> tuple[Any, InMemorySpanExporter]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("actus-test")
    return tracer, exporter


def test_record_decision_adds_event_with_outcome_and_reason(
    tracer_and_exporter,
):
    tracer, exporter = tracer_and_exporter
    with tracer.start_as_current_span("tool.shell_execute"):
        record_decision(
            "smart_approve",
            outcome="deny",
            reason="rm -rf /; clearly destructive",
        )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    events = list(span.events)
    assert len(events) == 1
    event = events[0]
    assert event.name == "decision.smart_approve"
    assert event.attributes.get("decision_outcome") == "deny"
    assert event.attributes.get("decision_reason") == (
        "rm -rf /; clearly destructive"
    )


def test_record_decision_canonical_attrs_ride_through(tracer_and_exporter):
    """``trace_id`` / ``request_id`` / ``event_id`` always present on
    the event so it joins to logs / other spans on the canonical
    contract.
    """
    tracer, exporter = tracer_and_exporter
    with tracer.start_as_current_span("tool.x"):
        record_decision("smart_approve", outcome="approve")

    event = list(exporter.get_finished_spans()[0].events)[0]
    assert event.attributes.get("trace_id")
    assert event.attributes.get("request_id")
    assert event.attributes.get("event_id")


def test_record_decision_caller_attrs_filtered_through_whitelist(
    tracer_and_exporter,
):
    """Non-canonical caller attrs are dropped; canonical ones flow
    through. Same whitelist as ``OtelToolSpanCallback``.
    """
    tracer, exporter = tracer_and_exporter
    with tracer.start_as_current_span("tool.x"):
        record_decision(
            "smart_approve",
            outcome="approve",
            attrs={
                "tool_name": "shell_execute",  # canonical → kept
                "model": "gpt-4o",  # canonical → kept
                "raw_secret": "DROP-ME",  # not canonical → dropped
                "input_str": "DROP-ME-TOO",  # not canonical → dropped
            },
        )

    event = list(exporter.get_finished_spans()[0].events)[0]
    assert event.attributes.get("tool_name") == "shell_execute"
    assert event.attributes.get("model") == "gpt-4o"
    assert "raw_secret" not in event.attributes
    assert "input_str" not in event.attributes


def test_record_decision_none_values_dropped(tracer_and_exporter):
    """``None`` reason / attr values must not appear on the event."""
    tracer, exporter = tracer_and_exporter
    with tracer.start_as_current_span("tool.x"):
        record_decision(
            "smart_approve",
            outcome="approve",
            reason=None,
            attrs={"tool_name": None, "model": "gpt-4o"},
        )

    event = list(exporter.get_finished_spans()[0].events)[0]
    assert "decision_reason" not in event.attributes
    assert "tool_name" not in event.attributes
    assert event.attributes.get("model") == "gpt-4o"


def test_record_decision_caller_attr_overrides_contextvar(
    tracer_and_exporter,
):
    """When the caller supplies a canonical attr explicitly, it takes
    precedence over whatever the contextvar carries (decision site
    has the most specific knowledge). Use ``graph_node`` here — it's
    the canonical-attr field that actually lives on ``TraceContext``,
    so we can pre-bind a value and watch the caller-supplied value
    win.
    """
    from app.domain.external.observability import TraceContext
    from app.infrastructure.observability.context import (
        reset_trace_context,
        set_trace_context,
    )

    tracer, exporter = tracer_and_exporter
    ctx = TraceContext(
        trace_id="0123456789abcdef0123456789abcdef",
        request_id="11111111-1111-4111-8111-111111111111",
        event_id="22222222-2222-4222-8222-222222222222",
        graph_node="ambient_node_from_ctx",
    )
    token = set_trace_context(ctx)
    try:
        with tracer.start_as_current_span("graph.node.executor"):
            record_decision(
                "smart_approve",
                outcome="deny",
                attrs={"graph_node": "explicit_executor_node"},
            )
    finally:
        reset_trace_context(token)

    event = list(exporter.get_finished_spans()[0].events)[0]
    assert event.attributes.get("graph_node") == "explicit_executor_node"


def test_record_decision_empty_name_or_outcome_is_noop(tracer_and_exporter):
    """Defensive: empty ``name`` or ``outcome`` → logged no-op, no
    event added (don't crash the agent step on a buggy caller).
    """
    tracer, exporter = tracer_and_exporter
    with tracer.start_as_current_span("tool.x"):
        record_decision("", outcome="deny")
        record_decision("smart_approve", outcome="")

    span = exporter.get_finished_spans()[0]
    assert len(span.events) == 0


def test_record_decision_no_active_span_is_safe_noop():
    """Without an active span, the underlying OTel call is a no-op
    (``INVALID_SPAN.add_event`` swallows the call). MUST NOT raise.
    """
    record_decision("smart_approve", outcome="approve", reason="trivial")


def test_record_decision_event_name_prefixed_with_decision(
    tracer_and_exporter,
):
    """Event name format is locked: ``decision.<name>``. Phoenix /
    Jaeger filter UIs key on this prefix to scope dashboards to
    decision events specifically.
    """
    tracer, exporter = tracer_and_exporter
    with tracer.start_as_current_span("tool.x"):
        record_decision("recovery", outcome="retry")
        record_decision("permission_engine", outcome="allow")

    span = exporter.get_finished_spans()[0]
    names = [e.name for e in span.events]
    assert names == ["decision.recovery", "decision.permission_engine"]


def test_record_decision_swallows_build_canonical_attributes_error(
    tracer_and_exporter, monkeypatch
):
    """Reviewer round-2 P2: ``record_decision`` MUST be fully
    best-effort.

    A malformed ``TraceContext`` on the contextvar makes
    ``build_canonical_attributes → validate_attributes`` raise
    ``ValueError``. Without the unified ``try/except``, that
    exception escapes into ``SmartApprove.evaluate``'s outer
    ``except Exception`` branch and silently turns legitimate
    ``approve``/``deny`` outcomes into ``escalate`` with
    ``decision_reason="llm_error"`` — changing agent behaviour, not
    just the observability picture.

    Strategy: monkeypatch ``build_canonical_attributes`` (the symbol
    used inside ``decision_trace``) to raise, call ``record_decision``
    inside an active span, assert NO exception leaks AND no event was
    added (the silent no-op is the contract).
    """
    import app.infrastructure.observability.decision_trace as dt

    def _boom(**kwargs):
        raise ValueError("malformed TraceContext: trace_id='bad'")

    monkeypatch.setattr(dt, "build_canonical_attributes", _boom)

    tracer, exporter = tracer_and_exporter
    # Must not raise.
    with tracer.start_as_current_span("tool.x"):
        record_decision("smart_approve", outcome="approve")

    span = exporter.get_finished_spans()[0]
    # No event emitted (the build failed, the whole call no-ops).
    assert len(span.events) == 0


