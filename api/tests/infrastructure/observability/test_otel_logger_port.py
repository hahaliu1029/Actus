"""B5 PR-S2-1 acceptance: ``OtelLogger`` satisfies ``LoggerPort``.

Verifies the LoggerPort surface (info / warning / error / exception),
that calls flow through stdlib (still hit RedactingFormatter on the
root pipeline), and that the OTel ``LoggingHandler`` produces
``LogRecord`` objects with canonical attributes attached when a
``TraceContext`` is bound on the contextvar.

A throwaway ``InMemoryLogExporter`` is wired into a fresh
``LoggerProvider`` so we can introspect emitted OTel records without
relying on disk or network.
"""
from __future__ import annotations

import logging
import uuid

import pytest
from opentelemetry import _logs as otel_logs
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import (
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)

from app.domain.external.observability import (
    CANONICAL_ATTRIBUTES,
    TraceContext,
)
from app.infrastructure.logging.redaction import RedactingFormatter
from app.infrastructure.observability import (
    OtelLogger,
    teardown_observability,
)
from app.infrastructure.observability.context import (
    reset_trace_context,
    set_trace_context,
)
from app.infrastructure.observability.init import (
    _PlaceholderStrippingLoggingHandler,
)


@pytest.fixture
def in_memory_log_provider():
    """Install a throwaway LoggerProvider + InMemoryLogExporter.

    Bypasses ``setup_observability`` so the test owns the exporter and
    can introspect captured records directly. Detaches its handler on
    teardown so the next test gets a clean root logger.
    """
    teardown_observability()

    provider = LoggerProvider()
    exporter = InMemoryLogExporter()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    otel_logs.set_logger_provider(provider)

    handler = _PlaceholderStrippingLoggingHandler(
        level=logging.NOTSET, logger_provider=provider
    )
    handler.setFormatter(RedactingFormatter("%(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    previous_level = root.level
    root.setLevel(logging.DEBUG)

    yield exporter

    root.removeHandler(handler)
    root.setLevel(previous_level)
    handler.close()
    teardown_observability()


def test_logger_port_methods_are_callable(in_memory_log_provider):
    log = OtelLogger("actus.test.s2_1_logger_port")
    log.info("hello info")
    log.warning("hello warning")
    log.error("hello error")
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        log.exception("hello exception")
    records = in_memory_log_provider.get_finished_logs()
    bodies = [r.log_record.body for r in records]
    assert "hello info" in bodies
    assert "hello warning" in bodies
    assert "hello error" in bodies
    # exception() must include the traceback in the rendered body.
    assert any("RuntimeError: boom" in (b or "") for b in bodies)


def test_canonical_attrs_ride_on_log_record(in_memory_log_provider):
    """Canonical attributes from the bound ``TraceContext`` end up on
    ``LogRecord.attributes`` so OTLP exporters can use them as join
    keys downstream.
    """
    ctx = TraceContext(
        trace_id=uuid.uuid4().hex,
        request_id=str(uuid.uuid4()),
        event_id=str(uuid.uuid4()),
        session_id=str(uuid.uuid4()),
    )
    token = set_trace_context(ctx)
    try:
        OtelLogger("actus.test.s2_1_attrs").info("with-context")
    finally:
        reset_trace_context(token)

    records = in_memory_log_provider.get_finished_logs()
    target = next(
        r for r in records if r.log_record.body == "with-context"
    )
    attrs = dict(target.log_record.attributes or {})
    assert attrs.get("trace_id") == ctx.trace_id
    assert attrs.get("request_id") == ctx.request_id
    assert attrs.get("session_id") == ctx.session_id
    # The fresh event_id from build_canonical_attributes is also present.
    assert "event_id" in attrs


def test_extra_overrides_canonical_defaults(in_memory_log_provider):
    """``extra={...}`` passed to ``OtelLogger`` overrides the canonical
    fallback values. Used by emit sites that already know the
    ``graph_node`` or ``tool_name`` for the call site.
    """
    OtelLogger("actus.test.s2_1_extra").info(
        "with-extra",
        extra={"graph_node": "executor"},
    )
    target = next(
        r
        for r in in_memory_log_provider.get_finished_logs()
        if r.log_record.body == "with-extra"
    )
    attrs = dict(target.log_record.attributes or {})
    assert attrs.get("graph_node") == "executor"


def test_redacting_formatter_scrubs_secret_in_body(in_memory_log_provider):
    """OTel ``LogRecord.body`` is the redacted full string.

    Sprint 1's RedactingFormatter is set on the LoggingHandler, so OTel
    bodies cannot leak ``sk-...`` keys even when those tokens surface
    only in the rendered traceback.
    """
    OtelLogger("actus.test.s2_1_redact").info(
        "leaked sk-abcdef0123456789012345 in message"
    )
    target = next(
        r
        for r in in_memory_log_provider.get_finished_logs()
        if r.log_record.body and "leaked" in r.log_record.body
    )
    body = target.log_record.body
    assert "sk-abcdef0123456789012345" not in body
    assert "[REDACTED]" in body or "REDACTED" in body


def test_no_none_canonical_attrs_on_log_record(in_memory_log_provider):
    """LOCK P2: ``None`` canonical attrs must not reach OTel attributes.

    The v1 contract makes most canonical attrs nullable
    (``graph_node`` / ``tool_name`` / ``attempt_ix`` / ...). OTLP
    collectors reject ``None`` attribute values (the protobuf schema
    disallows null), and downstream JSONL exporters surfacing
    ``"graph_node": null`` is wire-format noise. Filter ``v is not
    None`` before merging into ``LogRecord.extra`` and lock that
    behaviour here.
    """
    OtelLogger("actus.test.s2_1_no_null").info("with-null-canon")
    target = next(
        r
        for r in in_memory_log_provider.get_finished_logs()
        if r.log_record.body == "with-null-canon"
    )
    attrs = dict(target.log_record.attributes or {})
    null_attrs = {k: v for k, v in attrs.items() if v is None}
    assert null_attrs == {}, (
        f"OTel attributes contain None values: {null_attrs}; canonical "
        f"keys with no value must be omitted, not emitted as null"
    )


def test_no_context_emit_required_attrs_pass_v1_format(
    in_memory_log_provider,
):
    """LOCK P2: no ``TraceContext`` bound → OTel attrs still pass v1 format.

    Without a bound context, PR-S1-4's ``_actus_log_record_factory``
    pins ``trace_id`` / ``request_id`` / ``session_id`` to the ``"-"``
    placeholder so stdlib ``%(trace_id)s`` format strings don't crash.
    But that placeholder fails ``validate_attributes`` (which the
    canonical contract makes the load-bearing v1 join-key gate) when
    surfaced on the OTel side.

    ``OtelLogger`` synthesises a ``TraceContext`` from
    ``build_canonical_attributes`` fallback uuids and binds it for the
    duration of ``_logger.log()`` so the factory writes valid uuids
    onto the record. Exercise the CLI / startup / background path
    here and assert the resulting OTel attributes survive
    ``validate_attributes``.
    """
    from app.domain.external.observability import validate_attributes
    from app.infrastructure.observability.context import get_trace_context

    assert get_trace_context() is None, (
        "precondition: this test must run with no TraceContext bound"
    )

    OtelLogger("actus.test.s2_1_no_ctx").info("no-context-emit")

    target = next(
        r
        for r in in_memory_log_provider.get_finished_logs()
        if r.log_record.body == "no-context-emit"
    )
    attrs = dict(target.log_record.attributes or {})

    # Canonical-only dict; validate_attributes raises if any required
    # attr is missing or fails v1 format.
    sanitized = {
        "trace_id": attrs.get("trace_id"),
        "request_id": attrs.get("request_id"),
        "event_id": attrs.get("event_id"),
    }
    validate_attributes(sanitized)

    # Synthetic context must NOT leak past the emit.
    assert get_trace_context() is None, (
        "synthetic TraceContext leaked out of OtelLogger._emit"
    )


def test_no_context_session_id_absent_from_otel_attrs(
    in_memory_log_provider,
):
    """LOCK P2: factory ``"-"`` placeholder for ``session_id`` does
    not leak into OTel attributes when no ``TraceContext`` is bound.

    PR-S1-4 ``_actus_log_record_factory`` writes ``record.session_id =
    "-"`` so stdlib ``%(session_id)s`` format strings render. That
    placeholder is not real data — the right OTel-side answer is
    "attribute absent". The ``_PlaceholderStrippingLoggingHandler``
    drops the placeholder before translation.
    """
    from app.infrastructure.observability.context import get_trace_context

    assert get_trace_context() is None, "precondition: no ctx bound"

    OtelLogger("actus.test.s2_1_no_session").info("no-session-emit")
    target = next(
        r
        for r in in_memory_log_provider.get_finished_logs()
        if r.log_record.body == "no-session-emit"
    )
    attrs = dict(target.log_record.attributes or {})
    assert "session_id" not in attrs, (
        f"factory '-' placeholder for session_id leaked to OTel: "
        f"got attrs={attrs!r}"
    )


def test_request_without_session_session_id_absent_from_otel_attrs(
    in_memory_log_provider,
):
    """LOCK P2: HTTP request initial ctx (session_id=None) → OTel attrs
    do NOT contain ``session_id="-"``.

    ObservabilityMiddleware binds a ``TraceContext`` with valid
    ``trace_id`` / ``request_id`` and ``session_id=None``. The factory
    writes ``record.session_id = "-"`` because nullable-and-None →
    placeholder. OTel handler must strip it.
    """
    import uuid

    from app.domain.external.observability import TraceContext
    from app.infrastructure.observability.context import (
        reset_trace_context,
        set_trace_context,
    )

    ctx = TraceContext(
        trace_id=uuid.uuid4().hex,
        request_id=str(uuid.uuid4()),
        event_id=str(uuid.uuid4()),
        # session_id stays None — mirrors middleware initial bind
    )
    token = set_trace_context(ctx)
    try:
        OtelLogger("actus.test.s2_1_req_no_session").info(
            "request-no-session"
        )
    finally:
        reset_trace_context(token)

    target = next(
        r
        for r in in_memory_log_provider.get_finished_logs()
        if r.log_record.body == "request-no-session"
    )
    attrs = dict(target.log_record.attributes or {})
    assert "session_id" not in attrs, (
        f"factory '-' placeholder for session_id leaked to OTel "
        f"during request-without-session emit: got attrs={attrs!r}"
    )
    # Required attrs from ctx still present + valid.
    assert attrs.get("trace_id") == ctx.trace_id
    assert attrs.get("request_id") == ctx.request_id


def test_session_bound_real_session_id_kept_in_otel_attrs(
    in_memory_log_provider,
):
    """LOCK P2 inverse: real session_id values are NOT stripped.

    The placeholder filter only drops ``"-"`` — anything else (real
    UUID, slug, opaque token) flows through unchanged.
    """
    import uuid

    from app.domain.external.observability import TraceContext
    from app.infrastructure.observability.context import (
        reset_trace_context,
        set_trace_context,
    )

    real_session = str(uuid.uuid4())
    ctx = TraceContext(
        trace_id=uuid.uuid4().hex,
        request_id=str(uuid.uuid4()),
        event_id=str(uuid.uuid4()),
        session_id=real_session,
    )
    token = set_trace_context(ctx)
    try:
        OtelLogger("actus.test.s2_1_real_session").info("real-session")
    finally:
        reset_trace_context(token)

    target = next(
        r
        for r in in_memory_log_provider.get_finished_logs()
        if r.log_record.body == "real-session"
    )
    attrs = dict(target.log_record.attributes or {})
    assert attrs.get("session_id") == real_session, (
        f"real session_id was stripped: got attrs={attrs!r}"
    )


def test_extra_factory_owned_keys_are_dropped(in_memory_log_provider):
    """LOCK P2: caller passing ``extra={"trace_id": ...}`` must NOT crash.

    PR-S1-4's ``_actus_log_record_factory`` already pins ``trace_id`` /
    ``request_id`` / ``session_id`` onto every ``LogRecord``. Letting
    a caller-supplied value through ``extra`` would trip stdlib's
    ``KeyError: "Attempt to overwrite 'trace_id' in LogRecord"``. The
    canonical contract says these are factory-owned — silently filter
    them out instead of crashing the emit site.
    """
    OtelLogger("actus.test.s2_1_extra_factory").info(
        "extra-factory-owned",
        extra={
            "trace_id": "should-not-overwrite",
            "request_id": "should-not-overwrite",
            "session_id": "should-not-overwrite",
            "graph_node": "executor",
        },
    )
    target = next(
        r
        for r in in_memory_log_provider.get_finished_logs()
        if r.log_record.body == "extra-factory-owned"
    )
    attrs = dict(target.log_record.attributes or {})
    assert attrs.get("trace_id") != "should-not-overwrite", (
        "factory-owned trace_id must not be overridden by caller extra"
    )
    # graph_node passed in extra still wins for emit-site values.
    assert attrs.get("graph_node") == "executor"


def test_extra_cannot_override_event_id_or_user_id_hash(
    in_memory_log_provider,
):
    """LOCK P2: caller ``extra`` can NOT override ``event_id`` or ``user_id_hash``.

    ``event_id`` is required + fresh per emit per the v1 canonical
    contract — letting a caller pin one breaks uniqueness across emits
    and could push a non-UUIDv4 string past validate_attributes
    downstream. ``user_id_hash`` is policy-derived from
    ``TraceContext`` + secret salt — an emit site cannot hash a
    different user's id past the policy boundary.

    Both keys must be silently dropped from caller ``extra``; the
    canonical fallback (``build_canonical_attributes``) supplies the
    real ``event_id`` per emit and the ``user_id_hash`` from ctx.
    """
    from app.domain.external.observability import _UUID4_RE

    OtelLogger("actus.test.s2_1_extra_forbidden").info(
        "extra-forbidden",
        extra={
            "event_id": "bad-event-id",
            "user_id_hash": "bad-hash",
            "graph_node": "executor",
        },
    )
    target = next(
        r
        for r in in_memory_log_provider.get_finished_logs()
        if r.log_record.body == "extra-forbidden"
    )
    attrs = dict(target.log_record.attributes or {})
    # event_id stayed fresh (UUIDv4) and was NOT replaced by the bad caller value.
    assert attrs.get("event_id") != "bad-event-id", (
        f"caller-supplied event_id leaked into OTel attrs: {attrs!r}"
    )
    assert _UUID4_RE.fullmatch(attrs.get("event_id", "")), (
        f"event_id must remain a fresh UUIDv4: {attrs!r}"
    )
    # user_id_hash, if absent in ctx, must NOT take the caller's bad value.
    assert attrs.get("user_id_hash") != "bad-hash", (
        f"caller-supplied user_id_hash leaked: {attrs!r}"
    )
    # Allowed extra still flowed through.
    assert attrs.get("graph_node") == "executor"


def test_extra_unknown_keys_are_dropped(in_memory_log_provider):
    """LOCK P2: unknown caller ``extra`` keys must NOT reach OTel attrs.

    The v1 canonical contract is FROZEN — ``validate_attributes`` drops
    unknown keys for downstream JSONL emit sites. ``OtelLogger`` must
    apply the same drop-unknown semantics so non-canonical fields
    cannot leak into OTel ``LogRecord.attributes`` (and from there to
    OTLP exporters where they'd be free-form attribute pollution).
    """
    OtelLogger("actus.test.s2_1_extra_unknown").info(
        "extra-unknown",
        extra={"not_canonical": "leaks", "graph_node": "executor"},
    )
    target = next(
        r
        for r in in_memory_log_provider.get_finished_logs()
        if r.log_record.body == "extra-unknown"
    )
    attrs = dict(target.log_record.attributes or {})
    assert "not_canonical" not in attrs, (
        f"non-canonical key leaked into OTel attributes: {attrs}"
    )
    assert attrs.get("graph_node") == "executor"


def test_explicit_extra_none_is_dropped(in_memory_log_provider):
    """LOCK P2: caller-supplied ``extra={..., k: None}`` is also dropped.

    Mirrors the canonical filter: an emit site that explicitly passes
    ``extra={"step_id": None}`` should not surface a null on the OTel
    side. The drop happens in ``OtelLogger._emit`` so both code paths
    agree on the wire format.
    """
    OtelLogger("actus.test.s2_1_extra_null").info(
        "extra-null",
        extra={"step_id": None, "graph_node": "planner"},
    )
    target = next(
        r
        for r in in_memory_log_provider.get_finished_logs()
        if r.log_record.body == "extra-null"
    )
    attrs = dict(target.log_record.attributes or {})
    assert "step_id" not in attrs, (
        "explicit None extras must be dropped before reaching OTel"
    )
    assert attrs.get("graph_node") == "planner"
