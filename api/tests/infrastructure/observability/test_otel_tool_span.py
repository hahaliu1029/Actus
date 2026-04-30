"""B5 PR-S2-2: ``OtelToolSpanCallback`` emits one span per tool call.

Locks the contract:

- ``on_tool_start`` opens a ``tool.<name>`` span with ``tool_name`` /
  ``tool_args_hash`` / ``tool_args_size`` attributes.
- ``tool_args_hash`` is the first 16 hex chars of SHA-256 over the
  stable JSON serialization of ``inputs``.
- ``tool_args_size`` is the byte length of that serialization.
- ``on_tool_end`` closes the span (the in-memory exporter sees it).
- ``on_tool_error`` closes the span and records an exception.
- Concurrent tool calls (distinct ``run_id``) get distinct spans.
- The handler is robust to missing ``inputs`` (no crash; hash + size
  attributes simply absent).

Privacy invariant — see ``test_tool_args_hash_size_only.py``.
"""
from __future__ import annotations

import hashlib
import json
from uuid import uuid4

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from app.infrastructure.observability.otel_tool_span import OtelToolSpanCallback
from app.infrastructure.observability.otel_tracer import OtelTracer


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def callback_and_exporter() -> tuple[OtelToolSpanCallback, InMemorySpanExporter]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OtelTracer(provider.get_tracer("actus-test"))
    return OtelToolSpanCallback(tracer), exporter


@pytest.mark.anyio
async def test_on_tool_start_then_end_emits_one_span(callback_and_exporter):
    handler, exporter = callback_and_exporter
    run_id = uuid4()

    inputs = {"path": "/tmp/foo", "max_chars": 1000}
    await handler.on_tool_start(
        serialized={"name": "file_read"},
        input_str=json.dumps(inputs),
        run_id=run_id,
        inputs=inputs,
    )
    await handler.on_tool_end(output="ok", run_id=run_id)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "tool.file_read"
    assert span.attributes.get("tool_name") == "file_read"


@pytest.mark.anyio
async def test_tool_args_hash_is_sha256_truncated_16_hex(callback_and_exporter):
    handler, exporter = callback_and_exporter
    run_id = uuid4()

    inputs = {"path": "/tmp/foo", "max_chars": 1000}
    expected_serialized = json.dumps(inputs, sort_keys=True, default=str)
    expected_hash = hashlib.sha256(
        expected_serialized.encode("utf-8")
    ).hexdigest()[:16]
    expected_size = len(expected_serialized.encode("utf-8"))

    await handler.on_tool_start(
        serialized={"name": "file_read"},
        input_str=expected_serialized,
        run_id=run_id,
        inputs=inputs,
    )
    await handler.on_tool_end(output="ok", run_id=run_id)

    span = exporter.get_finished_spans()[0]
    assert span.attributes.get("tool_args_hash") == expected_hash
    assert len(span.attributes.get("tool_args_hash")) == 16
    assert span.attributes.get("tool_args_size") == expected_size


@pytest.mark.anyio
async def test_hash_is_stable_across_key_ordering(callback_and_exporter):
    """``sort_keys=True`` makes the hash invariant under dict ordering.

    Two semantically equal inputs (different key insertion order) must
    produce the same ``tool_args_hash``.
    """
    handler, exporter = callback_and_exporter

    a_id = uuid4()
    b_id = uuid4()
    a_inputs = {"a": 1, "b": 2}
    b_inputs = {"b": 2, "a": 1}

    await handler.on_tool_start(
        serialized={"name": "x"}, input_str="", run_id=a_id, inputs=a_inputs
    )
    await handler.on_tool_end(output="ok", run_id=a_id)
    await handler.on_tool_start(
        serialized={"name": "x"}, input_str="", run_id=b_id, inputs=b_inputs
    )
    await handler.on_tool_end(output="ok", run_id=b_id)

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert spans[0].attributes.get("tool_args_hash") == spans[1].attributes.get(
        "tool_args_hash"
    )


@pytest.mark.anyio
async def test_concurrent_tool_calls_get_distinct_spans(callback_and_exporter):
    handler, exporter = callback_and_exporter
    a_id = uuid4()
    b_id = uuid4()

    await handler.on_tool_start(
        serialized={"name": "tool_a"}, input_str="", run_id=a_id, inputs={"k": 1}
    )
    await handler.on_tool_start(
        serialized={"name": "tool_b"}, input_str="", run_id=b_id, inputs={"k": 2}
    )
    await handler.on_tool_end(output="ok-b", run_id=b_id)
    await handler.on_tool_end(output="ok-a", run_id=a_id)

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    names = {s.name for s in spans}
    assert names == {"tool.tool_a", "tool.tool_b"}


@pytest.mark.anyio
async def test_tool_error_records_exception_and_closes(callback_and_exporter):
    handler, exporter = callback_and_exporter
    run_id = uuid4()

    await handler.on_tool_start(
        serialized={"name": "shell_execute"},
        input_str="",
        run_id=run_id,
        inputs={"cmd": "ls"},
    )
    await handler.on_tool_error(error=RuntimeError("boom"), run_id=run_id)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "tool.shell_execute"
    event_names = {e.name for e in span.events}
    assert "exception" in event_names


@pytest.mark.anyio
async def test_tool_error_sets_span_status_to_error(callback_and_exporter):
    """Reviewer round-6 P2: ``record_exception`` only attaches an
    event — it does NOT flip the OTel span status. ``start_span``
    spans (unlike ``start_as_current_span``) get no auto-status on
    exit either. ``on_tool_error`` MUST explicitly set the span
    status to ERROR so dashboards / alerts that key on
    ``span.status_code == ERROR`` count tool failures alongside
    HTTP 5xx and graph node exceptions.
    """
    from opentelemetry.trace import StatusCode

    handler, exporter = callback_and_exporter
    run_id = uuid4()

    await handler.on_tool_start(
        serialized={"name": "shell_execute"},
        input_str="",
        run_id=run_id,
        inputs={"cmd": "ls"},
    )
    await handler.on_tool_error(error=RuntimeError("boom"), run_id=run_id)

    span = exporter.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR, (
        f"tool error span must have ERROR status; got {span.status!r}"
    )


@pytest.mark.anyio
async def test_tool_end_keeps_span_status_unset(callback_and_exporter):
    """Symmetric: ``on_tool_end`` (success) MUST NOT flip status
    to ERROR. Locked separately so a future regression that always
    sets ERROR (regardless of end vs error) gets caught.
    """
    from opentelemetry.trace import StatusCode

    handler, exporter = callback_and_exporter
    run_id = uuid4()

    await handler.on_tool_start(
        serialized={"name": "x"}, input_str="", run_id=run_id, inputs={}
    )
    await handler.on_tool_end(output="ok", run_id=run_id)

    span = exporter.get_finished_spans()[0]
    assert span.status.status_code != StatusCode.ERROR


@pytest.mark.anyio
async def test_missing_inputs_does_not_crash(callback_and_exporter):
    handler, exporter = callback_and_exporter
    run_id = uuid4()

    await handler.on_tool_start(
        serialized={"name": "noop"},
        input_str="",
        run_id=run_id,
        inputs=None,
    )
    await handler.on_tool_end(output="ok", run_id=run_id)

    span = exporter.get_finished_spans()[0]
    assert "tool_args_hash" not in span.attributes
    assert "tool_args_size" not in span.attributes


@pytest.mark.anyio
async def test_tool_name_falls_back_to_chain_id_tail(callback_and_exporter):
    """Some LangChain runnables only populate ``id`` not ``name``."""
    handler, exporter = callback_and_exporter
    run_id = uuid4()

    await handler.on_tool_start(
        serialized={"id": ["chain", "subchain", "fancy_tool"]},
        input_str="",
        run_id=run_id,
        inputs={},
    )
    await handler.on_tool_end(output="ok", run_id=run_id)

    span = exporter.get_finished_spans()[0]
    assert span.name == "tool.fancy_tool"
    assert span.attributes.get("tool_name") == "fancy_tool"


@pytest.mark.anyio
async def test_non_json_serializable_inputs_still_hash(callback_and_exporter):
    """``default=str`` lets us hash datetime / Path / custom objects."""
    from datetime import datetime, timezone

    handler, exporter = callback_and_exporter
    run_id = uuid4()

    await handler.on_tool_start(
        serialized={"name": "ts_tool"},
        input_str="",
        run_id=run_id,
        inputs={"ts": datetime(2026, 1, 1, tzinfo=timezone.utc)},
    )
    await handler.on_tool_end(output="ok", run_id=run_id)

    span = exporter.get_finished_spans()[0]
    h = span.attributes.get("tool_args_hash")
    assert isinstance(h, str) and len(h) == 16


@pytest.mark.anyio
async def test_unknown_run_id_on_end_is_noop(callback_and_exporter):
    """A late ``on_tool_end`` for a run we never saw must not crash."""
    handler, _ = callback_and_exporter
    await handler.on_tool_end(output="ok", run_id=uuid4())


@pytest.mark.anyio
async def test_tool_call_id_pulled_from_kwargs(callback_and_exporter):
    """Reviewer P2 fix: ``tool_call_id`` rides onto the tool span.

    LangChain's ``BaseTool._arun`` calls ``on_tool_start(... tool_call_id=...)``
    via ``**kwargs``. The handler reads it and surfaces it on the span
    so downstream consumers (ToolEvent / ApprovalCache / agent_task_runner)
    can correlate the span back to the originating tool_call.
    """
    handler, exporter = callback_and_exporter
    run_id = uuid4()

    await handler.on_tool_start(
        serialized={"name": "shell_execute"},
        input_str="",
        run_id=run_id,
        inputs={"cmd": "ls"},
        tool_call_id="call_abc123",
    )
    await handler.on_tool_end(output="ok", run_id=run_id)

    span = exporter.get_finished_spans()[0]
    assert span.attributes.get("tool_call_id") == "call_abc123"


@pytest.mark.anyio
async def test_tool_call_id_absent_when_not_provided(callback_and_exporter):
    """No ``tool_call_id`` in kwargs → attribute absent (tracer drops None)."""
    handler, exporter = callback_and_exporter
    run_id = uuid4()

    await handler.on_tool_start(
        serialized={"name": "x"},
        input_str="",
        run_id=run_id,
        inputs={},
    )
    await handler.on_tool_end(output="ok", run_id=run_id)

    span = exporter.get_finished_spans()[0]
    assert "tool_call_id" not in span.attributes


@pytest.mark.anyio
async def test_step_id_inherited_from_trace_context(callback_and_exporter):
    """Reviewer P2 fix: tool span carries ``step_id`` via the contextvar
    that ``traced_node`` binds while a node body is running.

    Simulates the production flow: ``traced_node`` binds a
    ``TraceContext`` with ``step_id`` for the duration of
    ``executor_node``'s body; tool calls inside fire this callback,
    which calls ``build_canonical_attributes()``, which reads the
    contextvar and pulls step_id automatically.
    """
    from app.domain.external.observability import TraceContext
    from app.infrastructure.observability.context import (
        reset_trace_context,
        set_trace_context,
    )

    handler, exporter = callback_and_exporter
    run_id = uuid4()

    ctx = TraceContext(
        trace_id="0123456789abcdef0123456789abcdef",
        request_id="11111111-1111-4111-8111-111111111111",
        event_id="22222222-2222-4222-8222-222222222222",
        graph_node="executor_node",
        step_id="step-from-ctx",
    )
    token = set_trace_context(ctx)
    try:
        await handler.on_tool_start(
            serialized={"name": "file_read"},
            input_str="",
            run_id=run_id,
            inputs={"path": "/tmp/x"},
        )
        await handler.on_tool_end(output="ok", run_id=run_id)
    finally:
        reset_trace_context(token)

    span = exporter.get_finished_spans()[0]
    assert span.attributes.get("step_id") == "step-from-ctx"
    assert span.attributes.get("graph_node") == "executor_node"
    # trace_id rides through too — same join key as the parent span.
    assert (
        span.attributes.get("trace_id")
        == "0123456789abcdef0123456789abcdef"
    )
