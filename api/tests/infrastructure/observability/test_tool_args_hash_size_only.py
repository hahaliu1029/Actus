"""B5 PR-S2-4 acceptance: tool span carries hash + size, NEVER raw args.

The privacy contract for tool spans (TODO2.md #13 Sprint 2 PR-S2-2):

> 禁止把 raw args 写进 span — PR-S2-4 acceptance 必须有
> ``test_tool_args_hash_size_only.py``.

This file is the canonical lock. It feeds the tool span emitter a
payload full of distinctive **canary tokens** (file paths, secrets,
PII, file content blobs — synthetic, not production data) and asserts
NONE of those tokens appear in ANY span attribute value. Only
``tool_args_hash`` (16 hex chars) and ``tool_args_size`` (an int) are
allowed to describe the input.

Because OTel exports flat ``(key, value)`` pairs, the assertion is a
simple "no canary substring in any stringified attribute value" sweep
across every emitted span. If a future refactor accidentally adds a
``tool_input`` / ``tool_args`` / ``raw_inputs`` attribute, the canary
strings would surface and this test would fail loudly.
"""
from __future__ import annotations

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


_CANARY_PATH = "/secret/path/CANARY_PATH_aXf9zQ"
_CANARY_VALUE = "CANARY_VALUE_b8K2pLm"
_CANARY_TOKEN = "sk-CANARY_TOKEN_3vY7nDq"
_CANARY_FILE_BODY = "BEGIN CANARY FILE BODY 7zR9Ms END"


@pytest.fixture
def emitter() -> tuple[OtelToolSpanCallback, InMemorySpanExporter]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OtelTracer(provider.get_tracer("actus-test"))
    return OtelToolSpanCallback(tracer), exporter


@pytest.mark.anyio
async def test_no_canary_appears_in_any_span_attribute(emitter):
    handler, exporter = emitter
    run_id = uuid4()

    inputs = {
        "path": _CANARY_PATH,
        "value": _CANARY_VALUE,
        "auth_token": _CANARY_TOKEN,
        "body": _CANARY_FILE_BODY,
    }
    await handler.on_tool_start(
        serialized={"name": "file_write"},
        input_str=str(inputs),
        run_id=run_id,
        inputs=inputs,
    )
    await handler.on_tool_end(output=_CANARY_FILE_BODY, run_id=run_id)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    canaries = (
        _CANARY_PATH,
        _CANARY_VALUE,
        _CANARY_TOKEN,
        _CANARY_FILE_BODY,
    )

    for k, v in span.attributes.items():
        for canary in canaries:
            assert canary not in str(k), (
                f"canary {canary!r} leaked into attr key {k!r}"
            )
            assert canary not in str(v), (
                f"canary {canary!r} leaked into attr {k}={v!r}"
            )

    # The hash + size summary IS allowed (and required) — assert they
    # are present so a future refactor that drops them altogether also
    # trips this test.
    assert isinstance(span.attributes.get("tool_args_hash"), str)
    assert len(span.attributes.get("tool_args_hash")) == 16
    assert isinstance(span.attributes.get("tool_args_size"), int)
    assert span.attributes.get("tool_args_size") > 0


@pytest.mark.anyio
async def test_no_forbidden_attribute_keys_set(emitter):
    """Belt-and-braces: the forbidden key list never appears.

    Even an empty-string value under ``tool_args`` would be a contract
    violation since downstream consumers might union schemas across
    span sources and start expecting the key.
    """
    handler, exporter = emitter
    run_id = uuid4()

    await handler.on_tool_start(
        serialized={"name": "x"},
        input_str="",
        run_id=run_id,
        inputs={"k": _CANARY_VALUE},
    )
    await handler.on_tool_end(output="", run_id=run_id)

    span = exporter.get_finished_spans()[0]
    forbidden = {
        "tool_args",
        "tool_input",
        "tool_inputs",
        "args",
        "raw_args",
        "input_str",
        "raw_inputs",
    }
    leaked = set(span.attributes.keys()) & forbidden
    assert not leaked, f"forbidden raw-arg keys leaked into span attrs: {leaked}"


@pytest.mark.anyio
async def test_event_payload_does_not_carry_canary(emitter):
    """OTel spans also carry event-attached attribute bundles. A future
    error-path that hands ``record_exception`` a real exception with a
    sensitive message could leak canaries into event attrs. Lock the
    on-error path stays clean.
    """
    handler, exporter = emitter
    run_id = uuid4()

    await handler.on_tool_start(
        serialized={"name": "x"},
        input_str="",
        run_id=run_id,
        inputs={"path": _CANARY_PATH, "token": _CANARY_TOKEN},
    )
    await handler.on_tool_error(
        error=RuntimeError(f"failed processing {_CANARY_PATH} {_CANARY_TOKEN}"),
        run_id=run_id,
    )

    span = exporter.get_finished_spans()[0]
    for event in span.events:
        for k, v in (event.attributes or {}).items():
            assert _CANARY_PATH not in str(v), (
                f"canary path leaked into event attr {k}={v!r}"
            )
            assert _CANARY_TOKEN not in str(v), (
                f"canary token leaked into event attr {k}={v!r}"
            )
