"""B5 PR-S2-2 reviewer fix: end-to-end observability wiring smoke.

Locks the contract that the **production wiring chain** holds:

   PlannerReActFlow._ensure_graphs   → build_main_graph(node_decorator=traced_node(tracer))
   PlannerReActFlow._build_config    → cfg["callbacks"] += [OtelToolSpanCallback]

Without these tests, the per-component unit tests would all stay green
even if planner_react never plugged the factories in (the original
reviewer P1 #1 finding). We:

1. Drive a real ``build_main_graph`` instance with the same surfaces
   as the production builder + a fake react_graph that fires the
   LangChain tool callback chain. Assert spans + attrs.
2. Source-level AST scan asserts ``planner_react._ensure_graphs`` and
   ``planner_react._build_config`` actually plug in the factories.

Pinned attrs
------------
- ``graph.node.executor_node`` span carries ``step_id`` (resolved from
  ``state["current_step"].id`` BEFORE the body runs — reviewer P1 #2 fix).
- ``tool.<name>`` span carries ``step_id`` via the contextvar that
  ``traced_node`` binds (reviewer P2 #3 part A fix).
- ``tool.<name>`` span carries ``tool_call_id`` from LangChain's kwargs
  (reviewer P2 #3 part B fix).
- All spans share the same ``trace_id`` (canonical join-key contract).
"""
from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from unittest.mock import AsyncMock, MagicMock

from app.application.composition import (
    build_observability_callbacks,
    build_traced_node_decorator,
)
from app.domain.models.llm_responses import (
    PlanResponse,
    PlanUpdateResponse,
    StepDef,
)
from app.infrastructure.observability.otel_tracer import OtelTracer


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_planner_llm() -> MagicMock:
    create = AsyncMock()
    create.ainvoke = AsyncMock(
        return_value=PlanResponse(
            title="t",
            goal="g",
            language="en",
            steps=[StepDef(description="step1")],
            message="m",
        )
    )
    update = AsyncMock()
    update.ainvoke = AsyncMock(
        return_value=PlanUpdateResponse(steps=[]),
    )
    llm = MagicMock()

    def _wso(schema, **kw):
        if schema is PlanResponse:
            return create
        if schema is PlanUpdateResponse:
            return update
        raise ValueError(schema)

    llm.with_structured_output = MagicMock(side_effect=_wso)

    async def _astream(messages, **kw):
        yield AIMessageChunk(content='{"message": "ok", "attachments": []}')

    llm.astream = _astream
    return llm


class _ToolCallbackFiringReactGraph:
    """Minimal react_graph that fires LangChain's tool callback once.

    Mimics what the real ``react_graph.tool_node`` does for one tool
    call: invokes ``on_tool_start`` + ``on_tool_end`` on every
    callback handler attached to the supplied ``config``, then yields
    a final AIMessage.
    """

    def __init__(self, tool_name: str, tool_call_id: str) -> None:
        self._tool_name = tool_name
        self._tool_call_id = tool_call_id

    async def astream(self, input_state, config=None, **kwargs):
        # LangGraph wraps cfg["callbacks"] into a CallbackManager before
        # handing it to subgraphs; pull the actual handlers off the
        # ``handlers`` / ``inheritable_handlers`` attributes when present,
        # otherwise treat the value as a plain iterable.
        raw = (config or {}).get("callbacks")
        if raw is None:
            handlers: list[Any] = []
        elif hasattr(raw, "handlers"):
            handlers = list(raw.handlers) + list(
                getattr(raw, "inheritable_handlers", []) or []
            )
        else:
            handlers = list(raw)
        run_id = uuid4()
        for cb in handlers:
            on_start = getattr(cb, "on_tool_start", None)
            if on_start is not None:
                await on_start(
                    serialized={"name": self._tool_name},
                    input_str="",
                    run_id=run_id,
                    inputs={"path": "/tmp/x"},
                    tool_call_id=self._tool_call_id,
                )
        for cb in handlers:
            on_end = getattr(cb, "on_tool_end", None)
            if on_end is not None:
                await on_end(output="ok", run_id=run_id)
        yield {
            "llm_node": {
                "events": [],
                "messages": [
                    AIMessage(
                        content='{"success": true, "result": "done", "attachments": []}'
                    ),
                ],
            }
        }


async def test_e2e_node_and_tool_spans_share_trace_id_and_step_id():
    """Drive one full ``build_main_graph.ainvoke`` pass and assert:

    - ``graph.node.*`` spans emit (proves the decorator was wired).
    - ``tool.<name>`` span emits (proves the callback was wired).
    - All spans share ``trace_id`` (canonical join key).
    - The executor node span + the tool span carry ``step_id``
      (resolved via state.current_step.id and the contextvar).
    - The tool span carries ``tool_call_id`` from LangChain kwargs.
    """
    from app.domain.services.graphs.main_graph import build_main_graph

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OtelTracer(provider.get_tracer("actus-e2e"))

    decorator = build_traced_node_decorator(tracer)
    callbacks = build_observability_callbacks(tracer)

    mock_uow = AsyncMock()
    mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
    mock_uow.__aexit__ = AsyncMock(return_value=False)
    mock_uow.session = AsyncMock()
    mock_uow.session.get_skill_graph_state = AsyncMock(return_value=None)
    mock_uow.session.get_summary = AsyncMock(return_value=[])

    planner_llm = _make_planner_llm()
    react = _ToolCallbackFiringReactGraph(
        tool_name="file_read",
        tool_call_id="call_abc123",
    )

    graph = build_main_graph(
        _allow_default_prompt_assembler=True,
        planner_llm=planner_llm,
        react_graph=react,
        summary_llm=planner_llm,
        uow_factory=MagicMock(return_value=mock_uow),
        session_id="sess-1",
        node_decorator=decorator,
    )

    # Production binds a ``TraceContext`` per HTTP request via
    # ``ObservabilityMiddleware`` so all spans within the request
    # share trace_id / request_id. Replicate that here so the
    # canonical-builder fallback path doesn't mint a fresh uuid per
    # span emit.
    from app.domain.external.observability import TraceContext
    from app.infrastructure.observability.context import (
        reset_trace_context,
        set_trace_context,
    )

    pre = TraceContext(
        trace_id="0123456789abcdef0123456789abcdef",
        request_id="11111111-1111-4111-8111-111111111111",
        event_id="22222222-2222-4222-8222-222222222222",
    )
    token = set_trace_context(pre)
    try:
        # Mimic ``ObservabilityMiddleware``'s OTel root span so child
        # spans (graph nodes, tools) share native OTel trace_id via
        # context propagation. Without this wrap the canonical attr
        # trace_id stays consistent but the native OTel trace tree
        # fragments — exactly what reviewer round-3 P1 caught in prod.
        with tracer.start_as_current_span("http.request"):
            await graph.ainvoke(
                {
                    "message": "do thing",
                    "language": "en",
                    "attachments": [],
                    "image_content_blocks": [],
                    "plan": None,
                },
                config={"configurable": {}, "callbacks": callbacks},
            )
    finally:
        reset_trace_context(token)

    spans = exporter.get_finished_spans()
    assert spans, "no spans emitted — wiring is broken"

    by_name: dict[str, Any] = {s.name: s for s in spans}

    # 1) ``graph.node.*`` spans emitted (decorator wired).
    assert "graph.node.planner_node" in by_name
    assert "graph.node.executor_node" in by_name
    assert "graph.node.updater_node" in by_name

    # 2) ``tool.file_read`` span emitted (tool span callback wired).
    assert "tool.file_read" in by_name

    # The "http.request" wrapper span is intentionally a bare envelope
    # — it does not carry the canonical attr ``trace_id`` (only graph
    # nodes / tools do). Filter it out of the attr-consistency check
    # so the assertion targets the canonical-emitting spans only.
    canonical_spans = [s for s in spans if s.name != "http.request"]

    # 3a) Every canonical-emitting span shares the SAME attr trace_id
    # (Actus join key).
    trace_ids = {s.attributes.get("trace_id") for s in canonical_spans}
    assert len(trace_ids) == 1, f"attr trace_id not consistent: {trace_ids}"

    # 3b) Every span (including the root) shares the SAME OTel NATIVE
    # trace_id (OTel UI key). Reviewer round-3 P1 lock: pre-binding
    # the Actus TraceContext is not enough — the native OTel trace
    # tree must also be unified. In this test we wrap the ainvoke in
    # a parent OTel span (mimicking ``ObservabilityMiddleware`` in
    # production); the decorator's child spans inherit via OTel
    # context propagation.
    native_trace_ids = {s.context.trace_id for s in spans}
    assert len(native_trace_ids) == 1, (
        f"NATIVE OTel trace_id not consistent (request fragments into "
        f"multiple OTel traces): {native_trace_ids}"
    )

    # 4) Reviewer P1 #2: executor span resolves step_id from state.
    exec_span = by_name["graph.node.executor_node"]
    step_id = exec_span.attributes.get("step_id")
    assert isinstance(step_id, str) and step_id, (
        f"executor span must carry step_id (reviewer P1 #2); got {step_id!r}"
    )

    # 5) Reviewer P2 #3: tool span carries step_id (via contextvar) +
    # tool_call_id (from LangChain kwargs).
    tool_span = by_name["tool.file_read"]
    assert tool_span.attributes.get("step_id") == step_id, (
        "tool span must inherit step_id from executor's bound TraceContext "
        "(reviewer P2 #3 part A)"
    )
    assert tool_span.attributes.get("tool_call_id") == "call_abc123", (
        "tool span must carry tool_call_id from LangChain kwargs "
        "(reviewer P2 #3 part B)"
    )


async def test_observability_middleware_opens_otel_root_span_with_matching_native_trace_id():
    """Reviewer round-3 P1: ``ObservabilityMiddleware`` MUST open an
    OTel root span around the request so child spans (graph nodes,
    tools) inherit the same OTel **native** ``trace_id`` via OTel
    context propagation. Otherwise Phoenix / Jaeger trace trees are
    fragmented even though Actus canonical attr ``trace_id`` is
    consistent.

    Additionally: the middleware uses the OTel native trace_id as the
    canonical Actus ``trace_id`` (32-hex format), so the ATTR-based
    join key and the NATIVE trace tree key are the SAME string —
    single source of truth.
    """
    from app.infrastructure.observability import (
        setup_observability,
        teardown_observability,
    )
    from app.interfaces.middlewares.observability_middleware import (
        ObservabilityMiddleware,
    )
    from opentelemetry import trace as otel_trace

    teardown_observability()
    providers = setup_observability()
    # Wire an in-memory exporter onto the SAME global TracerProvider
    # the middleware will use (``OtelTracer()`` reads the global).
    exporter = InMemorySpanExporter()
    providers.tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    try:
        # Drive the middleware via a tiny ASGI inner app that opens
        # one child span (mimicking what ``traced_node`` does).
        captured_native_trace_id: list[int] = []

        async def inner_app(scope, receive, send):
            tracer = OtelTracer()
            with tracer.start_as_current_span("graph.node.executor_node"):
                # Capture the live OTel trace_id from inside the request
                # so we can assert it matches the middleware's root.
                captured_native_trace_id.append(
                    otel_trace.get_current_span()
                    .get_span_context()
                    .trace_id
                )
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        middleware = ObservabilityMiddleware(inner_app)

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "headers": [],
        }
        sent: list[Any] = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(msg):
            sent.append(msg)

        await middleware(scope, receive, send)

        spans = exporter.get_finished_spans()
        # At least 2 spans: the http root + the executor child.
        names = [s.name for s in spans]
        assert any(n.startswith("http.") for n in names), (
            f"middleware must open an http.<method> root span; got {names}"
        )
        assert "graph.node.executor_node" in names

        root_span = next(s for s in spans if s.name.startswith("http."))
        child_span = next(s for s in spans if s.name == "graph.node.executor_node")

        # 1) Native OTel trace_id: child shares root's trace.
        assert root_span.context.trace_id == child_span.context.trace_id, (
            "child span MUST inherit OTel native trace_id via context "
            "propagation (reviewer round-3 P1)"
        )
        # 2) Child has parent = root.
        assert child_span.parent is not None
        assert child_span.parent.span_id == root_span.context.span_id

        # 3) Captured native trace_id from inside request matches root.
        assert captured_native_trace_id
        assert captured_native_trace_id[0] == root_span.context.trace_id

        # 4) Actus canonical attr trace_id == OTel native trace_id (32-hex).
        bound_ctx = scope.get("actus_trace_context")
        assert bound_ctx is not None
        expected_hex = format(root_span.context.trace_id, "032x")
        assert bound_ctx.trace_id == expected_hex, (
            "middleware must source canonical Actus trace_id from the "
            "OTel native trace_id so attr join key == native trace key"
        )
    finally:
        teardown_observability()


async def test_no_context_node_and_tool_spans_share_trace_id():
    """Reviewer round-2 P2 fix: the no-context path (CLI / startup /
    background task — no ``ObservabilityMiddleware``-bound
    ``TraceContext``) must NOT split the canonical ``trace_id``
    between the node span and tool spans inside its body.

    Reproduction without the fix:
      - ``traced_node`` calls ``build_canonical_attributes()`` once
        for node span attrs → fresh fallback ``trace_id_A``.
      - ``_bind_node_context`` calls ``build_canonical_attributes()``
        a second time for the synthetic context → fresh fallback
        ``trace_id_B``.
      - Tool callback inside the body reads contextvar → ``trace_id_B``.
      - Node span has ``trace_id_A``; tool span has ``trace_id_B`` —
        canonical join key diverges.

    The fix binds the synthetic context FIRST, then builds node span
    attrs from the bound contextvar. This test does NOT pre-bind a
    ``TraceContext`` (unlike the e2e test above) so the no-context
    path is exercised directly.
    """
    from app.domain.services.graphs.main_graph import build_main_graph
    from app.infrastructure.observability.context import (
        get_trace_context,
        reset_trace_context,
        set_trace_context,
    )

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OtelTracer(provider.get_tracer("actus-no-ctx"))

    decorator = build_traced_node_decorator(tracer)
    callbacks = build_observability_callbacks(tracer)

    mock_uow = AsyncMock()
    mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
    mock_uow.__aexit__ = AsyncMock(return_value=False)
    mock_uow.session = AsyncMock()
    mock_uow.session.get_skill_graph_state = AsyncMock(return_value=None)
    mock_uow.session.get_summary = AsyncMock(return_value=[])

    planner_llm = _make_planner_llm()
    react = _ToolCallbackFiringReactGraph(
        tool_name="x",
        tool_call_id="call-x",
    )

    graph = build_main_graph(
        _allow_default_prompt_assembler=True,
        planner_llm=planner_llm,
        react_graph=react,
        summary_llm=planner_llm,
        uow_factory=MagicMock(return_value=mock_uow),
        session_id="sess-1",
        node_decorator=decorator,
    )

    # Defensive precondition: explicitly clear any contextvar lingering
    # from a previous test (other tests in this module pre-bind one).
    cleared = None
    if get_trace_context() is not None:
        cleared = set_trace_context(None)
    try:
        await graph.ainvoke(
            {
                "message": "do thing",
                "language": "en",
                "attachments": [],
                "image_content_blocks": [],
                "plan": None,
            },
            config={"configurable": {}, "callbacks": callbacks},
        )
    finally:
        if cleared is not None:
            reset_trace_context(cleared)

    spans = exporter.get_finished_spans()
    by_name: dict[str, Any] = {s.name: s for s in spans}
    exec_span = by_name["graph.node.executor_node"]
    tool_span = by_name["tool.x"]

    # Both spans non-empty trace_id...
    exec_trace_id = exec_span.attributes.get("trace_id")
    tool_trace_id = tool_span.attributes.get("trace_id")
    assert isinstance(exec_trace_id, str) and exec_trace_id
    assert isinstance(tool_trace_id, str) and tool_trace_id
    # ...AND identical (canonical join key).
    assert exec_trace_id == tool_trace_id, (
        "no-context path must NOT split trace_id between node span and "
        f"tool span: exec={exec_trace_id!r} vs tool={tool_trace_id!r}"
    )
    # request_id is also part of the canonical join surface — same fix
    # naturally lines them up.
    assert exec_span.attributes.get("request_id") == tool_span.attributes.get(
        "request_id"
    )


async def test_http_route_uses_template_not_raw_path_no_pii_leak():
    """Reviewer round-4 P2: ``http.route`` must be the matched route
    TEMPLATE (e.g. ``/sessions/{session_id}/chat``), not the raw URL.

    Why this matters:
    - **Cardinality**: real path-param values (session ids, object
      ids) explode OTel cardinality if surfaced as ``http.route``.
      One dimension per session breaks histograms / metric backends.
    - **PII / privacy**: surfacing literal URLs leaks user-tied
      identifiers into span attrs visible in Phoenix / Jaeger.

    The middleware backfills ``http.route`` from ``scope["route"].path``
    (Starlette match template) AFTER routing finishes — driven by the
    ``http.response.start`` send event. This test exercises the real
    Starlette routing layer with a parameterized route and a canary
    session id, then asserts:
    - ``http.route`` carries the TEMPLATE (``/sessions/{session_id}/chat``).
    - The canary session id appears nowhere in any span attribute
      (key or value).
    """
    # Use FastAPI (not raw Starlette) because production runs FastAPI
    # and FastAPI's ``APIRoute`` sets ``scope["route"]`` after routing
    # — which is what our middleware reads. Raw Starlette ``Route``
    # does NOT set this scope key, so a starlette-only test would
    # exercise a different code path.
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from app.infrastructure.observability import (
        setup_observability,
        teardown_observability,
    )
    from app.interfaces.middlewares.observability_middleware import (
        ObservabilityMiddleware,
    )

    teardown_observability()
    providers = setup_observability()
    exporter = InMemorySpanExporter()
    providers.tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    canary_session = "CANARY-SESSION-aXf9zQ"

    try:
        api = FastAPI()

        @api.post("/sessions/{session_id}/chat")
        async def chat_endpoint(session_id: str):
            return {"ok": True}

        api.add_middleware(ObservabilityMiddleware)

        client = TestClient(api)
        resp = client.post(f"/sessions/{canary_session}/chat")
        assert resp.status_code == 200

        spans = exporter.get_finished_spans()
        root = next(s for s in spans if s.name.startswith("http."))

        # 1) http.route is the TEMPLATE, not the raw URL.
        route_attr = root.attributes.get("http.route")
        assert route_attr == "/sessions/{session_id}/chat", (
            f"http.route must be the route template; got {route_attr!r}"
        )

        # 2) Canary session id NEVER appears in any attribute (key
        # or value) on any span emitted from this request.
        for span in spans:
            for k, v in span.attributes.items():
                assert canary_session not in str(k), (
                    f"canary session id leaked into attr key on {span.name}"
                )
                assert canary_session not in str(v), (
                    f"canary session id leaked into attr {k}={v!r} "
                    f"on {span.name}"
                )
    finally:
        teardown_observability()


async def test_http_route_absent_when_route_unmatched():
    """When no route matches (404 / unknown path), ``scope["route"]``
    is unset — the middleware MUST skip the ``http.route`` attr
    entirely rather than fall back to the raw URL. Otherwise an
    attacker probing arbitrary URLs (``/admin/secret/{guess}``) would
    flood span attrs with attacker-controlled cardinality.
    """
    from app.infrastructure.observability import (
        setup_observability,
        teardown_observability,
    )
    from app.interfaces.middlewares.observability_middleware import (
        ObservabilityMiddleware,
    )

    teardown_observability()
    providers = setup_observability()
    exporter = InMemorySpanExporter()
    providers.tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    try:
        async def inner_app(scope, receive, send):
            # NO route matching — ``scope["route"]`` stays unset.
            await send({"type": "http.response.start", "status": 404, "headers": []})
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        wrapped = ObservabilityMiddleware(inner_app)
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/probe/CANARY-PROBE-7zK9Mq",
            "headers": [],
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(_msg):
            return None

        await wrapped(scope, receive, send)

        root = next(
            s for s in exporter.get_finished_spans() if s.name.startswith("http.")
        )
        # http.route attr absent (no template available).
        assert "http.route" not in root.attributes
        # And no span attr leaks the raw probe path.
        for k, v in root.attributes.items():
            assert "CANARY-PROBE-7zK9Mq" not in str(v), (
                f"raw probe path leaked into attr {k}={v!r}"
            )
    finally:
        teardown_observability()


async def test_root_span_records_http_status_code_200_ok():
    """Reviewer round-5 P2: 2xx response → ``http.status_code`` set,
    span status NOT marked ERROR (server side success)."""
    from fastapi import FastAPI
    from opentelemetry.trace import StatusCode
    from starlette.testclient import TestClient

    from app.infrastructure.observability import (
        setup_observability,
        teardown_observability,
    )
    from app.interfaces.middlewares.observability_middleware import (
        ObservabilityMiddleware,
    )

    teardown_observability()
    providers = setup_observability()
    exporter = InMemorySpanExporter()
    providers.tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    try:
        api = FastAPI()

        @api.get("/ok")
        async def ok():
            return {"ok": True}

        api.add_middleware(ObservabilityMiddleware)
        client = TestClient(api)
        resp = client.get("/ok")
        assert resp.status_code == 200

        root = next(
            s for s in exporter.get_finished_spans() if s.name.startswith("http.")
        )
        assert root.attributes.get("http.status_code") == 200
        # Span status is OTel "OK" (or "UNSET"); 2xx must NOT flip to ERROR.
        assert root.status.status_code != StatusCode.ERROR
    finally:
        teardown_observability()


async def test_root_span_records_http_status_code_404_not_error():
    """Reviewer round-5 P2: 4xx response → ``http.status_code`` set,
    span status NOT marked ERROR. 4xx is a client-observable outcome
    (missing route, invalid input, auth rejection) — not a server
    fault. Per OTel server-span semantic guidance only ``>=500``
    flips span status to ERROR.
    """
    from fastapi import FastAPI
    from opentelemetry.trace import StatusCode
    from starlette.testclient import TestClient

    from app.infrastructure.observability import (
        setup_observability,
        teardown_observability,
    )
    from app.interfaces.middlewares.observability_middleware import (
        ObservabilityMiddleware,
    )

    teardown_observability()
    providers = setup_observability()
    exporter = InMemorySpanExporter()
    providers.tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    try:
        api = FastAPI()
        api.add_middleware(ObservabilityMiddleware)
        client = TestClient(api)
        # No route registered → 404 from Starlette router.
        resp = client.get("/does-not-exist")
        assert resp.status_code == 404

        root = next(
            s for s in exporter.get_finished_spans() if s.name.startswith("http.")
        )
        assert root.attributes.get("http.status_code") == 404
        assert root.status.status_code != StatusCode.ERROR
    finally:
        teardown_observability()


async def test_root_span_records_http_status_code_500_marks_error():
    """Reviewer round-5 P2: handled 5xx response → ``http.status_code``
    set, OTel span status flipped to ERROR so dashboards can compute
    error rate by span status without parsing attrs.

    Uses ``HTTPException(500)`` from inside the route — FastAPI's
    registered handler converts it to a normal 500 response that
    flows through the send wrapper as ``http.response.start`` with
    ``status=500``. This is the production-equivalent path for
    ``AppException`` / ``HTTPException`` that the
    ``ServerErrorMiddleware`` does NOT see.
    """
    from fastapi import FastAPI, HTTPException
    from opentelemetry.trace import StatusCode
    from starlette.testclient import TestClient

    from app.infrastructure.observability import (
        setup_observability,
        teardown_observability,
    )
    from app.interfaces.middlewares.observability_middleware import (
        ObservabilityMiddleware,
    )

    teardown_observability()
    providers = setup_observability()
    exporter = InMemorySpanExporter()
    providers.tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    try:
        api = FastAPI()

        @api.get("/boom")
        async def boom():
            raise HTTPException(status_code=500, detail="kaboom")

        api.add_middleware(ObservabilityMiddleware)
        client = TestClient(api)
        resp = client.get("/boom")
        assert resp.status_code == 500

        root = next(
            s for s in exporter.get_finished_spans() if s.name.startswith("http.")
        )
        assert root.attributes.get("http.status_code") == 500
        assert root.status.status_code == StatusCode.ERROR


    finally:
        teardown_observability()


async def test_root_span_records_500_on_uncaught_exception_catch_all_path():
    """Reviewer round-6 P2: an uncaught ``Exception`` (no FastAPI
    handler claims it) bubbles past inner middleware to
    ``ServerErrorMiddleware`` which renders 500 and re-raises. The
    re-raised exception propagates back to ``ObservabilityMiddleware``,
    which MUST set ``http.status_code=500`` + ``http.route`` (if route
    matched) + OTel ERROR status BEFORE re-raising. Without this fix
    the catch-all path would emit a root span with only
    ``http.method`` and a magic-set ERROR status (from OTel's
    auto-status-on-exit) but no ``http.status_code`` attr — invisible
    to dashboards keyed by status code.
    """
    from fastapi import FastAPI
    from opentelemetry.trace import StatusCode
    from starlette.testclient import TestClient

    from app.infrastructure.observability import (
        setup_observability,
        teardown_observability,
    )
    from app.interfaces.middlewares.observability_middleware import (
        ObservabilityMiddleware,
    )

    teardown_observability()
    providers = setup_observability()
    exporter = InMemorySpanExporter()
    providers.tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    try:
        api = FastAPI()

        @api.get("/sessions/{session_id}/crash")
        async def crash(session_id: str):
            raise RuntimeError("kaboom (uncaught)")

        api.add_middleware(ObservabilityMiddleware)
        # ``raise_server_exceptions=False`` — let ServerErrorMiddleware
        # render 500 and let the response surface; otherwise TestClient
        # re-raises the underlying RuntimeError.
        client = TestClient(api, raise_server_exceptions=False)
        resp = client.get("/sessions/abc-123/crash")
        assert resp.status_code == 500

        root = next(
            s for s in exporter.get_finished_spans() if s.name.startswith("http.")
        )
        assert root.attributes.get("http.status_code") == 500, (
            f"catch-all 500 path must set http.status_code; got "
            f"{dict(root.attributes)!r}"
        )
        # Route matched before the handler raised → template backfilled.
        assert root.attributes.get("http.route") == (
            "/sessions/{session_id}/crash"
        )
        assert root.status.status_code == StatusCode.ERROR
        # Path-param canary not in attrs.
        for k, v in root.attributes.items():
            assert "abc-123" not in str(v), (
                f"path-param leaked into attr {k}={v!r}"
            )
    finally:
        teardown_observability()


async def test_cancelled_error_does_not_taint_root_span_as_500():
    """Reviewer round-7 P2: ``asyncio.CancelledError`` is control flow
    (SSE disconnect, client abort, upstream cancel, lifespan
    shutdown), NOT a server fault. Middleware MUST:

    1. Re-raise ``CancelledError`` without backfilling 500 attrs.
    2. Pre-set span status to ``OK`` so OTel's auto-ERROR-on-exit
       (in the ``start_as_current_span`` context-manager) doesn't
       flip status.
    3. Still run the ``finally`` block to reset the contextvar so
       subsequent requests on the same task start clean.

    Otherwise legitimate cancellations would pollute 5xx error-rate
    dashboards on every SSE disconnect.
    """
    import asyncio

    from opentelemetry.trace import StatusCode

    from app.infrastructure.observability import (
        setup_observability,
        teardown_observability,
    )
    from app.infrastructure.observability.context import get_trace_context
    from app.interfaces.middlewares.observability_middleware import (
        ObservabilityMiddleware,
    )

    teardown_observability()
    providers = setup_observability()
    exporter = InMemorySpanExporter()
    providers.tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    try:
        async def cancelling_app(scope, receive, send):
            # Raise BEFORE sending any response — mirrors SSE
            # generator cancel-on-disconnect / lifespan task cancel.
            raise asyncio.CancelledError()

        wrapped = ObservabilityMiddleware(cancelling_app)
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/sse/stream",
            "headers": [],
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(_msg):
            return None

        # CancelledError MUST propagate out untouched.
        with pytest.raises(asyncio.CancelledError):
            await wrapped(scope, receive, send)

        # finally branch ran → contextvar reset to None.
        assert get_trace_context() is None, (
            "middleware finally MUST reset contextvar even on cancel"
        )

        spans = exporter.get_finished_spans()
        root = next(s for s in spans if s.name.startswith("http."))

        # No http.status_code attr (no response was produced).
        assert "http.status_code" not in root.attributes, (
            f"CancelledError must NOT backfill http.status_code; "
            f"got {dict(root.attributes)!r}"
        )

        # Span status NOT ERROR — pre-set to OK suppresses OTel's
        # auto-ERROR-on-exception in the with-block exit.
        assert root.status.status_code != StatusCode.ERROR, (
            f"CancelledError must NOT mark span ERROR; "
            f"got {root.status!r}"
        )
    finally:
        teardown_observability()


async def test_cancelled_error_after_500_response_keeps_error_status():
    """Reviewer round-8 P2: when a ``>=500`` response has ALREADY
    streamed through the send wrapper (so ``http.status_code=500``
    + ``StatusCode.ERROR`` are recorded), a subsequent
    ``CancelledError`` (e.g. body chunk write cancelled by client
    disconnect) MUST NOT overwrite the ERROR with OK.

    OTel SDK allows ``ERROR → OK`` transitions, so the bare
    ``set_status(OK)`` from round-7 would silently erase a real
    server fault from error-rate dashboards.

    Strategy: drive a minimal ASGI app that:
    1. Sends ``http.response.start`` with ``status=500`` (legitimate
       5xx response — handled exception path).
    2. Then raises ``CancelledError`` BEFORE the body finishes
       (mirrors a real client disconnect mid-error-response).

    Assert: root span retains ``http.status_code=500`` and
    ``span.status.status_code == StatusCode.ERROR``.
    """
    import asyncio

    from opentelemetry.trace import StatusCode

    from app.infrastructure.observability import (
        setup_observability,
        teardown_observability,
    )
    from app.interfaces.middlewares.observability_middleware import (
        ObservabilityMiddleware,
    )

    teardown_observability()
    providers = setup_observability()
    exporter = InMemorySpanExporter()
    providers.tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    try:
        async def five_hundred_then_cancel(scope, receive, send):
            # Step 1: emit the 500 response start (send wrapper sees
            # this and records http.status_code=500 + ERROR status).
            await send({
                "type": "http.response.start",
                "status": 500,
                "headers": [],
            })
            # Step 2: cancel before body completes.
            raise asyncio.CancelledError()

        wrapped = ObservabilityMiddleware(five_hundred_then_cancel)
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/x",
            "headers": [],
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(_msg):
            return None

        with pytest.raises(asyncio.CancelledError):
            await wrapped(scope, receive, send)

        spans = exporter.get_finished_spans()
        root = next(s for s in spans if s.name.startswith("http."))

        # The 500 outcome MUST be preserved through the cancel.
        assert root.attributes.get("http.status_code") == 500, (
            "5xx already emitted via send wrapper must NOT be erased "
            f"by subsequent CancelledError; got "
            f"{dict(root.attributes)!r}"
        )
        assert root.status.status_code == StatusCode.ERROR, (
            "ERROR status must NOT be downgraded to OK by a "
            f"post-5xx CancelledError; got {root.status!r}"
        )
    finally:
        teardown_observability()


async def test_cancelled_error_after_2xx_response_does_not_become_error():
    """Symmetric round-8 lock: a CancelledError after a 2xx response
    started (e.g. SSE stream emits headers + first chunks then the
    client disconnects mid-stream) is still a cancellation — span
    status should be OK, not ERROR.

    Without round-8 the same bug-pattern could go the other way:
    if we ONLY checked "any prior status started → leave alone", a
    cancelled 2xx would inherit OTel's auto-ERROR. The fix policy is
    "suppress to OK iff started is None OR started < 500".
    """
    import asyncio

    from opentelemetry.trace import StatusCode

    from app.infrastructure.observability import (
        setup_observability,
        teardown_observability,
    )
    from app.interfaces.middlewares.observability_middleware import (
        ObservabilityMiddleware,
    )

    teardown_observability()
    providers = setup_observability()
    exporter = InMemorySpanExporter()
    providers.tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    try:
        async def two_hundred_then_cancel(scope, receive, send):
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [],
            })
            raise asyncio.CancelledError()

        wrapped = ObservabilityMiddleware(two_hundred_then_cancel)
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/sse",
            "headers": [],
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(_msg):
            return None

        with pytest.raises(asyncio.CancelledError):
            await wrapped(scope, receive, send)

        root = next(
            s for s in exporter.get_finished_spans() if s.name.startswith("http.")
        )
        assert root.attributes.get("http.status_code") == 200
        assert root.status.status_code != StatusCode.ERROR, (
            f"2xx + cancel must not become ERROR; got {root.status!r}"
        )
    finally:
        teardown_observability()


async def test_planner_react_build_config_includes_observability_callbacks():
    """Source-level lock: ``PlannerReActFlow._build_config`` MUST merge
    ``build_observability_callbacks()`` into ``cfg["callbacks"]``.
    Reviewer P1 #1 fix.
    """
    import ast
    from pathlib import Path

    src_path = (
        Path(__file__).resolve().parents[3]
        / "app"
        / "domain"
        / "services"
        / "flows"
        / "planner_react.py"
    )
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == "_build_config":
            found.append(ast.unparse(node))
    assert found, "_build_config not found on PlannerReActFlow"
    body = "\n".join(found)
    assert "build_observability_callbacks" in body, (
        "PlannerReActFlow._build_config must call "
        "build_observability_callbacks() and merge it into cfg['callbacks'] "
        "(reviewer P1 #1)"
    )


async def test_planner_react_ensure_graphs_passes_node_decorator():
    """Source-level lock: ``PlannerReActFlow._ensure_graphs`` MUST pass
    ``node_decorator=build_traced_node_decorator()`` to ``build_main_graph``.
    Reviewer P1 #1 fix.
    """
    import ast
    from pathlib import Path

    src_path = (
        Path(__file__).resolve().parents[3]
        / "app"
        / "domain"
        / "services"
        / "flows"
        / "planner_react.py"
    )
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == "_ensure_graphs":
            found.append(ast.unparse(node))
    assert found, "_ensure_graphs not found"
    body = "\n".join(found)
    assert "build_traced_node_decorator" in body, (
        "PlannerReActFlow._ensure_graphs must call "
        "build_traced_node_decorator() (reviewer P1 #1)"
    )
    assert "node_decorator=" in body, "node_decorator kwarg missing on build_main_graph call"
