"""B5 PR-S2-2 + FOLLOW-10: ``traced_node`` decorator factory.

Wraps a LangGraph async node coroutine in an OTel span so each
``planner_node`` / ``executor_node`` / ``updater_node`` / ``summarizer_node``
invocation produces a ``graph.node.<name>`` span.

Step-id resolution order (FOLLOW-10 + reviewer P1 fix)
-------------------------------------------------------
1. ``state["current_step"].id`` — the planner pins ``current_step`` on
   state BEFORE the runtime calls ``executor_node``, so the wrapper
   sees the live step before the body runs. This is what makes the
   ``graph.node.executor_node`` span carry the right step_id (the
   prior version pulled from ``configurable.step_id`` which the body
   itself populates — too late for the OUTER span).
2. ``config["configurable"]["step_id"]`` — the explicit injection
   path from ``executor_node``'s ``fresh_configurable`` (used by the
   inner react subgraph + tool callbacks, and as a fallback when state
   doesn't carry ``current_step``).

In both cases ``build_canonical_attributes`` produces the full
canonical snapshot (trace_id / request_id / session_id / event_id /
graph_node / step_id / ...) so the resulting span attribute set
satisfies the v1 join-key contract.

Contextvar binding
------------------
The wrapper also binds a ``TraceContext`` for the duration of the
node body so that **downstream emit sites** — including the
LangChain tool callback (``OtelToolSpanCallback``) — read the same
``step_id`` / ``graph_node`` via ``build_canonical_attributes`` →
``get_trace_context()``. Without this binding, tool spans inside a
node body would see ``step_id=None`` because LangGraph's
``metadata`` propagation does not carry our ``configurable.step_id``
(it only forwards ``langgraph_step``, an unrelated integer counter).
"""
from __future__ import annotations

import functools
import inspect
from dataclasses import replace
from typing import Any, Awaitable, Callable

from app.domain.external.observability import (
    TraceContext,
    TracerPort,
    build_canonical_attributes,
)
from app.infrastructure.observability.context import (
    get_trace_context,
    reset_trace_context,
    set_trace_context,
)


NodeCallable = Callable[..., Awaitable[Any]]


def traced_node(tracer: TracerPort) -> Callable[[NodeCallable], NodeCallable]:
    """Return a decorator that wraps a node coroutine in a tracer span.

    The returned decorator preserves the wrapped function's name +
    signature so LangGraph's ``add_node`` can introspect it identically.
    The span name is ``graph.node.<wrapped.__name__>``.
    """

    def decorator(fn: NodeCallable) -> NodeCallable:
        node_name = fn.__name__
        accepts_config = _accepts_config_arg(fn)

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            step_id = _resolve_step_id(args, kwargs, accepts_config)

            # Reviewer P2 fix: bind the ``TraceContext`` BEFORE building
            # node span attrs. ``build_canonical_attributes`` mints a
            # fresh fallback ``trace_id`` / ``request_id`` whenever the
            # contextvar is empty — calling it once here for the node
            # span and a second time inside ``_bind_node_context`` (for
            # the synthetic ctx) would mint TWO independent trace_ids,
            # so the node span and any tool spans inside its body would
            # diverge on the canonical join key. Bind first, then build
            # — every emit in the node body reads the same contextvar
            # source of truth.
            ctx_token = _bind_node_context(node_name, step_id)
            try:
                attrs = build_canonical_attributes(
                    graph_node=node_name,
                    step_id=step_id,
                )
                with tracer.start_as_current_span(
                    f"graph.node.{node_name}",
                    attributes=attrs,
                ):
                    return await fn(*args, **kwargs)
            finally:
                if ctx_token is not None:
                    reset_trace_context(ctx_token)

        return wrapper

    return decorator


def _accepts_config_arg(fn: NodeCallable) -> bool:
    """Return True when ``fn`` declares a ``config`` parameter."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return "config" in sig.parameters


def _resolve_step_id(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    accepts_config: bool,
) -> str | None:
    """Resolve step_id with the FOLLOW-10 + reviewer-P1 priority:

    1. ``state["current_step"].id`` (Pydantic ``Step`` instance) —
       primary source for ``executor_node`` because the planner has
       already pinned the step before the runtime dispatches.
    2. ``state["current_step"]["id"]`` — dict shape used by tests.
    3. ``config["configurable"]["step_id"]`` — explicit injection
       (e.g. for nodes that don't read ``current_step``).
    """
    step_id_from_state = _step_id_from_state(args, kwargs)
    if step_id_from_state:
        return step_id_from_state
    if accepts_config:
        return _step_id_from_config(args, kwargs)
    return None


def _step_id_from_state(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str | None:
    """Read ``state["current_step"].id`` (or ``["id"]`` for dicts)."""
    state: Any = kwargs.get("state")
    if state is None and args:
        state = args[0]
    if state is None:
        return None
    current_step = _try_get(state, "current_step")
    if current_step is None:
        return None
    candidate = _try_get(current_step, "id")
    if isinstance(candidate, str) and candidate:
        return candidate
    return None


def _step_id_from_config(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str | None:
    config = kwargs.get("config")
    if config is None and len(args) >= 2:
        candidate = args[1]
        if isinstance(candidate, dict):
            config = candidate
    if not isinstance(config, dict):
        return None
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return None
    step_id = configurable.get("step_id")
    return step_id if isinstance(step_id, str) and step_id else None


def _try_get(obj: Any, key: str) -> Any:
    """Read ``obj[key]`` (Mapping) or ``obj.key`` (attr) — best effort.

    ``MainGraphState`` is a TypedDict at runtime → mapping access.
    ``Step`` is a Pydantic model → attribute access. Tests sometimes
    use plain dicts; we accept both shapes.
    """
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _bind_node_context(graph_node: str, step_id: str | None) -> Any:
    """Bind ``graph_node`` + ``step_id`` onto the live ``TraceContext``.

    Returns the token to pass to ``reset_trace_context`` on exit. The
    binding is **always** performed — even when no current ``TraceContext``
    exists AND no ``step_id`` is bound — so a single contextvar acts as
    the source of truth for ``trace_id`` / ``request_id`` across all
    emits inside the node body (graph node span attrs + tool spans +
    domain logs). Without an unconditional bind, ``build_canonical_attributes``
    would mint independent fallback uuids each call, splitting the
    canonical join key (reviewer P2 round-2 finding).

    When a ``TraceContext`` IS already bound (HTTP request path,
    middleware-installed), we replace its ``graph_node`` / ``step_id``
    fields and restore on exit. The trace_id / request_id / session_id
    survive verbatim so canonical join keys stay stable across the
    span hierarchy.

    No-context path: build a single canonical snapshot and freeze its
    ``trace_id`` / ``request_id`` into a synthetic ``TraceContext``.
    All later ``build_canonical_attributes`` calls inside the node
    body read the same trace_id from that snapshot — node span and
    tool spans share the canonical join key.
    """
    current = get_trace_context()
    if current is not None:
        new_ctx = replace(current, graph_node=graph_node, step_id=step_id)
        return set_trace_context(new_ctx)
    # No context bound: synthesize one with a single fresh trace_id /
    # request_id pair and freeze it onto the contextvar so subsequent
    # emits inside this node body read consistent join keys.
    canon = build_canonical_attributes(
        graph_node=graph_node, step_id=step_id
    )
    synthetic = TraceContext(
        trace_id=canon["trace_id"],
        request_id=canon["request_id"],
        event_id=canon["event_id"],
        graph_node=graph_node,
        step_id=step_id,
    )
    return set_trace_context(synthetic)

