"""Contextvars-backed ``TraceContext`` carrier.

PR-S1-1 shipped the read path (``get_trace_context``) so the
``domain.external.observability.build_canonical_attributes`` helper
could resolve its lazy import. PR-S1-2 extends this module with the
write path:

- ``set_trace_context(ctx) -> Token`` and ``reset_trace_context(token)``
  for callers that need explicit lifetime control (e.g., the pure-ASGI
  ``ObservabilityMiddleware`` in PR-S1-5).
- ``bind_session_context(session_id)`` and ``bind_request_context(
  request_id)`` async context managers (per Q8) for scope-bound mutation
  of a single field. Idiomatic usage::

      async with bind_session_context(session.id):
          # downstream code reads ``get_trace_context().session_id``
          # automatically; reset on exit (and on exception) is guaranteed.
          ...

The ``ContextVar`` lives at module scope (not inside a class) because
asyncio task scheduling reads contextvars from the binding scope at task
creation time — class-attribute storage would introduce a per-instance
binding that breaks propagation across spawned tasks.

Thread / task safety
--------------------
``ContextVar`` is async-safe by design: each ``asyncio.Task`` snapshots
the contextvar map at creation, so child tasks observe the parent's
binding but cannot mutate it. ``bind_*_context`` mutations only affect
the awaiting task's view; concurrent tasks see their own bindings.
"""
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from dataclasses import replace

from app.domain.external.observability import TraceContext

_trace_context_var: ContextVar[TraceContext | None] = ContextVar(
    "actus_trace_context",
    default=None,
)


def get_trace_context() -> TraceContext | None:
    """Return the current request-scoped ``TraceContext`` or ``None``.

    Returns ``None`` when called outside a request scope (CLI scripts,
    startup paths, unit tests without context binding). Callers that
    require a context (every span emit, every JSONL write) treat this
    as the contract: ``build_canonical_attributes`` generates a fallback
    trace_id / request_id when the getter returns ``None`` so the
    canonical contract's NEVER-null guarantee still holds.
    """
    return _trace_context_var.get()


def set_trace_context(
    ctx: TraceContext | None,
) -> Token[TraceContext | None]:
    """Set the current ``TraceContext`` and return a reset token.

    The caller is responsible for invoking ``reset_trace_context(token)``
    when the bound scope ends (e.g., in a ``finally`` block) so the
    contextvar reverts to its previous value. Failing to reset leaks the
    binding into subsequent code that runs in the same task.

    Prefer ``bind_session_context`` / ``bind_request_context`` for
    scope-bound mutations; ``set_trace_context`` is the lower-level
    primitive used by middleware to install the request-root context.
    """
    return _trace_context_var.set(ctx)


def reset_trace_context(token: Token[TraceContext | None]) -> None:
    """Restore the ``TraceContext`` to the value before ``token`` was issued."""
    _trace_context_var.reset(token)


def _fresh_context(**fields: object) -> TraceContext:
    """Create a fresh ``TraceContext`` with auto-generated required IDs.

    Used when ``bind_*_context`` is called outside a request scope (CLI,
    background tasks, unit tests). The required attrs follow the v1
    canonical format: ``trace_id`` is 32-char ``uuid4().hex`` and
    ``request_id`` / ``event_id`` are UUIDv4 strings — both pass the
    tightened ``validate_attributes`` format gate.
    """
    base: dict[str, object] = {
        "trace_id": uuid.uuid4().hex,
        "request_id": str(uuid.uuid4()),
        "event_id": str(uuid.uuid4()),
    }
    base.update(fields)
    return TraceContext(**base)  # type: ignore[arg-type]


@asynccontextmanager
async def bind_session_context(
    session_id: str,
) -> AsyncIterator[TraceContext]:
    """Bind ``session_id`` for the duration of the ``async with`` block.

    If a ``TraceContext`` is already bound (the common HTTP request
    case), the existing trace_id / request_id / etc. are preserved and
    only the ``session_id`` field is replaced. If no context is bound
    (CLI, background tasks, scheduler-driven flows), a fresh context is
    created with auto-generated trace_id / request_id / event_id and the
    supplied ``session_id``.

    The previous context (or ``None``) is restored on exit, including
    when the body raises — ``contextlib.asynccontextmanager`` propagates
    the exception after running the cleanup.
    """
    current = _trace_context_var.get()
    if current is None:
        new_ctx = _fresh_context(session_id=session_id)
    else:
        new_ctx = replace(current, session_id=session_id)

    token = _trace_context_var.set(new_ctx)
    try:
        yield new_ctx
    finally:
        _trace_context_var.reset(token)


@asynccontextmanager
async def bind_request_context(
    request_id: str,
) -> AsyncIterator[TraceContext]:
    """Bind ``request_id`` for the duration of the ``async with`` block.

    Mirrors ``bind_session_context``: existing context has its
    ``request_id`` replaced; missing context spawns a fresh one with
    auto-generated trace_id / event_id and the supplied ``request_id``.
    Restores the previous binding on exit and on exception.

    ``ObservabilityMiddleware`` (PR-S1-5) uses ``set_trace_context``
    directly to install the request-root context; ``bind_request_context``
    is for sub-flows that need a different request_id (fan-out, retry
    sub-spans) without losing the surrounding trace_id.
    """
    current = _trace_context_var.get()
    if current is None:
        new_ctx = _fresh_context(request_id=request_id)
    else:
        new_ctx = replace(current, request_id=request_id)

    token = _trace_context_var.set(new_ctx)
    try:
        yield new_ctx
    finally:
        _trace_context_var.reset(token)
