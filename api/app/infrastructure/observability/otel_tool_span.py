"""B5 PR-S2-2: tool span emission via LangChain callback handler.

Each tool invocation inside ``react_graph`` (and any other LangChain
runnable that fires the callback chain) produces an OTel ``tool.<name>``
span. Span attributes:

- ``tool_name`` — resolved from ``serialized.name`` or the run id chain.
- ``tool_args_hash`` — first 16 hex chars of SHA-256 over a stable JSON
  serialization of the tool's input dict. The hash + size combination
  is the **canonical privacy-preserving fingerprint**: it lets ops
  correlate "which tool call ran" / "is this a duplicate" without ever
  surfacing the raw arguments (which routinely carry user PII, API
  tokens, file content, ...).
- ``tool_args_size`` — byte length of the serialized JSON.
- Canonical attrs (trace_id / request_id / session_id / step_id /
  graph_node / event_id / ...) via ``build_canonical_attributes``.

Hard invariant
--------------
**The handler must NEVER write the raw tool inputs onto a span.**
``test_tool_args_hash_size_only.py`` (PR-S2-4 acceptance) does an AST /
attribute scan to enforce this — adding a ``"tool_args"`` or
``"tool_input"`` attribute would break the test and the spec contract.

Run-id keyed span store
-----------------------
LangChain fires ``on_tool_start`` / ``on_tool_end`` with the same
``run_id``. We keep a per-handler ``_active_spans`` dict so concurrent
tool calls (parallel tool calls in a single LLM turn) get distinct
spans. The store is cleared on end / error to avoid leaks.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler

from app.domain.external.observability import (
    TracerPort,
    build_canonical_attributes,
)


logger = logging.getLogger(__name__)


_FORBIDDEN_RAW_ARG_KEYS: frozenset[str] = frozenset(
    ("tool_args", "tool_input", "tool_inputs", "args", "raw_args")
)


def _hash_args(inputs: Any) -> tuple[str | None, int | None]:
    """Return ``(sha256[:16], byte_size)`` for the tool input dict.

    Uses ``json.dumps(..., sort_keys=True, default=str)`` so semantically
    equal dicts hash identically across runs (ordering-independent) and
    so non-JSON-native values (datetime, Path, ...) don't crash the hash
    path. Returns ``(None, None)`` when the input is missing — the
    tracer drops ``None`` attributes so spans stay valid.
    """
    if inputs is None:
        return None, None
    try:
        serialized = json.dumps(inputs, sort_keys=True, default=str)
    except (TypeError, ValueError):
        # Defensive fallback — we still emit a fingerprint, just based
        # on ``repr``. Better than dropping the span entirely.
        serialized = repr(inputs)
    encoded = serialized.encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return digest, len(encoded)


def _coerce_str(value: Any) -> str | None:
    """Return a non-empty ``str`` or ``None`` (for OTel-side filtering).

    Tool call ids arrive as ``str`` in modern LangChain runtimes but
    older LangChain releases occasionally pass ``UUID`` objects. We
    accept any value with a meaningful ``str`` cast and normalise to
    string so the canonical attr stays text-typed.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value if value else None
    coerced = str(value)
    return coerced if coerced else None


def _resolve_tool_name(
    serialized: dict[str, Any] | None,
    kwargs: dict[str, Any],
) -> str:
    """Pull a stable tool name out of LangChain's callback payload.

    Search order:

    1. ``serialized["name"]`` — the standard LangChain field.
    2. ``serialized["id"][-1]`` — the qualified path tail (some
       runnables only populate the chain id).
    3. ``kwargs["name"]`` — newer LangChain runtimes pass the name
       directly.
    4. ``"unknown"`` — defensive default; the span is still useful.
    """
    if serialized:
        name = serialized.get("name")
        if isinstance(name, str) and name:
            return name
        chain_id = serialized.get("id")
        if isinstance(chain_id, (list, tuple)) and chain_id:
            tail = chain_id[-1]
            if isinstance(tail, str) and tail:
                return tail
    name = kwargs.get("name")
    if isinstance(name, str) and name:
        return name
    return "unknown"


class OtelToolSpanCallback(AsyncCallbackHandler):
    """LangChain callback handler that opens a span per tool call.

    Wired into ``cfg["callbacks"]`` alongside the existing
    ``CostCallbackHandler``. LangGraph propagates the callback list to
    sub-runnables so every tool call in ``react_graph.tool_node`` fires
    the start / end pair.
    """

    def __init__(self, tracer: TracerPort) -> None:
        self._tracer = tracer
        # ``run_id -> span``. Concurrent tool calls in a single LLM
        # turn get distinct entries; the dict is cleared on end /
        # error to avoid leaks.
        self._active_spans: dict[UUID, Any] = {}

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        tool_name = _resolve_tool_name(serialized, kwargs)
        args_hash, args_size = _hash_args(inputs)
        # Reviewer P2 fix: read ``tool_call_id`` from kwargs.
        # ``langchain_core.tools.base.BaseTool`` passes it through to
        # ``callback_manager.on_tool_start(... tool_call_id=...)``; we
        # surface it on the span so downstream joins (ToolEvent /
        # tool_node accounting / approval grant records) can correlate exactly.
        # ``step_id`` is NOT read from metadata: LangGraph's metadata
        # carries an unrelated ``langgraph_step`` integer counter, not
        # our ``configurable.step_id``. Instead ``traced_node`` binds
        # the live ``TraceContext`` so ``build_canonical_attributes``
        # picks step_id up via the contextvar — same path used by the
        # domain logger and the cost callback.
        tool_call_id = _coerce_str(kwargs.get("tool_call_id"))

        canonical = build_canonical_attributes(
            tool_name=tool_name,
            tool_args_hash=args_hash,
            tool_args_size=args_size,
            tool_call_id=tool_call_id,
        )
        # Defence in depth: scrub forbidden keys at the boundary too.
        # ``build_canonical_attributes`` already drops unknown keys, but
        # this filter holds the line if a future contract change ever
        # adds ``tool_args`` / ``tool_input`` to the canonical set.
        attrs = {
            k: v
            for k, v in canonical.items()
            if k not in _FORBIDDEN_RAW_ARG_KEYS
        }

        # Use ``start_span`` (not ``start_as_current_span``) — start /
        # end are split across two callback methods, and we don't want
        # to install the tool span as the active context for everything
        # the tool calls into.
        span = self._tracer.start_span(
            f"tool.{tool_name}",
            attributes=attrs,
        )
        self._active_spans[run_id] = span

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        span = self._active_spans.pop(run_id, None)
        if span is None:
            return
        try:
            span.end()
        except Exception:  # pragma: no cover  defensive
            logger.debug(
                "OtelToolSpanCallback: span.end() failed", exc_info=True
            )

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        span = self._active_spans.pop(run_id, None)
        if span is None:
            return
        try:
            # Record the exception TYPE only — message could carry
            # data the tool was about to redact. The OTel API requires
            # an exception instance, so we synthesize one without a
            # message body.
            span.record_exception(type(error)(""))
        except Exception:  # pragma: no cover  defensive
            logger.debug(
                "OtelToolSpanCallback: record_exception failed",
                exc_info=True,
            )
        # B5 PR-S2-2 round-6 P2 fix: ``record_exception`` only attaches
        # an event, it does NOT flip the OTel span status. We need to
        # explicitly set ERROR so dashboards / alerts that key by
        # ``span.status_code == ERROR`` count tool failures (not just
        # 5xx responses). Manual ``start_span`` spans don't get the
        # auto-status-on-exit behaviour that ``start_as_current_span``
        # provides on exception, so this is required.
        try:
            from opentelemetry.trace import Status, StatusCode

            span.set_status(Status(StatusCode.ERROR))
        except Exception:  # pragma: no cover  defensive
            logger.debug(
                "OtelToolSpanCallback: set_status failed",
                exc_info=True,
            )
        try:
            span.end()
        except Exception:  # pragma: no cover  defensive
            logger.debug(
                "OtelToolSpanCallback: span.end() failed", exc_info=True
            )
