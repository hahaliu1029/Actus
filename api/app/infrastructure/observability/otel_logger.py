"""B5 PR-S2-1: ``LoggerPort`` implementation backed by stdlib + OTel bridge.

``OtelLogger`` satisfies the ``LoggerPort`` Protocol declared in
``app/domain/external/observability.py``. It wraps a stdlib
``logging.Logger`` so the existing Sprint 1 pipeline (RedactingFormatter
on root, ConcurrentRotatingFileHandler, propagate-only third-party
isolation) keeps working unchanged. The OTel bridge — installed by
``setup_observability()`` — attaches a ``LoggingHandler`` to the root
logger, so every record this class emits ALSO flows to the OTel
``LoggerProvider`` (and from there to whatever exporter the deployment
configures).

Each call appends ``build_canonical_attributes(...)`` output to the
``LogRecord`` ``extra`` dict so canonical join-keys (``trace_id`` /
``request_id`` / ``session_id`` / ...) ride alongside the message.
Stdlib's ``RedactingFormatter`` reads ``record.__dict__`` for
``%(...)s`` substitution; OTel's ``LoggingHandler._translate`` copies
``record.__dict__`` into ``LogRecord.attributes`` so the same canonical
keys land on the OTel side too.

Domain code that depends on ``LoggerPort`` should not import this
module directly — it depends on the Protocol and gets ``OtelLogger``
injected at composition time. Tests instantiate ``OtelLogger`` directly
to assert the LoggerPort surface is satisfied.
"""
from __future__ import annotations

import logging
from typing import Any

from app.domain.external.observability import (
    CANONICAL_ATTRIBUTES,
    TraceContext,
    build_canonical_attributes,
)
from app.infrastructure.observability.context import (
    get_trace_context,
    reset_trace_context,
    set_trace_context,
)

# PR-S1-4's ``_actus_log_record_factory`` already pins ``trace_id`` /
# ``request_id`` / ``session_id`` directly onto every ``LogRecord``.
# Passing them again via ``extra`` would trip stdlib's collision guard
# (``KeyError: "Attempt to overwrite 'trace_id' in LogRecord"``). The
# remaining canonical attrs (graph_node, step_id, tool_name, ...) are
# emit-site values not handled by the factory; those go through ``extra``.
_FACTORY_OWNED_ATTRS: frozenset[str] = frozenset(
    ("trace_id", "request_id", "session_id")
)
_EXTRA_OWNED_ATTRS: frozenset[str] = (
    frozenset(CANONICAL_ATTRIBUTES) - _FACTORY_OWNED_ATTRS
)

# Keys a caller MUST NOT override via ``extra``:
#
# - ``trace_id`` / ``request_id`` / ``session_id`` are factory-owned
#   (see comment above).
# - ``event_id`` is REQUIRED + fresh-per-emit per the v1 canonical
#   contract — letting a caller pin one would break uniqueness across
#   emits and could push a non-UUIDv4 string past validate_attributes
#   downstream.
# - ``user_id_hash`` is derived from ``TraceContext`` + the deployment-
#   secret salt; an emit site cannot hash a different user's id past
#   the policy boundary.
#
# Anything left (``graph_node`` / ``step_id`` / ``tool_*`` /
# ``llm_provider`` / ``model`` / ``attempt_ix`` / ``decision_reason``)
# is emit-site information the caller is the canonical source for —
# those flow through ``extra`` unchanged.
_CALLER_FORBIDDEN_KEYS: frozenset[str] = _FACTORY_OWNED_ATTRS | frozenset(
    ("event_id", "user_id_hash")
)
_CALLER_ALLOWED_KEYS: frozenset[str] = (
    frozenset(CANONICAL_ATTRIBUTES) - _CALLER_FORBIDDEN_KEYS
)


class OtelLogger:
    """``LoggerPort`` impl wrapping a stdlib logger + OTel bridge."""

    __slots__ = ("_logger",)

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    @property
    def name(self) -> str:
        return self._logger.name

    def _emit(
        self,
        level: int,
        message: str,
        *,
        extra: dict[str, Any] | None,
        exc_info: bool = False,
    ) -> None:
        # Build the canonical attribute snapshot at the call site so a
        # fresh ``event_id`` is generated per emit. Filter out the keys
        # that PR-S1-4's record factory already pins onto the record
        # (``trace_id`` / ``request_id`` / ``session_id``); anything
        # left (graph_node / step_id / tool_* / ...) is safe to pass
        # through ``extra``.
        #
        # Drop ``None`` values: most canonical attrs are nullable in
        # the v1 contract, but OTLP collectors reject ``None`` attribute
        # values (the protobuf schema disallows null). The OTel Python
        # SDK silently drops null-valued attributes, but downstream
        # JSONL / Prometheus exporters could surface ``"graph_node":
        # null`` lines — emit only present values to keep the wire
        # format clean across exporters.
        #
        # Caller-supplied ``extra`` is filtered through the same
        # ``_EXTRA_OWNED_ATTRS`` whitelist as the canonical fallback:
        # (1) factory-owned keys (``trace_id`` / ``request_id`` /
        # ``session_id``) are dropped silently — passing them would
        # trip stdlib's ``"Attempt to overwrite ... in LogRecord"``
        # KeyError; (2) keys outside the v1 canonical contract are
        # dropped silently — matches ``validate_attributes`` drop-
        # unknown semantics so non-canonical fields cannot leak into
        # OTel attributes / downstream JSONL.
        canon = build_canonical_attributes()
        merged: dict[str, Any] = {
            k: v
            for k, v in canon.items()
            if k in _EXTRA_OWNED_ATTRS and v is not None
        }
        if extra:
            # Caller-supplied extras use the stricter
            # ``_CALLER_ALLOWED_KEYS`` whitelist (drops factory-owned
            # plus ``event_id`` / ``user_id_hash``) so callers cannot
            # override required join keys or the per-emit fresh
            # ``event_id`` from ``build_canonical_attributes``.
            merged.update(
                {
                    k: v
                    for k, v in extra.items()
                    if k in _CALLER_ALLOWED_KEYS and v is not None
                }
            )
        # No-context path: PR-S1-4's ``_actus_log_record_factory``
        # pins ``trace_id`` / ``request_id`` / ``session_id`` to the
        # placeholder ``"-"`` when no ``TraceContext`` is bound (so
        # ``%(trace_id)s`` format strings don't blow up). That
        # placeholder fails v1 ``validate_attributes`` and breaks the
        # OTel canonical join-key contract.
        #
        # ``build_canonical_attributes`` already generates canonical
        # uuids in ``canon`` for this exact case (CLI / startup /
        # background / direct unit test); bind them via a synthetic
        # ``TraceContext`` so the factory writes the valid uuids onto
        # the record. Reset on exit so we don't leak the synthetic
        # context to surrounding code.
        ctx_token = None
        if get_trace_context() is None:
            synthetic = TraceContext(
                trace_id=canon["trace_id"],
                request_id=canon["request_id"],
                event_id=canon["event_id"],
            )
            ctx_token = set_trace_context(synthetic)
        try:
            # ``stacklevel=2`` so the caller's site (not this method)
            # is recorded as the source location.
            self._logger.log(
                level,
                message,
                extra=merged,
                exc_info=exc_info,
                stacklevel=2,
            )
        finally:
            if ctx_token is not None:
                reset_trace_context(ctx_token)

    def info(self, message: str, *, extra: dict[str, Any] | None = None) -> None:
        self._emit(logging.INFO, message, extra=extra)

    def warning(
        self, message: str, *, extra: dict[str, Any] | None = None
    ) -> None:
        self._emit(logging.WARNING, message, extra=extra)

    def error(
        self, message: str, *, extra: dict[str, Any] | None = None
    ) -> None:
        self._emit(logging.ERROR, message, extra=extra)

    def exception(
        self, message: str, *, extra: dict[str, Any] | None = None
    ) -> None:
        self._emit(logging.ERROR, message, extra=extra, exc_info=True)
