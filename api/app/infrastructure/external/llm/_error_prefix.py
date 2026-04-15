"""R2 CS2.14: Error prefix injection helper shared by all LLM adapters.

Called from ``_messages_for_api()`` of each ``ActusXxxModel`` when
``msg.status == "error"`` to prepend a machine-readable prefix into
``msg.content`` before sending to the LLM wire format.

Rationale: ``ActusChatModel`` / ``ActusResponsesModel`` drop the
``ToolMessage.status`` field when serializing to OpenAI wire format (the
OpenAI tool-message schema has no ``status`` slot). The LLM therefore
cannot see the R2 CS2 success/error signal unless we encode it into
``content``. We prepend ``[TOOL_FAILED: {reason.type}]`` for
``AllowError`` and ``[TOOL_DENIED: {reason.type}]`` for ``Denied``. This
replaces the R1 inline ``[TOOL_ERROR]`` prefix that was removed from
``react_graph.tool_node`` in R2 Commit 1 (see
``docs/superpowers/specs/2026-04-15-r2-toolstatus-taxonomy-design.md``
CS2.14 for the full rationale and the "Layer 3 ↔ LLM adapter"
boundary).

The helper accepts three shapes on purpose so the adapter can call it
without knowing whether the artifact round-tripped through the
LangGraph checkpointer:

1. ``ToolArtifact`` Pydantic object — produced by Layer 3 in-process.
2. A raw variant (``AllowError`` / ``Denied``) — produced if a caller
   only has the outcome.
3. A ``dict`` — produced after checkpointer rehydration
   (``ToolArtifact.model_dump(mode="json", by_alias=True)``).

Anything else (``None`` / a bare string / unknown shape) returns
``None`` so the adapter silently skips the prefix rather than crashing
the LLM call.
"""
from __future__ import annotations

from typing import Any

from app.domain.models.tool_result import (
    AllowError,
    Denied,
    ToolArtifact,
)


def _format_error_prefix(artifact: Any) -> str | None:
    """Return the error prefix to prepend to ``ToolMessage.content``, or ``None``.

    Returns ``"[TOOL_FAILED: {reason.type}]"`` for ``AllowError`` outcomes,
    ``"[TOOL_DENIED: {reason.type}]"`` for ``Denied`` outcomes, and ``None``
    for every other variant / shape (including success variants and
    malformed payloads).
    """
    # Case 1: typed ToolArtifact (Pydantic object, pre-checkpointer)
    if isinstance(artifact, ToolArtifact):
        outcome = artifact.outcome
        if isinstance(outcome, AllowError):
            return f"[TOOL_FAILED: {outcome.reason.type}]"
        if isinstance(outcome, Denied):
            return f"[TOOL_DENIED: {outcome.reason.type}]"
        return None

    # Case 2: typed outcome directly (some callers only have the outcome)
    if isinstance(artifact, AllowError):
        return f"[TOOL_FAILED: {artifact.reason.type}]"
    if isinstance(artifact, Denied):
        return f"[TOOL_DENIED: {artifact.reason.type}]"

    # Case 3: dict form (from checkpointer round-trip or golden matrix JSON).
    # This path must be **fail-soft** — the whole point of the dict branch is
    # to absorb pollution from external producers (older checkpoints,
    # hand-crafted fixtures, buggy Layer 2 wrappers). Any AttributeError
    # leaking out of here would crash the LLM call on a malformed artifact,
    # defeating the ``_format_error_prefix(...) or "[TOOL_ERROR]"`` fallback
    # that the adapters rely on. So every nested ``.get()`` must be guarded
    # by ``isinstance(..., dict)`` — we cannot assume the checkpointer
    # preserves the typed shape.
    if isinstance(artifact, dict):
        outcome_dict = (
            artifact.get("outcome") if "outcome" in artifact else artifact
        )
        if isinstance(outcome_dict, dict):
            variant = outcome_dict.get("variant")
            raw_reason = outcome_dict.get("reason")
            reason_type = (
                raw_reason.get("type", "unknown")
                if isinstance(raw_reason, dict)
                else "unknown"
            )
            if variant == "allow_error":
                return f"[TOOL_FAILED: {reason_type}]"
            if variant == "denied":
                return f"[TOOL_DENIED: {reason_type}]"

    # Case 4: unknown / None / string / anything else → no prefix
    return None
