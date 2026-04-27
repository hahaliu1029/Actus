"""Shared telemetry helper for Actus LLM adapters (B5 C11).

The three adapter classes (``ActusChatModel``, ``ActusResponsesModel``,
``ActusFallbackChatModel``) all need the same pre-call telemetry logic:
hash the SystemMessage + bound tools and emit a
``PromptTelemetryPort.record_llm_invocation`` event. Rather than
duplicate the code three times, we expose two free functions and each
adapter calls them from its ``_agenerate`` entry.

Why not a base class mixin? LangChain's ``BaseChatModel`` uses Pydantic v2
which makes adding instance-level mutable state through inheritance
awkward. Free functions + a private attribute that each adapter declares
(``_telemetry: PromptTelemetryPort | None = None``) is simpler and keeps
the adapters free of MRO surprises.

Non-blocking guarantee: every helper swallows any exception and logs at
WARNING level so telemetry failures can never break the LLM call path.
"""
from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.messages import BaseMessage, SystemMessage

if TYPE_CHECKING:
    from app.domain.external.telemetry import PromptTelemetryPort


logger = logging.getLogger(__name__)


def attach_telemetry(
    adapter: Any,
    telemetry: "PromptTelemetryPort | None",
    lang: str = "zh",
) -> None:
    """Attach a telemetry port to an LLM adapter instance.

    Uses ``object.__setattr__`` to bypass Pydantic field validation so
    callers can attach/detach at runtime without touching the model
    schema.

    Passing ``None`` detaches — subsequent ``_agenerate`` calls will
    no-op at the telemetry hook. Used by tests to clear state between
    cases.

    ``lang`` is the ISO language code attributed to every subsequent
    invocation from this adapter. Defaults to ``"zh"`` to match the
    pre-audit hardcode. B5 post-audit LOW #4 moved this from a buried
    hardcode in ``emit_invocation_telemetry`` to a caller-facing
    parameter; ``AgentTaskRunner.set_language`` re-invokes this helper
    from ``main_graph.planner_node``'s ``language_callback`` on every
    turn, so the per-session precision is actually **per-turn** and
    already sufficient for the B5.5 bench use case.

    **TODOS #32 deferred (2026-04-13 brainstorm)**: moving to a
    per-message read via ``var_child_runnable_config.get()`` +
    ``configurable["session_language"]`` was considered and deferred
    as YAGNI. All three reference products (Claude Code / Hermes /
    OpenClaw) don't track language as a telemetry dimension at all,
    no observed bug exists, and the per-turn mechanism already covers
    every edge case the refactor would fix. See TODOS.md #32 for
    unblock conditions.
    """
    object.__setattr__(adapter, "_telemetry", telemetry)
    object.__setattr__(adapter, "_telemetry_lang", lang)


def _extract_system_prompt(messages: list[BaseMessage]) -> str:
    """Return the first SystemMessage content as text, or empty string.

    Multimodal content (list of blocks) is stringified via ``str(...)``
    as a last resort — SystemMessage is overwhelmingly text-only in
    Actus, and the fallback keeps the hash deterministic rather than
    raising.
    """
    for msg in messages:
        if isinstance(msg, SystemMessage):
            content = msg.content
            if isinstance(content, str):
                return content
            return str(content)
    return ""


def _extract_tool_names(tools: list[Any] | None) -> list[str]:
    """Return sorted unique tool names from an OpenAI-style tool list.

    Handles BOTH schema variants used by the Actus LLM adapters:

    - Chat Completions (``ActusChatModel``)::

          {"type": "function", "function": {"name": "shell_execute", ...}}

    - Responses API (``ActusResponsesModel``)::

          {"type": "function", "name": "shell_execute", "parameters": ...}

    Sorting ensures hash stability across registration order.

    **Codex audit HIGH #3 fix**: earlier versions of this helper only
    looked at ``tool["function"]["name"]``, which meant every
    ``ActusResponsesModel`` call produced an empty ``tools_hash``. Both
    schemas are now supported.
    """
    if not tools:
        return []
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        # Chat Completions: nested under "function"
        function = tool.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str) and name:
                names.add(name)
                continue
        # Responses API: flat top-level "name" (only consider entries
        # that explicitly declare type="function" to avoid picking up
        # unrelated dict keys named "name").
        if tool.get("type") == "function":
            name = tool.get("name")
            if isinstance(name, str) and name:
                names.add(name)
    return sorted(names)


def emit_invocation_telemetry(
    adapter: Any,
    messages: list[BaseMessage],
    tools: list[Any] | None,
) -> None:
    """Record a single LLM-invocation telemetry event.

    Called from each adapter's ``_agenerate`` AFTER the tool list and
    messages are assembled but BEFORE the provider API call. All
    exceptions are swallowed and logged at WARNING level — telemetry
    failures must never propagate to the main call path.

    Fields emitted:
    - ``system_prompt_bytes``: UTF-8 byte length of the first SystemMessage
    - ``system_prompt_hash``: sha256 hex digest of the SystemMessage text,
      truncated to 16 chars (collision-resistant enough for dedup analysis)
    - ``tools_hash``: sha256[:16] of ``"|".join(sorted_tool_names)``
    - ``lang``: hardcoded ``"zh"`` for C11 — B5.5 bench will wire a real
      language channel through the LangGraph config if the metric matters
    - ``provider``: the adapter's ``provider_name`` attribute
    """
    telemetry = getattr(adapter, "_telemetry", None)
    if telemetry is None:
        return
    try:
        system_text = _extract_system_prompt(messages)
        system_bytes = len(system_text.encode("utf-8"))
        system_hash = hashlib.sha256(system_text.encode("utf-8")).hexdigest()[:16]

        tool_names = _extract_tool_names(tools)
        tools_joined = "|".join(tool_names)
        tools_hash = hashlib.sha256(tools_joined.encode("utf-8")).hexdigest()[:16]

        provider = getattr(adapter, "provider_name", "openai")
        # B5 post-audit LOW #4: lang comes from attach_telemetry's
        # ``lang`` argument (stored on the adapter as ``_telemetry_lang``).
        # Default "zh" matches the pre-audit hardcode — the real plumbing
        # from main_graph state through AgentTaskRunner is TODOS #32.
        lang = getattr(adapter, "_telemetry_lang", "zh")
        telemetry.record_llm_invocation(
            system_prompt_hash=system_hash,
            system_prompt_bytes=system_bytes,
            tools_hash=tools_hash,
            lang=lang,
            provider=provider,
        )
    except Exception as exc:
        logger.warning(
            "[LLM Telemetry] record_llm_invocation failed (swallowed): %s", exc
        )


def emit_recovery_event(adapter, event) -> None:
    """Dispatch a RecoveryEvent to the adapter's attached telemetry.

    Matches `emit_invocation_telemetry`'s non-blocking contract (line 141):
    any telemetry failure is swallowed and logged at WARNING level so the
    LLM call path is never broken by a telemetry bug.

    Round 22 P1 #1 update: `PromptTelemetryPort.emit_recovery_event` is now
    a declared Protocol method (Step 1), so production telemetries
    implement it. The `getattr(... None)` fallback below is only for
    legacy / test stubs that haven't been migrated to the new Protocol.
    """
    telemetry = getattr(adapter, "_telemetry", None)
    if telemetry is None:
        return
    try:
        hook = getattr(telemetry, "emit_recovery_event", None)
        if hook is None:
            return
        hook(event)
    except Exception as exc:  # pragma: no cover — defensive only
        logger.warning(
            "[LLM Telemetry] emit_recovery_event failed (swallowed): %s", exc,
        )
