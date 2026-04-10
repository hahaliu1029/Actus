"""Domain protocol for prompt assembly and LLM invocation telemetry.

B5 C1: defines ``PromptTelemetryPort`` Protocol consumed by ``PromptAssembler``
(domain layer) and implemented by ``JsonlPromptTelemetry`` (infrastructure).

The port keeps file I/O out of the domain layer (Clean Architecture).

Known design debt (B5 eng review): the port currently has 4 methods. If a
second consumer is added (Datadog adapter etc.), the port should be split
into ``AssemblyTelemetryPort`` / ``InvocationTelemetryPort`` /
``DegradationTelemetryPort`` per ISP. See TODOS.md #28.
"""
from __future__ import annotations

from typing import Protocol


class PromptTelemetryPort(Protocol):
    """Port for prompt assembly and LLM invocation telemetry.

    All methods are sync (no I/O await) and side-effect-only — they return
    None and never raise. Implementations must swallow internal errors and
    log them locally; telemetry failures must not propagate to callers.
    """

    def record_assembly(
        self,
        *,
        sections_included: list[str],
        sections_dropped: list[str],
        tokens_used: int,
        lang: str,
        provider: str,
        mode: str,
        version_hash: str,
        fallback_used: bool = False,
    ) -> None:
        """Called by ``PromptAssembler.assemble`` after producing the result.

        ``fallback_used`` is True when the executor used the legacy
        state.skill_context fallback path (``react_graph_provider`` not
        configured). Used by B5.7 follow-up to decide when to retire the
        two-clock architecture.
        """
        ...

    def record_llm_invocation(
        self,
        *,
        system_prompt_hash: str,
        system_prompt_bytes: int,
        tools_hash: str,
        lang: str,
        provider: str,
    ) -> None:
        """Called by the LLM adapter (C11) before sending each request.

        Used by B5.5 follow-up to compute hit-rate ceilings before deciding
        whether to enable Anthropic prompt caching.
        """
        ...

    def record_lc_tools_degradation(
        self,
        *,
        reason: str,
    ) -> None:
        """Called by ``_build_step_react_graph`` when it falls back to
        ``_build_minimal_lc_tools_for_step`` due to skill tool init failure.
        """
        ...
