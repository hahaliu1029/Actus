"""Recovery chain core data types (B2).

This module must NOT import from app.domain.services.graphs or
app.application.* — locked by test_no_forbidden_imports.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Literal, Protocol

from langchain_core.messages import BaseMessage

from app.domain.services.provider_profiles._base import ErrorClass, ProviderProfile


ApiMode = Literal["chat_completions", "responses"]


OnContextOverflow = Callable[
    [list[BaseMessage], dict],
    Awaitable[list[BaseMessage] | None],
]
"""Runner-provided compact callback.

Arguments:
    messages: current message list (will not be mutated by callback)
    kwargs: shallow-copied LLM kwargs (MUST NOT be mutated by callback;
        kwargs changes go through RewriteAction only)

Returns:
    list of compressed messages, OR None meaning "no progress / give up" —
    Recovery re-raises the original LLM error.
"""


@dataclass(frozen=True)
class RecoveryContext:
    """Per-call immutable context passed to every RewriteAction.apply()."""

    profile: ProviderProfile
    api_mode: ApiMode
    attempt_index: int
    attempted_action_codes: frozenset[str]
    on_context_overflow: OnContextOverflow | None


class RewriteAction(Protocol):
    """Typed rewrite strategy. Implementations MUST be idempotent-safe."""

    @property
    def code(self) -> str:
        """Stable identifier for telemetry + dedup. Snake_case."""
        ...

    async def apply(
        self,
        messages: list[BaseMessage],
        kwargs: dict,
        ctx: RecoveryContext,
    ) -> tuple[list[BaseMessage], dict] | None:
        """Return new (messages, kwargs) OR None to skip this candidate.

        Returning None DOES NOT consume the rewrite budget.
        """
        ...


# Rule key grammar: (profile_id, api_mode, error_class, fingerprint_code).
# fingerprint_code=None input to match_rule is treated as equivalent to "*".
RuleKey = tuple[str, str, ErrorClass, str]
