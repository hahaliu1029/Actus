"""Context assembler: manages token budgets and trims message history.

Provides deterministic, synchronous trimming before LLM calls via three phases:
  Phase 1 — compress ToolMessage content (head+tail truncation)
  Phase 2 — drop oldest non-protected tool_call groups
  Phase 3 — drop oldest non-protected standalone turns (human / ai_text)

See spec: docs/superpowers/specs/2026-03-31-context-assembler-design.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from .message_utils import truncate_tool_content
from .token_estimator import TokenEstimator

# Decision-recorder port — same Callable type alias as in smart_approve.py.
# Composition layer injects the OTel-backed ``record_decision``; tests/callers
# that don't care about observability pass ``None`` (no-op). Domain code MUST
# NOT import ``app.infrastructure.observability.decision_trace`` directly.
_DecisionRecorder = Callable[..., None] | None

logger = logging.getLogger(__name__)


def _safe_record_decision(
    recorder: _DecisionRecorder,
    name: str,
    *,
    outcome: str,
) -> None:
    """Call *recorder* swallowing any exception (port contract: observability
    failure MUST NOT taint trimming). ``None`` recorder is a no-op."""
    if recorder is None:
        return
    try:
        recorder(name, outcome=outcome)
    except Exception:  # noqa: BLE001
        pass  # OTel hiccup must never abort context assembly


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MessageGroup:
    """An atomic group of one or more messages sharing a logical role.

    kind values:
      "system"    — SystemMessage
      "human"     — HumanMessage
      "ai_text"   — AIMessage without tool_calls
      "tool_call" — AIMessage-with-tool_calls + matching ToolMessages
    """

    kind: str
    messages: list[BaseMessage]
    protected: bool = False


@dataclass(frozen=True)
class AssemblyResult:
    """Outcome of a single assemble() call.

    Attributes:
        messages:        Trimmed (or original) message list.
        original_tokens: Estimated tokens before trimming.
        final_tokens:    Estimated tokens after trimming.
        actions:         Human-readable log of phases that were applied.
    """

    messages: list[BaseMessage]
    original_tokens: int
    final_tokens: int
    actions: list[str] = field(default_factory=list)


# ── Grouping ───────────────────────────────────────────────────────────────────

def group_messages(messages: list[BaseMessage]) -> list[MessageGroup]:
    """Group a flat message list into MessageGroup objects.

    Tool-call pairing: an AIMessage that has tool_calls is grouped together
    with all immediately following ToolMessages whose tool_call_id matches
    one of the AIMessage's tool call IDs.

    Orphaned ToolMessages (no preceding AI with matching IDs) are wrapped
    defensively in their own tool_call group.
    """
    groups: list[MessageGroup] = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        if isinstance(msg, SystemMessage):
            groups.append(MessageGroup(kind="system", messages=[msg]))
            i += 1

        elif isinstance(msg, HumanMessage):
            groups.append(MessageGroup(kind="human", messages=[msg]))
            i += 1

        elif isinstance(msg, AIMessage) and msg.tool_calls:
            # Collect expected tool_call_ids from this AI message
            expected_ids = {tc["id"] for tc in msg.tool_calls}
            group_msgs: list[BaseMessage] = [msg]
            j = i + 1
            while j < len(messages):
                candidate = messages[j]
                if (
                    isinstance(candidate, ToolMessage)
                    and candidate.tool_call_id in expected_ids
                ):
                    group_msgs.append(candidate)
                    expected_ids.discard(candidate.tool_call_id)
                    j += 1
                else:
                    break
            groups.append(MessageGroup(kind="tool_call", messages=group_msgs))
            i = j

        elif isinstance(msg, AIMessage):
            groups.append(MessageGroup(kind="ai_text", messages=[msg]))
            i += 1

        elif isinstance(msg, ToolMessage):
            # Orphaned ToolMessage — defensive grouping
            groups.append(MessageGroup(kind="tool_call", messages=[msg]))
            i += 1

        else:
            # Unknown message type — treat as standalone human-like group
            groups.append(MessageGroup(kind="human", messages=[msg]))
            i += 1

    return groups


# ── Assembler ──────────────────────────────────────────────────────────────────

class ContextAssembler:
    """Trims a message list to fit within a token budget before LLM calls.

    Deterministic and synchronous (CPU-bound only).  Create one instance per
    flow; reuse across assemble() calls (no mutable state modified after init).
    """

    def __init__(
        self,
        estimator: TokenEstimator,
        effective_window: int | None = None,
        context_window: int | None = None,
        reserved_output_tokens: int = 4096,
        safety_factor: float = 1.15,
        tool_compress_trigger_ratio: float = 0.75,
        tool_compress_target_chars: int = 500,
        decision_recorder: _DecisionRecorder = None,
    ) -> None:
        """Construct the context assembler.

        B5 C9: accepts either ``effective_window`` (preferred, new API) or
        ``context_window`` (deprecated shim). The ``effective_window`` is the
        amount of tokens history is allowed to use — already with system
        prompt budget and reserved output subtracted via
        ``compute_effective_window()``. The deprecated ``context_window``
        path reproduces the pre-C9 formula
        ``(context_window - reserved_output_tokens) / safety_factor``.

        Must provide exactly one of the two window parameters. Passing both
        favors ``effective_window``. Passing neither raises ``ValueError``.

        The ``context_window`` shim will be removed alongside other legacy
        cleanup in B5.5 / B5.6. Migrate callers to ``effective_window`` +
        ``compute_effective_window()``.

        ``decision_recorder`` — optional OTel-backed callable injected by the
        composition layer (``build_decision_recorder()``). ``None`` silently
        skips all decision recording. Domain code MUST NOT import the
        infrastructure ``record_decision`` directly; pass via DI instead.
        """
        if effective_window is None and context_window is None:
            raise ValueError(
                "ContextAssembler requires either 'effective_window' "
                "(preferred) or 'context_window' (deprecated). Both are None."
            )
        self._estimator = estimator
        self._safety_factor = safety_factor
        self._tool_compress_trigger_ratio = tool_compress_trigger_ratio
        self._tool_compress_target_chars = tool_compress_target_chars
        self._decision_recorder = decision_recorder

        if effective_window is not None:
            # New API: caller already applied system + reserved subtraction
            # via compute_effective_window(). Safety factor is still applied
            # in _compute_budget() to match the legacy expectation that the
            # estimator's output is over-counted by ~15%.
            self._effective_window = effective_window
            self._uses_shim = False
        else:
            # Deprecated shim: reproduce pre-C9 formula exactly. We
            # pre-subtract reserved_output_tokens here so _compute_budget()
            # can use the same formula for both paths.
            import warnings

            warnings.warn(
                "ContextAssembler(context_window=...) is deprecated; "
                "pass effective_window=compute_effective_window(...) instead. "
                "The shim will be removed in B5.5/B5.6.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._effective_window = context_window - reserved_output_tokens
            self._uses_shim = True

    # ── Public API ─────────────────────────────────────────────────────────────

    def assemble(self, messages: list[BaseMessage]) -> AssemblyResult:
        """Trim *messages* to fit within the token budget.

        Returns an AssemblyResult whose .messages is ready for the LLM call.
        The original messages list is never mutated.
        """
        if not messages:
            return AssemblyResult(messages=[], original_tokens=0, final_tokens=0)

        budget = self._compute_budget()
        original_tokens = self._estimator.estimate_messages(messages)

        if original_tokens <= budget:
            return AssemblyResult(
                messages=list(messages),
                original_tokens=original_tokens,
                final_tokens=original_tokens,
            )

        actions: list[str] = []
        groups = group_messages(messages)
        groups = self._mark_protected(groups)

        # ── Phase 1: compress ToolMessage content ──────────────────────────────
        trigger = budget * self._tool_compress_trigger_ratio
        current_tokens = self._estimate_groups(groups)
        if current_tokens > trigger:
            groups, compressed = self._phase1_compress(groups)
            if compressed:
                actions.append(
                    f"phase1:compress_tool_content "
                    f"(target_chars={self._tool_compress_target_chars})"
                )
                _safe_record_decision(self._decision_recorder, "context_assembler_trim", outcome="phase_1")

        # ── Phase 2: remove oldest non-protected tool_call groups ──────────────
        current_tokens = self._estimate_groups(groups)
        if current_tokens > budget:
            groups, removed = self._phase2_remove_tool_groups(groups, budget)
            if removed:
                actions.append(f"phase2:removed_tool_groups count={removed}")
                _safe_record_decision(self._decision_recorder, "context_assembler_trim", outcome="phase_2")

        # ── Phase 3: remove oldest non-protected standalone turns ──────────────
        current_tokens = self._estimate_groups(groups)
        if current_tokens > budget:
            groups, removed = self._phase3_remove_turns(groups, budget)
            if removed:
                actions.append(f"phase3:removed_turns count={removed}")
                _safe_record_decision(self._decision_recorder, "context_assembler_trim", outcome="phase_3")

        # ── Fallback: still over budget ────────────────────────────────────────
        current_tokens = self._estimate_groups(groups)
        if current_tokens > budget:
            logger.warning(
                "ContextAssembler: still over budget after all phases "
                "(budget=%d, tokens=%d). Returning trimmed remainder.",
                budget,
                current_tokens,
            )
            actions.append("fallback:over_budget_warning")

        final_messages = self._flatten_groups(groups)
        final_tokens = self._estimator.estimate_messages(final_messages)
        return AssemblyResult(
            messages=final_messages,
            original_tokens=original_tokens,
            final_tokens=final_tokens,
            actions=actions,
        )

    # ── Budget ─────────────────────────────────────────────────────────────────

    def _compute_budget(self) -> int:
        """Return the maximum number of input tokens available.

        B5 C9: ``self._effective_window`` is already net of system prompt
        budget and reserved output (via ``compute_effective_window()`` in
        the new API path). The legacy shim path pre-subtracted
        ``reserved_output_tokens`` at construction time, so both paths land
        at the same semantic: "remaining tokens for history, before applying
        safety_factor".
        """
        return int(self._effective_window / self._safety_factor)

    # ── Protection ────────────────────────────────────────────────────────────

    def _mark_protected(self, groups: list[MessageGroup]) -> list[MessageGroup]:
        """Return a new group list with protected flags set.

        Protected groups:
          - First SystemMessage group (index of first "system" group)
          - Last HumanMessage group
          - Last 2 groups (regardless of kind)
        """
        if not groups:
            return groups

        protected_indices: set[int] = set()

        # First system group
        for idx, g in enumerate(groups):
            if g.kind == "system":
                protected_indices.add(idx)
                break

        # Last human group
        last_human = self._last_human_idx(groups)
        if last_human >= 0:
            protected_indices.add(last_human)

        # Last 2 groups
        n = len(groups)
        if n >= 1:
            protected_indices.add(n - 1)
        if n >= 2:
            protected_indices.add(n - 2)

        result: list[MessageGroup] = []
        for idx, g in enumerate(groups):
            if idx in protected_indices and not g.protected:
                result.append(
                    MessageGroup(kind=g.kind, messages=g.messages, protected=True)
                )
            else:
                result.append(g)
        return result

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _flatten_groups(self, groups: list[MessageGroup]) -> list[BaseMessage]:
        """Flatten a list of MessageGroups back to a flat message list."""
        result: list[BaseMessage] = []
        for g in groups:
            result.extend(g.messages)
        return result

    def _estimate_groups(self, groups: list[MessageGroup]) -> int:
        """Estimate total token count for all groups."""
        return self._estimator.estimate_messages(self._flatten_groups(groups))

    def _last_human_idx(self, groups: list[MessageGroup]) -> int:
        """Return the index of the last "human" group, or -1 if none."""
        for idx in range(len(groups) - 1, -1, -1):
            if groups[idx].kind == "human":
                return idx
        return -1

    # ── Phase implementations ──────────────────────────────────────────────────

    def _phase1_compress(
        self, groups: list[MessageGroup]
    ) -> tuple[list[MessageGroup], bool]:
        """Compress ToolMessage content in non-protected tool_call groups.

        Returns (new_groups, any_compressed).
        """
        new_groups: list[MessageGroup] = []
        any_compressed = False
        for g in groups:
            if g.kind == "tool_call" and not g.protected:
                new_msgs: list[BaseMessage] = []
                changed = False
                for msg in g.messages:
                    if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
                        compressed = truncate_tool_content(
                            msg.content, max_chars=self._tool_compress_target_chars
                        )
                        if compressed != msg.content:
                            # Build a new ToolMessage with compressed content
                            new_tool = ToolMessage(
                                content=compressed,
                                tool_call_id=msg.tool_call_id,
                                name=msg.name or "",
                            )
                            new_msgs.append(new_tool)
                            changed = True
                            any_compressed = True
                        else:
                            new_msgs.append(msg)
                    else:
                        new_msgs.append(msg)
                if changed:
                    new_groups.append(
                        MessageGroup(
                            kind=g.kind, messages=new_msgs, protected=g.protected
                        )
                    )
                else:
                    new_groups.append(g)
            else:
                new_groups.append(g)
        return new_groups, any_compressed

    def _phase2_remove_tool_groups(
        self, groups: list[MessageGroup], budget: int
    ) -> tuple[list[MessageGroup], int]:
        """Remove oldest non-protected tool_call groups until under budget.

        Returns (new_groups, count_removed).
        """
        result = list(groups)
        removed = 0
        # Work from front (oldest) to back; rebuild after each removal
        i = 0
        while i < len(result):
            if self._estimate_groups(result) <= budget:
                break
            g = result[i]
            if g.kind == "tool_call" and not g.protected:
                del result[i]
                removed += 1
                # Do NOT advance i; next group slides into position i
            else:
                i += 1
        return result, removed

    def _phase3_remove_turns(
        self, groups: list[MessageGroup], budget: int
    ) -> tuple[list[MessageGroup], int]:
        """Remove oldest non-protected human/ai_text groups until under budget.

        As a last resort, also removes protected groups (but never the first
        SystemMessage or last HumanMessage).
        Returns (new_groups, count_removed).
        """
        removable_kinds = {"human", "ai_text"}
        result = list(groups)
        removed = 0

        # First pass: remove non-protected human/ai_text from oldest
        i = 0
        while i < len(result):
            if self._estimate_groups(result) <= budget:
                break
            g = result[i]
            if g.kind in removable_kinds and not g.protected:
                del result[i]
                removed += 1
            else:
                i += 1

        # Second pass: remove priority-protected groups (excluding first system
        # and last human) if still over budget
        if self._estimate_groups(result) > budget:
            first_system_idx: int | None = None
            for idx, g in enumerate(result):
                if g.kind == "system":
                    first_system_idx = idx
                    break

            i = 0
            while i < len(result):
                if self._estimate_groups(result) <= budget:
                    break
                g = result[i]
                last_human = self._last_human_idx(result)
                if g.kind not in removable_kinds or i == first_system_idx or i == last_human:
                    i += 1
                    continue
                del result[i]
                removed += 1
                # Update first_system_idx if it shifted
                if first_system_idx is not None and i < first_system_idx:
                    first_system_idx -= 1

        return result, removed
