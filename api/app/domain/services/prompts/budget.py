"""System prompt token budget governance.

B5 C1: defines ``SystemPromptBudget`` dataclass and the canonical
``compute_effective_window`` helper used by both ``ContextAssembler`` (sync
in-step trimmer) and ``GradualCompactor`` (async LLM summarization) to agree
on the same "effective window for message history".

Without a single helper, the two layers can drift in their effective
window calculations and the LLM call ends up over the total context window.
See B5 design doc "三层预算责任划分" for the rationale.

C1 ships only the data structures and helper. Wiring into
``ContextOverflowConfig`` and the actual compactor calls happens in C9.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemPromptBudget:
    """Token budget for the system prompt segment.

    The budget is exclusive of message history. ``PromptAssembler`` uses
    ``max_tokens`` as a hard cap (sections with priority below
    ``critical_priority_min`` get dropped if the total exceeds it).

    ``ContextOverflowConfig.system_prompt_max_tokens`` (added in C9) is the
    canonical configuration source. C1 just defines the dataclass.
    """

    max_tokens: int
    critical_priority_min: int = 8
    warn_on_overflow: bool = True


def compute_effective_window(
    total_context_window: int,
    system_prompt_max_tokens: int,
    reserved_output_tokens: int,
    min_ratio: float = 0.1,
) -> int:
    """Single source of truth for "how many tokens can message history use".

    The formula is:

        effective_window = max(
            total_context_window - system_prompt_max_tokens - reserved_output_tokens,
            int(total_context_window * min_ratio),  # 10% floor
        )

    The floor protects against misconfiguration where ``system_prompt_max_tokens``
    plus ``reserved_output_tokens`` consume the entire window (or more), which
    would yield zero or negative effective window.

    Used by:
    - ``ContextAssembler.__init__`` (caller computes effective_window externally)
    - ``planner_react._check_overflow`` (passed as ``context_window`` to ``GradualCompactor.try_compact``)

    Both call sites must use the same helper to stay in sync.

    Raises ``ValueError`` if ``min_ratio`` is outside [0.0, 1.0] — a ratio
    above 1.0 would make the floor exceed total window, which is nonsense.
    """
    if not (0.0 <= min_ratio <= 1.0):
        raise ValueError(
            f"min_ratio must be in [0.0, 1.0], got {min_ratio}"
        )
    if total_context_window <= 0:
        return 0
    raw = total_context_window - system_prompt_max_tokens - reserved_output_tokens
    floor = int(total_context_window * min_ratio)
    return max(raw, floor)
