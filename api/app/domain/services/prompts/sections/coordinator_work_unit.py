"""C2 PR-4 §8.7 — Coordinator child work-unit prompt section.

A standalone, functional builder (NOT a Section subclass) for the restricted
prompt fed to coordinator-step children. Lives outside the registry-driven
``PromptAssembler.assemble`` pipeline because:

1. The coordinator child's prompt is fixed-shape (objective + phase +
   allowed_paths + optional expected_result_schema). It does not need
   priority-ordered budget arbitration like a full agent prompt.
2. The child runs with NO skill_context / tool_summary / conversation_summaries
   — those are root-only concerns and would over-permission the child.

``build_coordinator_work_unit_section`` returns the raw markdown for the
work-unit block. ``PromptAssembler.build_minimal_for_coordinator_child``
(in ``assembler.py``) is the caller that prepends identity/behavior/output
guidance and joins the sections with the canonical separator.
"""
from __future__ import annotations

from typing import Literal


_EXPLORATION_GUIDANCE = (
    "This is the EXPLORATION phase. You should READ files, analyze, and "
    "produce a structured proposed_write_plan in your final response. You "
    "CANNOT write files (no write lease in this phase). The parent will "
    "review your proposal and decide whether to dispatch a write-phase child."
)

_WRITE_GUIDANCE = (
    "This is the WRITE phase. You are authorized to write to the paths "
    "listed below. ChildScopeGate enforces the lease at runtime — attempts "
    "to write outside the lease will be denied and surface as "
    "NEEDS_AUTHORIZATION to the parent."
)


def build_coordinator_work_unit_section(
    *,
    objective: str,
    phase: Literal["exploration", "write"],
    allowed_paths: list[str],
    work_unit_id: str,
    expected_result_schema: str | None = None,
) -> str:
    """Build the markdown body for the coordinator child's work-unit block.

    ``phase`` is narrowed to ``Literal["exploration", "write"]`` (r1 P2#5):
    runtime callers (CoordinatorChildRunner) get the value from
    ``WorkUnit.phase`` which is itself the same Literal — but a non-conforming
    string would otherwise silently fall through to WRITE guidance and emit
    an authorization-tone prompt for a non-write phase. Fail closed instead.
    """
    if phase not in ("exploration", "write"):
        raise ValueError(
            f"phase must be 'exploration' or 'write', got {phase!r}"
        )
    paths_block = (
        "\n".join(f"- {p}" for p in allowed_paths)
        if allowed_paths
        else "_(none — read-only exploration)_"
    )
    guidance = (
        _EXPLORATION_GUIDANCE if phase == "exploration" else _WRITE_GUIDANCE
    )
    schema_block = ""
    if expected_result_schema:
        schema_block = f"\n\n**Expected result schema:**\n{expected_result_schema}"

    return (
        f"## Your Work Unit\n\n"
        f"**Work Unit**: {work_unit_id}\n\n"
        f"**Objective**: {objective}\n\n"
        f"**Phase**: {phase}\n\n"
        f"**Authorized paths**:\n{paths_block}\n\n"
        f"{guidance}{schema_block}"
    )
