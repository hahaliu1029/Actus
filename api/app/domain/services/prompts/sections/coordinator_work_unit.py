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
    "listed below. Write to EXACTLY those paths — each is a directory-qualified "
    "workspace-relative path (e.g. 'workspace/foo.py', 'api/bar.py'); do NOT "
    "write a bare filename at the workspace root (e.g. 'foo.py') and do NOT "
    "invent a different directory. ChildScopeGate enforces the lease at runtime "
    "— attempts to write outside the lease will be denied and surface as "
    "NEEDS_AUTHORIZATION to the parent."
)


_TREE_GUIDANCE = (
    "**Authorized directory trees (ADD-ONLY)**:\n"
    "{trees}\n\n"
    "These are ADD-ONLY directory trees: you MAY create NEW files anywhere "
    "under each prefix above, but you may NOT modify or delete EXISTING files "
    "there (to modify/delete a file you need an exact path lease above). "
    "ChildScopeGate / snapshot capture enforces this at runtime — creating a "
    "file outside these trees, or modifying/deleting an existing file inside "
    "them, will be rejected and surface as NEEDS_AUTHORIZATION to the parent."
)


def build_coordinator_work_unit_section(
    *,
    objective: str,
    phase: Literal["exploration", "write"],
    allowed_paths: list[str],
    work_unit_id: str,
    allowed_trees: list[str] = (),
    expected_result_schema: str | None = None,
) -> str:
    """Build the markdown body for the coordinator child's work-unit block.

    ``phase`` is narrowed to ``Literal["exploration", "write"]`` (r1 P2#5):
    runtime callers (CoordinatorChildRunner) get the value from
    ``WorkUnit.phase`` which is itself the same Literal — but a non-conforming
    string would otherwise silently fall through to WRITE guidance and emit
    an authorization-tone prompt for a non-write phase. Fail closed instead.

    ``allowed_trees`` (C2-full S2 PR-5) are workspace-relative ADD-only directory
    prefixes from the unit's ``write_tree_lease`` (shell-mode write units). When
    non-empty, an ADD-only tree block is appended so the child knows which trees
    it may CREATE new files under. EMPTY ``allowed_trees`` (the non-shell /
    flag-OFF case) renders BYTE-FOR-BYTE identically to the legacy no-trees call:
    no tree block, and the ``_(none — read-only exploration)_`` placeholder still
    applies — but ONLY when BOTH ``allowed_paths`` AND ``allowed_trees`` are
    empty, so a tree-only write unit is never mislabeled as read-only.
    """
    if phase not in ("exploration", "write"):
        raise ValueError(
            f"phase must be 'exploration' or 'write', got {phase!r}"
        )
    # [codex PR-5 R3 P2] Fail-closed: an exploration unit is read-only and
    # carries NO leases (WorkUnit enforces exploration ⇒ no write/tree lease
    # upstream). Reject a direct-helper misuse that would otherwise emit
    # contradictory exploration guidance alongside an ADD-only tree block.
    if phase == "exploration" and allowed_trees:
        raise ValueError(
            "exploration phase must not carry allowed_trees (read-only)"
        )
    if allowed_paths:
        paths_block = "\n".join(f"- {p}" for p in allowed_paths)
    elif allowed_trees:
        # [codex PR-5 R4 P2] tree-only write unit: there are no exact paths, the
        # real authorization is the ADD-only tree block below. Redirect to it
        # rather than leave an empty list under "write to EXACTLY those paths".
        paths_block = (
            "_(none — create NEW files under the authorized directory "
            "trees below)_"
        )
    else:
        paths_block = "_(none — read-only exploration)_"
    guidance = (
        _EXPLORATION_GUIDANCE if phase == "exploration" else _WRITE_GUIDANCE
    )
    # [PR-5] ADD-only tree block, only when tree leases are present. Empty
    # allowed_trees ⇒ no block ⇒ byte-for-byte identical to the legacy output.
    tree_block = ""
    if allowed_trees:
        trees = "\n".join(f"- {t}" for t in allowed_trees)
        tree_block = "\n\n" + _TREE_GUIDANCE.format(trees=trees)
    schema_block = ""
    if expected_result_schema:
        schema_block = f"\n\n**Expected result schema:**\n{expected_result_schema}"

    return (
        f"## Your Work Unit\n\n"
        f"**Work Unit**: {work_unit_id}\n\n"
        f"**Objective**: {objective}\n\n"
        f"**Phase**: {phase}\n\n"
        f"**Authorized paths**:\n{paths_block}\n\n"
        f"{guidance}{tree_block}{schema_block}"
    )
