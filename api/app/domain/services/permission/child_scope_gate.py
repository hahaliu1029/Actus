"""ChildScopeGate — 4-way intersection (spec §5.4).

Position: DefaultPermissionEngine.evaluate prologue (BEFORE source loop).
Skipped when EvaluationContext.child_permission_context is None.

[r11] Tool-call cap stays in gate (per-call signal available).
Token cost + wallclock are runner-internal (gate has no signal).

INV-1b/2/3 safe: pure function; no writer/queue/SSM touches.

DEFERRED to PR-3+ (cold code in PR-2):
---------------------------------------
- WIRING: nothing in production currently constructs EvaluationContext
  with child_permission_context set (`grep child_permission_context= app/`
  returns no matches as of PR-2 staging). PR-3 ChildAgentRunnerFactory.build
  is responsible for plumbing the manifest through into the engine's
  per-task ctx. Until then this gate is cold code.

- PRE-PE BYPASS: react_graph.py has a pre-PE special branch for
  `message_ask_user` (around `_pe_dispatch` ~line 1214/1304) that executes
  BEFORE pe.evaluate runs. `message_ask_user` is in HARD_BLOCKED_FOR_CHILDREN
  here, but the pre-loop bypass would let a child execute it without the
  gate ever firing. PR-3 wiring task MUST either (a) gate the pre-loop on
  child_permission_context None, or (b) add an inline child-scope check at
  the bypass point. Codex R5 flagged this; ticket the fix as PR-3
  acceptance criteria.

- REPLAY FAST-PATH: react_graph.py's `pe_resume_outcomes` /
  `approved_tool_call_ids` replay paths re-execute approved tool calls
  without re-running ChildScopeGate. Lease expiry / revision drift
  detection therefore only fires at original evaluate time. PR-3 wiring
  MUST add a child-scope revalidation step on replay paths when
  child_permission_context is non-None.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.domain.models.work_unit import PathLease
    from app.domain.services.permission.child_permission_context import (
        ChildPermissionContext,
    )
    from app.domain.services.permission.context import EvaluationContext


class ScopeDecision(StrEnum):
    IN_SCOPE = "in_scope"
    OUT_OF_TOOL_ALLOWLIST = "out_of_tool_allowlist"
    HARD_BLOCKED = "hard_blocked"
    OUT_OF_PATH_LEASE = "out_of_path_lease"
    OP_MISMATCH = "op_mismatch"
    BUDGET_EXHAUSTED = "budget_exhausted"
    LEASE_EXPIRED = "lease_expired"
    REVISION_DRIFT = "revision_drift"


HARD_BLOCKED_FOR_CHILDREN: frozenset[str] = frozenset({
    # Shell — full live canonical set from tool_source_resolver.py:132-137.
    # Includes shell_read_output even though it's read-only: child tasks have
    # their own sandbox; reading parent shell state has no legitimate use and
    # the hard-block here defends against manifest typos / privilege confusion.
    "shell_execute", "shell_wait_process", "shell_kill_process",
    "shell_write_input", "shell_read_output",
    # User-interaction
    "message_ask_user", "message_notify_user",
    # Memory mutation
    "memory_save",
    # Subagent / runtime mutation (PR-3+ tools, defensive forward-include)
    "spawn_subagent",
    "install_skill",  # live name from tool_source_resolver.py:166
    "set_tool_approval",
    "publish_mailbox_envelope",
})

TYPED_WRITE_TOOL_NAMES: frozenset[str] = frozenset({
    "file_write", "file_str_replace",
})

# Tools subject to PathLease enforcement (typed writes + typed deletes).
# TYPED_WRITE_TOOL_NAMES is kept for _op_compatible's add/modify branch.
PATH_LEASED_TOOL_NAMES: frozenset[str] = TYPED_WRITE_TOOL_NAMES | frozenset({"file_delete"})


def extract_target_path(call) -> str | None:
    """Extract target file path from tool_args (filepath canonical, path fallback).

    Public helper so DefaultPermissionEngine.evaluate prologue can produce the
    SAME target_path as the gate uses for lease lookup — preventing the
    "gate denied, but ChildScopeViolation.target_path is None" drift.
    """
    args = getattr(call, "tool_args", None)
    if args is None or not hasattr(args, "get"):
        return None
    path = args.get("filepath")
    if path is None:
        path = args.get("path")
    return path if isinstance(path, str) else None


class ChildScopeGate:
    async def check_in_scope(
        self,
        call,
        ctx: "EvaluationContext",
        child_ctx: "ChildPermissionContext",
    ) -> ScopeDecision:
        if child_ctx is None:
            raise ValueError(
                "ChildScopeGate.check_in_scope called with child_ctx=None; "
                "DefaultPermissionEngine.evaluate prologue must only call gate "
                "when child_permission_context is non-None."
            )
        # 1. tool name in manifest allowlist?
        if call.tool_name not in child_ctx.spawn_manifest.allowed_tools:
            return ScopeDecision.OUT_OF_TOOL_ALLOWLIST
        # 2. hardcoded HARD_BLOCKED (overrides allowlist)
        if call.tool_name in HARD_BLOCKED_FOR_CHILDREN:
            return ScopeDecision.HARD_BLOCKED
        # 3. path lease check for typed write/delete (PATH_LEASED_TOOL_NAMES)
        if call.tool_name in PATH_LEASED_TOOL_NAMES:
            target_path = self._extract_target_path(call)
            if target_path is None:
                return ScopeDecision.OUT_OF_PATH_LEASE
            lease = self._lookup_lease(target_path, child_ctx.spawn_manifest.path_leases)
            if lease is None:
                return ScopeDecision.OUT_OF_PATH_LEASE
            if not self._op_compatible(call, lease.op):
                return ScopeDecision.OP_MISMATCH
        # 4. budget — tool call count only (spec §5.4 r11)
        if self._tool_call_budget_exhausted(child_ctx):
            return ScopeDecision.BUDGET_EXHAUSTED
        # 5. [r3 P1-2] lease expiry check (coerce naive datetime → UTC to avoid TypeError)
        if child_ctx.lease_expiry is not None:
            expiry = child_ctx.lease_expiry
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) > expiry:
                return ScopeDecision.LEASE_EXPIRED
        # 6. revision drift
        if ctx.session_mode_revision != child_ctx.session_mode_revision:
            return ScopeDecision.REVISION_DRIFT
        return ScopeDecision.IN_SCOPE

    @staticmethod
    def _extract_target_path(call) -> str | None:
        """Delegate to module-level extract_target_path (shared with engine prologue)."""
        return extract_target_path(call)

    @staticmethod
    def _lookup_lease(
        target_path: str,
        leases: "tuple[PathLease, ...]",
    ) -> "PathLease | None":
        for lease in leases:
            if lease.path == target_path:
                return lease
        return None

    @staticmethod
    def _op_compatible(call, lease_op: str) -> bool:
        if lease_op in ("modify", "add"):
            return call.tool_name in TYPED_WRITE_TOOL_NAMES
        if lease_op == "delete":
            return call.tool_name == "file_delete"
        return False

    @staticmethod
    def _tool_call_budget_exhausted(child_ctx) -> bool:
        # PR-2 minimal: max_tool_calls==0 means exhausted; PR-4 wires cumulative counter
        return child_ctx.budget.max_tool_calls <= 0
