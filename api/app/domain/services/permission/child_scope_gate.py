"""ChildScopeGate — coordinator-child tool authorization boundary (spec §5.4).

Pure 6-step sequential check (the "4-way intersection" name is historical):
allowlist -> HARD_BLOCKED -> path-lease/op -> budget-cap -> lease-expiry ->
revision-drift. Skipped when EvaluationContext.child_permission_context is None.

[r11] Tool-call cap stays in gate (per-call signal available).
Token cost + wallclock are runner-internal (gate has no signal).

INV-1b/2/3 safe: pure function; no writer/queue/SSM touches.

LIVE wiring (C2b, 2026-06): two callers pass the SAME
EvaluationContext.child_permission_context and call check_in_scope (DRY):
- DefaultPermissionEngine.evaluate prologue — defense-in-depth (BEFORE the
  source loop).
- react_graph.tool_node entry guard ``_enforce_child_scope_or_raise`` — the
  PRODUCTION path. Coordinator children run with tool_confirmation.enabled=False
  so they never enter the PE branch; the tool_node-entry guard fires BEFORE the
  PE/legacy fork over every pending tool_call, so message_ask_user (HARD_BLOCKED)
  and replay paths (``pe_resume_outcomes`` / ``approved_tool_call_ids`` — the
  guard's skip set is ``completed_tool_call_prefix`` ONLY, so pre-approved
  replays are re-checked) are all covered. cpc reaches cfg via
  PlannerReActFlow._build_config's child branch.

Known limitations (spec §6, deferred — NOT bugs introduced here):
- Cumulative tool-count cap: budget is still the static ``max_tool_calls <= 0``
  kill-switch, not a cumulative counter.
- Intra-batch race; symlink / non-canonical lease escape (exact-string match);
  manifest ``allowed_tools`` has no known-name validator.
- Child session DB row termination on violation: the runner re-raises
  ChildScopeViolation WITHOUT a terminal status write (sibling of the
  ``CancelledByEventError`` PR-5 deferred gap), so a denied child's session row
  stays RUNNING until the coordinator/PR-5 runner_starter adapter reaps it.
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
