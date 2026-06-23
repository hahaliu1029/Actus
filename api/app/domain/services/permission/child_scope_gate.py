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
- Intra-batch race; symlink lease escape; manifest ``allowed_tools`` has no
  known-name validator.

Resolved (C2b §6-lim2 follow-up): child session DB row termination on
violation/event-cancel — the runner now writes ``COMPLETED`` + ``"natural"``
through the SSM (INV-4) in both the ``ChildScopeViolation`` and
``CancelledByEventError`` arms before re-raising, so a denied/cancelled child no
longer leaks as a zombie ``RUNNING``. The deny/cancel/budget granularity stays
in the coordinator envelope (NEEDS_AUTHORIZATION / CANCEL_ACK), not the DB row.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import TYPE_CHECKING

from app.domain.models.path_validation import (
    CoordinatorPathContractError,
    validate_coordinator_path,
)
from app.domain.services.coordinator_shell_mode_flag import (
    is_coordinator_shell_mode_enabled,
)

if TYPE_CHECKING:
    from app.domain.models.work_unit import PathLease, TreeLease
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

# [S2 §3.5] The 5 raw-shell entries inside HARD_BLOCKED_FOR_CHILDREN that the
# shell-mode dual-loosening conditionally un-blocks. Must be the EXACT live
# canonical shell set (tool_source_resolver.py:132-137). Non-shell hard-blocks
# (message_*/memory_save/spawn_subagent/install_skill/set_tool_approval/
# publish_mailbox_envelope) stay unconditional.
SHELL_HARD_BLOCKED_NAMES: frozenset[str] = frozenset({
    "shell_execute", "shell_wait_process", "shell_kill_process",
    "shell_write_input", "shell_read_output",
})


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
        # 2. hardcoded HARD_BLOCKED (overrides allowlist).
        # [S2 §3.5] Dual-loosening: the 5 raw-shell entries skip HARD_BLOCK
        # iff master flag ON AND this child's shell_mode is True; non-shell
        # hard-blocks stay unconditional. An un-blocked shell tool then falls
        # through steps 3-6 with NO path-lease check (shell has no path arg) —
        # capture happens via the snapshot differ (§3.2), not the typed gate.
        if call.tool_name in HARD_BLOCKED_FOR_CHILDREN:
            _shell_allowed = (
                call.tool_name in SHELL_HARD_BLOCKED_NAMES
                and child_ctx.shell_mode
                and is_coordinator_shell_mode_enabled()
            )
            if not _shell_allowed:
                return ScopeDecision.HARD_BLOCKED
        # 3. path lease check for typed write/delete (PATH_LEASED_TOOL_NAMES)
        if call.tool_name in PATH_LEASED_TOOL_NAMES:
            target_path = self._extract_target_path(call)
            if target_path is None:
                return ScopeDecision.OUT_OF_PATH_LEASE
            lease = self._lookup_lease(target_path, child_ctx.spawn_manifest.path_leases)
            if lease is not None:
                # [S2 §3.4] Exact file lease GOVERNS — op must match, NO tree
                # fallback (a tree lease can never widen a file lease's op, F21).
                if not self._op_compatible(call, lease.op):
                    return ScopeDecision.OP_MISMATCH
            else:
                # [S2 §3.5] No exact file lease: a covering TreeLease authorizes
                # a typed op=add write ONLY (tree leases are ADD-only). Such a
                # write is captured by the snapshot differ, not the typed
                # extractor, so gate-allows / capture-sees stay consistent.
                # GATED ON BOTH the master flag AND this child's shell_mode —
                # identical fail-safe to the step-2 shell un-block. With the flag
                # OFF (default), a stale/hand-crafted manifest carrying a
                # tree_lease can NEVER widen the gate → OUT_OF_PATH_LEASE.
                _tree_allowed = (
                    child_ctx.shell_mode
                    and is_coordinator_shell_mode_enabled()
                    and self._tree_covers_add(
                        call, target_path, child_ctx.spawn_manifest.tree_leases
                    )
                )
                if not _tree_allowed:
                    return ScopeDecision.OUT_OF_PATH_LEASE
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
        # Legacy compatibility: older fixtures / stored manifests may carry
        # absolute non-workspace paths (e.g. "/x"). Preserve exact-match behavior
        # for those shapes while adding the single-path canonical fallback below.
        for lease in leases:
            if lease.path == target_path:
                return lease
        try:
            canonical_target = validate_coordinator_path(target_path)
        except CoordinatorPathContractError:
            return None
        for lease in leases:
            try:
                canonical_lease = validate_coordinator_path(lease.path)
            except CoordinatorPathContractError:
                continue
            if canonical_lease == canonical_target:
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
    def _tree_covers_add(
        call,
        target_path: str,
        tree_leases: "tuple[TreeLease, ...]",
    ) -> bool:
        """[S2 §3.5] True iff a typed op=add write to ``target_path`` is covered
        by an ADD-only TreeLease. Only file_write / file_str_replace (the
        TYPED_WRITE_TOOL_NAMES = add/modify shapes) can be tree-covered; a
        file_delete is never covered (tree=ADD-only). Coverage is POSIX
        component-aware via ``tree_contains``.

        [codex PR-5 R2 P2] Canonicalize ``target_path`` first, mirroring
        ``_lookup_lease``. ``tree_contains`` requires BOTH sides canonical
        workspace-relative (``lease.prefix`` is canonicalized at build via
        ``validate_coordinator_tree_prefix``); a raw absolute target (e.g.
        ``/home/ubuntu/workspace/gen/x.py``) would otherwise be wrongly rejected.
        A non-canonicalizable target stays fail-closed → False."""
        from app.domain.models.path_validation import tree_contains

        if call.tool_name not in TYPED_WRITE_TOOL_NAMES:
            return False  # file_delete shape — tree is ADD-only
        try:
            canonical_target = validate_coordinator_path(target_path)
        except CoordinatorPathContractError:
            return False  # fail-closed: a non-canonicalizable path is uncovered
        for lease in tree_leases:
            if "add" not in lease.ops:
                continue
            if tree_contains(lease.prefix, canonical_target):
                return True
        return False

    @staticmethod
    def _tool_call_budget_exhausted(child_ctx) -> bool:
        # PR-2 minimal: max_tool_calls==0 means exhausted; PR-4 wires cumulative counter
        return child_ctx.budget.max_tool_calls <= 0
