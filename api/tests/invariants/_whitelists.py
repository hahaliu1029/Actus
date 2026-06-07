# api/tests/invariants/_whitelists.py
"""Centralized whitelist constants for invariant tests.

WHITELIST CHANGES ARE A SECURITY DECISION. Any modification to this
file MUST be reviewed by codex (per feedback_pr_boundary_codex_audit.md).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
API_APP = REPO_ROOT / "api" / "app"

# INV-1a: only files allowed to write the two approval tables directly.
# ApprovalStateWriter wraps two repository modules (one per table) — both
# must be whitelisted because both perform ORM-level INSERT / UPDATE / DELETE.
INV1A_TABLE_WRITER_WHITELIST: tuple[str, ...] = (
    "api/app/infrastructure/repositories/db_approval_grant_repository.py",
    "api/app/infrastructure/repositories/db_tool_approval_log_repository.py",
)

# INV-1b: only files allowed to import / receive / call ApprovalStateWriter
# methods. PermissionEngine internals + DI factory only.
# NOTE: agent_service.py contains _preflight_resume_tool_confirmation_legacy
# which is the legacy fail-open path retained for PE-0 (Phase 8.1 comment).
# Direct writer calls there are intentional and documented — whitelisted until
# A4-1 removes the legacy path.
INV1B_WRITER_CALLER_WHITELIST: tuple[str, ...] = (
    "api/app/domain/services/permission/default_engine.py",
    "api/app/interfaces/service_dependencies.py",  # DI assembly only
    "api/app/application/services/approval_state_writer.py",  # writer module itself
    "api/app/application/services/agent_service.py",  # legacy preflight path (A4-1 sunset)
)

# INV-2: PermissionEngine subpackage MUST NOT call SSM mutators.
INV2_SSM_MUTATOR_NAMES: tuple[str, ...] = (
    "request_takeover",
    "release_takeover",
    "enter_finishing",
    "complete",
    "transition",
    "set_mode",      # A4-1: PE must not drive status writes
    "terminate",     # A4-1: PE must not drive status writes
)

# INV-3: SSM subpackage MUST NOT call ApprovalStateWriter mutators.
INV3_WRITER_MUTATOR_NAMES: tuple[str, ...] = (
    "write", "write_audit_only", "delete_grant",
)

# INV-4-hard (A4-1): the three session-status repo mutators. Only the SSM
# subpackage + the exempt repo files may CALL these (Gate A).
INV4_SESSION_STATUS_MUTATOR_NAMES: tuple[str, ...] = (
    "update_status",
    "update_to_terminal",
    "transition_status",
)

# INV-5: tool_node _invoke_wrapper callsites must be PE-dominated.
INV5_REACT_GRAPH_PATH = "api/app/domain/services/graphs/react_graph.py"

# INV-5: functions in react_graph that are exempt from PE-dominance check.
# tool_node: when PE is enabled it delegates immediately to _pe_dispatch which
#   enforces PE dominance. The residual tool_node body retains TWO permanent /
#   long-lived non-PE-dominated _invoke_wrapper callsites that are NOT removed by
#   PE-4:
#     1. meta-tool / unknown direct-execute passthrough (skill-creator/guide,
#        mcp-discovery, unresolvable-source sentinel) — non-PE-eligible carve-outs
#        by design, permanent.
#     2. legacy-approved interrupt-replay bridge (pre-approved native
#        direct-execute / missing-claim_nonce path) — sunsets with the legacy
#        interrupt fallback, a LATER epic (NOT PE-4).
#   The earlier "PE-4 removes this legacy body" claim was over-optimistic; PE-4
#   removed the native risk gate + per-source flags, not the whole legacy body.
# _legacy_*: any function starting with _legacy_ is also exempt.
INV5_SKIP_FUNCTION_NAMES: frozenset[str] = frozenset({
    "tool_node",
})

# INV-5 per-callsite documented safety-net whitelist.
#
# Format: tuple of (function_name, callsite_lineno, sunset_ref, justification).
# The PE-0 round 32 strict end_lineno dominance filter surfaced these; before
# the fix, the loose `lineno <` filter silently swept past-callsite references
# from the same parent for-loop into the prior-statements set.
#
# Whitelist changes MUST be reviewed by codex per
# feedback_pr_boundary_codex_audit.md.
INV5_CALLSITE_SAFETY_NET_WHITELIST: tuple[
    tuple[str, int, str, str], ...
] = (
    # PE-4c: the lone _pe_dispatch per-call safety-net (formerly a fail-open
    # _invoke_wrapper) was converted to fail-CLOSED (Denied), so it no longer
    # executes a tool without PE dominance — and the whitelist entry that
    # exempted it is removed. No documented fail-open safety-net callsites
    # remain. Re-adding any entry here is a SECURITY decision (codex review
    # required per feedback_pr_boundary_codex_audit.md).
)
