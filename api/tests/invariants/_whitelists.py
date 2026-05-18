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
)

# INV-3: SSM subpackage MUST NOT call ApprovalStateWriter mutators.
INV3_WRITER_MUTATOR_NAMES: tuple[str, ...] = (
    "write", "write_audit_only", "delete_grant",
)

# INV-4-soft: sessions.status writes outside SSM are warning-only.
# Each entry must include a sunset PR / issue.
INV4_SOFT_KNOWN_OFFENDERS: tuple[tuple[str, str], ...] = (
    # (relative_path, sunset_ref)
    ("api/app/infrastructure/repositories/db_session_repository.py:231",
     "A4-1 INV-4-hard"),
    ("api/app/infrastructure/repositories/db_session_repository.py:421",
     "A4-1 INV-4-hard"),
)

# INV-5: tool_node _invoke_wrapper callsites must be PE-dominated.
INV5_REACT_GRAPH_PATH = "api/app/domain/services/graphs/react_graph.py"

# INV-5: functions in react_graph that are exempt from PE-dominance check.
# tool_node: legacy fallback path — when PE is enabled it delegates immediately
#   to _pe_dispatch (line ~1654) which enforces PE dominance. The remaining
#   body handles the legacy (PE-disabled) path and is expected to call
#   _invoke_wrapper without pe.evaluate. PE-3 will remove this legacy body.
# _legacy_*: any function starting with _legacy_ is also exempt.
INV5_SKIP_FUNCTION_NAMES: frozenset[str] = frozenset({
    "tool_node",
})

# INV-5 per-callsite documented safety-net whitelist.
#
# Some `_invoke_wrapper` callsites inside otherwise-PE-enforced functions are
# documented unreachable safety nets — they execute only if an earlier guard
# fails. These callsites legitimately lack a `pe.evaluate` / `pe_resume_outcomes`
# dominator because they are the *fallback* for the case where the upstream
# routing guard misclassified a call. They are expected to be unreachable after
# the guard, and PE-3 will remove them along with the legacy path.
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
    (
        "_pe_dispatch",
        1375,
        "PE-3",
        "PE-1 §2.5 (T15 P1#2 defensive) + Round 2 P1#2 per-call escape: "
        "the pre-loop guard now routes any non-PE-eligible call (mcp/a2a, "
        "skill creator/guide, or skill/native with operator flag off) and "
        "any unknown source to the legacy tool_node path for the WHOLE "
        "batch via ``is_pe_eligible_tool_source``, so this per-call branch "
        "is documented unreachable. The body is retained as a defensive "
        "fail-open (logged at ERROR) to avoid stalling the graph if a "
        "caller-side invariant ever regresses. PE-2/3 will lift mcp/a2a "
        "sources into PE and remove this branch. Documented at "
        "react_graph.py:1341-1358.",
    ),
)
