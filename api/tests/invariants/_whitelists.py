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
    "emit_session_mode_changed",  # A4-2: PE must not emit control-mode events
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

# INV-5: files scanned for _invoke_wrapper reachability (v2 rule 3).
# The executor PACKAGE is scanned to assert ZERO escape surface: no file in it
# may ever contain _invoke_wrapper / the factories / pe.evaluate, and each may
# only import from the allowlist.
INV5_REACT_GRAPH_PATH = "api/app/domain/services/graphs/react_graph.py"  # kept: primary scan target

# The executor package directory (relative to REPO_ROOT). The zero-escape +
# import-allowlist rules (Rule 3 / R6#2) apply to EVERY .py file in this
# package, not just batch_tool_executor.py — a forbidden escape (importlib,
# _invoke_wrapper, pe.evaluate) added in a SIBLING module (e.g.
# tool_call_stream_collector.py, __init__.py) must also be caught.
# codex R1-P2: previously only batch_tool_executor.py was scanned, leaving
# sibling modules an unscanned escape surface.
INV5_EXECUTOR_PACKAGE_DIR = "api/app/domain/services/executor"

# Kept for backward compatibility / anti-drift: the primary executor module.
# It remains a member of the globbed package set below.
INV5_EXECUTOR_MODULE_PATH = "api/app/domain/services/executor/batch_tool_executor.py"


def executor_package_files() -> tuple[str, ...]:
    """Dynamic RECURSIVE glob of every ``.py`` file in the executor package
    (relative to REPO_ROOT), sorted for determinism.

    Chosen over a hard-coded file list so that FUTURE sibling modules added to
    the package are AUTOMATICALLY brought under the INV-5 zero-escape +
    import-allowlist scan without a whitelist edit. Adding a file here is not a
    security decision — it is scanned, not exempted.

    codex R2-P2: uses ``rglob`` (not top-level ``glob``) so a future
    ``executor/subpkg/backdoor.py`` is scanned too — a subdirectory module was
    previously an UNSCANNED escape surface. Excludes ``__pycache__`` (compiled
    ``.pyc`` files never match ``*.py``; source dirs under __pycache__ do not
    exist).
    """
    pkg = REPO_ROOT / INV5_EXECUTOR_PACKAGE_DIR
    return tuple(sorted(
        str(p.relative_to(REPO_ROOT)).replace("\\", "/")
        for p in pkg.rglob("*.py")
        if "__pycache__" not in p.parts
    ))


# Rule 1 (sink) scan surface: react_graph (the ONLY file that legitimately
# carries _invoke_wrapper callsites) PLUS every executor package file (all of
# which must carry ZERO callsites — the expected zero-escape state).
INV5_SCAN_PATHS: frozenset[str] = frozenset(
    {INV5_REACT_GRAPH_PATH} | set(executor_package_files())
)

# INV-5 v2 (B1-1a): the ONLY two factory names whose nested thunk may call
# _invoke_wrapper. Adding a name here is a SECURITY decision (codex review
# required per feedback_pr_boundary_codex_audit.md).
INV5_SINK_FACTORY_NAMES: frozenset[str] = frozenset({
    "_make_execute_thunk",
    "_legacy_make_execute_thunk",
})

# INV-5 v2 final (B1-1a Task 4): NO function-level exemptions remain.
# tool_node's legacy body now routes execution through
# _legacy_make_execute_thunk (whose CALLSITES are exempt from dominance —
# the precise successor of the old fail-open semantics). Adding any name
# back here is a SECURITY decision (codex review required).
INV5_SKIP_FUNCTION_NAMES: frozenset[str] = frozenset()

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
