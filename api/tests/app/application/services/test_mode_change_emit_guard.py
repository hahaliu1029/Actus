"""A4-0 T-GUARD: every control-mode update_status write must emit a
SessionModeChangedEvent. Scans agent_service.py (6 HTTP sites) and
agent_task_runner.py (allowlisted RUNNING writes + the agent-loop emitter).
A new control-mode write site not accounted for here fails CI."""
from __future__ import annotations

import ast
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[4]  # tests/app/application/services → api/
AGENT_SERVICE = API_ROOT / "app" / "application" / "services" / "agent_service.py"
RUNNER = API_ROOT / "app" / "domain" / "services" / "agent_task_runner.py"

CONTROL_STATUSES = frozenset({"RUNNING", "WAITING", "TAKEOVER", "TAKEOVER_PENDING"})

# The 6 HTTP-takeover methods that MUST emit (F5).
EXPECTED_SERVICE_EMITTERS = frozenset({
    "start_takeover",
    "_complete_takeover_after_cancel",
    "reject_takeover",
    "end_takeover",
    "reopen_takeover",
    "_handle_takeover_lease_timeout",
})
# Runner literal-update_status RUNNING sites are user-message-driven run start/
# resume, NOT control hand-offs — allowlisted (no emit). The agent-loop
# WAITING/TAKEOVER_PENDING writes go through _emit_control_mode_changed (a
# variable `to`), so they do NOT appear as literal SessionStatus.X sites.
RUNNER_ALLOWLIST_RUNNING = frozenset({"invoke", "resume"})
RUNNER_EMITTER = "_emit_control_mode_changed"
# R4#P2: the literal scan is blind to update_status(sid, <variable>). The ONLY
# legitimate variable-status writer is the runner's _emit_control_mode_changed
# (it writes update_status(self._session_id, to)). Any other variable-status site
# is a drift blind spot that must be reviewed for A4-0 emission.
RUNNER_VARIABLE_STATUS_ALLOWLIST = frozenset({"_emit_control_mode_changed"})


def _parents(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _nearest_func(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str | None:
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, (ast.AsyncFunctionDef, ast.FunctionDef)):
            return cur.name
        cur = parents.get(cur)
    return None


def _find_function(tree: ast.Module, name: str):
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    return None


def _literal_control_update_status_sites(tree: ast.Module) -> dict[str, list[str]]:
    """Map enclosing-function name → LIST of SessionStatus.<CONTROL> literals
    written via update_status(...), one entry PER callsite (duplicates preserved
    so per-function COUNTS catch same-function drift — R1#P2). Only LITERAL
    SessionStatus.X args count (a variable `to` is invisible here — by design,
    see RUNNER_EMITTER)."""
    parents = _parents(tree)
    sites: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "update_status"):
            continue
        if len(node.args) < 2:
            continue
        arg = node.args[1]
        if (
            isinstance(arg, ast.Attribute)
            and isinstance(arg.value, ast.Name)
            and arg.value.id == "SessionStatus"
            and arg.attr in CONTROL_STATUSES
        ):
            fn = _nearest_func(node, parents)
            if fn is not None:
                sites.setdefault(fn, []).append(arg.attr)
    return sites


def _emits_mode_changed(func: ast.AST) -> bool:
    """True if the function subtree calls _emit_session_mode_changed OR
    _emit_control_mode_changed OR constructs SessionModeChangedEvent."""
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr in (
                "_emit_session_mode_changed",
                "_emit_control_mode_changed",
            ):
                return True
            if isinstance(f, ast.Name) and f.id == "SessionModeChangedEvent":
                return True
    return False


def _nonliteral_update_status_functions(tree: ast.Module) -> set[str]:
    """R4#P2: enclosing-function names of update_status(...) calls whose status
    arg is NOT a literal SessionStatus.X (e.g. a variable). The literal-only site
    scan above cannot see these, so they are a drift blind spot — every such site
    must be explicitly accounted for. FINISHING/other LITERAL writes are
    SessionStatus.X and are NOT flagged here (only genuine non-literals)."""
    parents = _parents(tree)
    funcs: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "update_status"):
            continue
        if len(node.args) < 2:
            continue
        arg = node.args[1]
        is_literal_session_status = (
            isinstance(arg, ast.Attribute)
            and isinstance(arg.value, ast.Name)
            and arg.value.id == "SessionStatus"
        )
        if not is_literal_session_status:
            fn = _nearest_func(node, parents)
            if fn is not None:
                funcs.add(fn)
    return funcs


def test_agent_service_files_exist() -> None:
    assert AGENT_SERVICE.exists(), AGENT_SERVICE
    assert RUNNER.exists(), RUNNER


def test_agent_service_control_mode_sites_match_expected_set() -> None:
    tree = ast.parse(AGENT_SERVICE.read_text(encoding="utf-8"))
    sites = _literal_control_update_status_sites(tree)
    assert set(sites) == EXPECTED_SERVICE_EMITTERS, (
        f"control-mode update_status sites in agent_service.py drifted: "
        f"{set(sites) ^ EXPECTED_SERVICE_EMITTERS}. A4-0 requires every "
        f"control-mode write to emit SessionModeChangedEvent — wire an emit and "
        f"update EXPECTED_SERVICE_EMITTERS."
    )
    # R1#P2: each of the 6 emitters has EXACTLY ONE control-mode write today. A
    # second write added to an existing emitter would still pass
    # _emits_mode_changed (it emits once), so assert the count to catch it.
    for name, statuses in sites.items():
        assert len(statuses) == 1, (
            f"{name} now has {len(statuses)} control-mode update_status writes "
            f"({statuses}); A4-0 expects exactly one per emitter — verify the new "
            f"write also emits SessionModeChangedEvent and update this guard."
        )


def test_agent_service_each_emitter_emits_mode_changed() -> None:
    tree = ast.parse(AGENT_SERVICE.read_text(encoding="utf-8"))
    for name in EXPECTED_SERVICE_EMITTERS:
        func = _find_function(tree, name)
        assert func is not None, f"{name} not found in agent_service.py"
        assert _emits_mode_changed(func), (
            f"{name} writes a control-mode status but does not emit "
            f"SessionModeChangedEvent (A4-0 INV / biggest-limitation guard)."
        )


def test_runner_literal_control_sites_are_allowlisted_running_only() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    sites = _literal_control_update_status_sites(tree)
    assert set(sites) == RUNNER_ALLOWLIST_RUNNING, (
        f"agent_task_runner.py literal control-mode update_status sites drifted: "
        f"{set(sites) ^ RUNNER_ALLOWLIST_RUNNING}. The agent-loop WAITING/"
        f"TAKEOVER_PENDING writes must flow through {RUNNER_EMITTER}; the only "
        f"literal sites are the allowlisted RUNNING run-start/resume writes."
    )
    # R1#P2: exact per-function callsite COUNTS lock the drift surface without
    # brittle line numbers. invoke restores RUNNING twice (run-start + finishing
    # restore); resume restores RUNNING once. A new RUNNING write in either
    # function changes the count and trips this guard (a same-function addition a
    # name-set check alone would miss).
    counts = {name: len(statuses) for name, statuses in sites.items()}
    assert counts == {"invoke": 2, "resume": 1}, (
        f"runner RUNNING-restore callsite counts drifted: {counts} (expected "
        f"invoke=2, resume=1). A new control-mode update_status in invoke/resume "
        f"must be reviewed against the A4-0 feed-exclusion rationale."
    )
    for name, statuses in sites.items():
        assert set(statuses) <= {"RUNNING"}, (
            f"{name} writes a non-RUNNING control mode as a literal update_status; "
            f"route it through {RUNNER_EMITTER}."
        )


def test_runner_agent_loop_emitter_emits_mode_changed() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    emitter = _find_function(tree, RUNNER_EMITTER)
    assert emitter is not None, f"{RUNNER_EMITTER} not found in agent_task_runner.py"
    assert _emits_mode_changed(emitter)


def test_agent_service_has_no_variable_status_writes() -> None:
    # R4#P2: close the literal-scan blind spot. agent_service.py writes status
    # only via literal SessionStatus.X today; a NEW variable-status write would
    # evade the literal site guard, so fail until it is reviewed + (if control-mode)
    # wired to emit.
    tree = ast.parse(AGENT_SERVICE.read_text(encoding="utf-8"))
    funcs = _nonliteral_update_status_functions(tree)
    assert funcs == set(), (
        f"agent_service.py update_status calls with a NON-literal status arg in "
        f"{sorted(funcs)} are invisible to the A4-0 literal drift guard. Review the "
        f"new site: if it writes a control-mode status, wire a SessionModeChangedEvent "
        f"emit; then allowlist the function here."
    )


def test_runner_variable_status_writes_only_in_emitter() -> None:
    # R4#P2: the runner's ONLY legitimate variable-status write is the emitter.
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    funcs = _nonliteral_update_status_functions(tree)
    extra = funcs - RUNNER_VARIABLE_STATUS_ALLOWLIST
    assert not extra, (
        f"agent_task_runner.py has variable-status update_status calls outside the "
        f"allowlist: {sorted(extra)}. The only legitimate variable-status write is "
        f"{RUNNER_EMITTER} (which emits SessionModeChangedEvent). A new one must be "
        f"reviewed for A4-0 emission."
    )
