# api/tests/invariants/test_inv4_soft_session_status_audit.py
"""INV-4-hard SHIPPED — best-effort raw-SQL/ORM-attr defense behind the
AST gates.

This test scans for direct sessions.status writes (raw SQL / ORM attr
assignment) that bypass the SessionStateMachine / repository layer.

WHY: the hard INV-4 guarantee is enforced by the AST gates in
tests/invariants/test_inv4_ssm_single_writer.py (Gate A/B); this regex
scan is a best-effort, non-exhaustive second line of defense against NEW
direct-SQL / ORM-attr status writes outside the known DB layer.

WHAT: grep for raw SQL `UPDATE sessions SET status` or ORM attribute
assignment `sessions.status =`. Does NOT flag repository method calls
like `repo.update_status(...)` — Gate A (the AST gate) owns that check
(non-SSM callers of the repo mutators are banned hard there).

SKIP: db_session_repository.py (the one file allowed to do direct writes)
      SSM internals (domain/services/session/)

FAIL CONDITION: any direct-SQL / ORM-attr violation triggers a hard test
failure so new direct writes are caught immediately.
"""

import re
from pathlib import Path

from tests.invariants._whitelists import (
    API_APP, REPO_ROOT,
)

# Only flag direct SQL/ORM-level status writes — NOT repository method calls.
# update_status() / update_to_terminal() are legitimate repo interface methods;
# the dangerous pattern is bypassing the repo entirely.
_PATTERNS = [
    re.compile(r"sessions\.status\s*="),          # ORM attr assignment
    re.compile(r"UPDATE\s+sessions\s+SET\s+status\b", re.IGNORECASE),  # raw SQL
]

# Files that are allowed to perform direct sessions.status writes
# (repository implementation layer — the one blessed exit point for SQL).
# Also includes the domain ABC (session_repository.py) which has a docstring
# mentioning "UPDATE sessions SET status" in the transition_status method doc.
_EXEMPT_FILES = {
    "api/app/infrastructure/repositories/db_session_repository.py",
    "api/app/domain/repositories/session_repository.py",  # docstring only, no real SQL
}


def test_no_new_direct_session_status_writes():
    """No direct sessions.status SQL/ORM writes outside the DB repo layer.

    Best-effort defense: if a file performs a direct raw-SQL / ORM-attr
    status write outside the blessed repo layer, the build fails. This is
    NOT exhaustive — the hard INV-4 guarantee is the AST gates (Gate A/B
    in test_inv4_ssm_single_writer.py).
    """
    violations: list[str] = []
    for path in sorted(API_APP.rglob("*.py")):
        if "/__pycache__/" in str(path):
            continue
        rel = str(path.relative_to(REPO_ROOT))
        # Skip the canonical write location (db_session_repository.py)
        if rel in _EXEMPT_FILES:
            continue
        # Skip SSM internals — they are allowed to call the repo methods
        if rel.startswith("api/app/domain/services/session/"):
            continue
        src = path.read_text()
        for pat in _PATTERNS:
            for m in pat.finditer(src):
                ln = src[: m.start()].count("\n") + 1
                violations.append(f"{rel}:{ln} — {m.group(0).strip()}")

    # A4-1c: the soft known-offender whitelist was retired (it was vestigial
    # — zero real matches). This regex stays as a best-effort raw-SQL/ORM-attr
    # defense (NOT exhaustive; the hard guarantee is the INV-4 AST gates).
    new_violations = violations

    assert not new_violations, (
        "INV-4-hard (best-effort defense): NEW direct sessions.status writes "
        "detected outside the repository layer. Route status writes through "
        "`ssm.set_mode` / `ssm.terminate` (INV-4-hard, Gate A/B):\n"
        + "\n".join(new_violations)
    )
