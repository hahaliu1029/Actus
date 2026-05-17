# api/tests/invariants/test_inv4_soft_session_status_audit.py
"""INV-4-soft: report (but NOT fail) every direct sessions.status write
that bypasses the SessionStateMachine / repository layer.

WHY: PE-0 keeps existing update_status / update_to_terminal in
db_session_repository.py to avoid scope creep; A4-1 makes this hard.
Soft gate ensures no NEW direct-SQL / ORM-attr status writes are added
outside the known DB layer.

WHAT: grep for raw SQL `UPDATE sessions SET status` or ORM attribute
assignment `sessions.status =`. Does NOT flag repository method calls
like `repo.update_status(...)` — those are the correct call pattern.

SKIP: db_session_repository.py (the one file allowed to do direct writes)
      SSM internals (domain/services/session/)

FAIL CONDITION: any violation NOT in the known-offender list triggers
a hard test failure so new direct writes are caught immediately.
"""

import re
from pathlib import Path

from tests.invariants._whitelists import (
    API_APP, INV4_SOFT_KNOWN_OFFENDERS, REPO_ROOT,
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

    This is a hard gate: if a file not in the known-offenders list performs
    a direct write, the build fails. Existing offenders are tracked in
    _whitelists.INV4_SOFT_KNOWN_OFFENDERS with A4-1 sunset references.
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

    # Build the known offender set (path:line format)
    known_keys = {entry for entry, _ in INV4_SOFT_KNOWN_OFFENDERS}

    # Remove known offenders from violations
    new_violations = [v for v in violations if v.split(" —")[0] not in known_keys]

    assert not new_violations, (
        "INV-4-soft: NEW direct sessions.status writes detected outside the "
        "repository layer. Route writes through SessionRepository.update_status() "
        "or SessionStateMachine.transition_status(), or add to "
        "_whitelists.INV4_SOFT_KNOWN_OFFENDERS with an A4-1 sunset ref:\n"
        + "\n".join(new_violations)
    )
