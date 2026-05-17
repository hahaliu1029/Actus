# api/tests/invariants/test_inv2_pe_no_session_writes.py
"""INV-2: PermissionEngine subpackage MUST NOT call SessionStateMachine
mutators (request_takeover / release_takeover / enter_finishing /
complete / transition). PE READS mode via get_mode_with_revision only.

WHY: keeps PE pure (no session-lifecycle side effects). SSM is the only
mutator authority.

WHITELIST: none — every file under permission/ is in-scope."""

import ast
from pathlib import Path

from tests.invariants._whitelists import (
    API_APP, INV2_SSM_MUTATOR_NAMES, REPO_ROOT,
)

class _Visitor(ast.NodeVisitor):
    def __init__(self):
        self.hits: list[tuple[int, str]] = []

    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in INV2_SSM_MUTATOR_NAMES:
            self.hits.append((node.lineno, f"...{f.attr}(...)"))
        self.generic_visit(node)

def test_pe_does_not_call_ssm_mutators():
    pe_dir = API_APP / "domain" / "services" / "permission"
    violations: list[str] = []
    for path in sorted(pe_dir.rglob("*.py")):
        if "/__pycache__/" in str(path):
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        v = _Visitor()
        v.visit(tree)
        for ln, snippet in v.hits:
            rel = str(path.relative_to(REPO_ROOT))
            violations.append(f"{rel}:{ln} — {snippet}")
    assert not violations, (
        "INV-2 violation — PE may not call SSM mutators:\n"
        + "\n".join(violations)
    )
