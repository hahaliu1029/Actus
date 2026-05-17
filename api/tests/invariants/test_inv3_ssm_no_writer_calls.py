# api/tests/invariants/test_inv3_ssm_no_writer_calls.py
"""INV-3: SessionStateMachine subpackage MUST NOT call
ApprovalStateWriter mutators.

WHY: SSM owns sessions.status / mode_revision; it has no business
writing approval state.

WHITELIST: none — every file under domain/services/session/ is in-scope.
"""

import ast
from pathlib import Path

from tests.invariants._whitelists import (
    API_APP, INV3_WRITER_MUTATOR_NAMES, REPO_ROOT,
)

class _Visitor(ast.NodeVisitor):
    def __init__(self):
        self.hits: list[tuple[int, str]] = []

    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in INV3_WRITER_MUTATOR_NAMES:
            self.hits.append((node.lineno, f"...{f.attr}(...)"))
        self.generic_visit(node)

def test_ssm_does_not_call_writer_mutators():
    ssm_dir = API_APP / "domain" / "services" / "session"
    violations: list[str] = []
    for path in sorted(ssm_dir.rglob("*.py")):
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
        "INV-3 violation — SSM must not call writer mutators:\n"
        + "\n".join(violations)
    )
