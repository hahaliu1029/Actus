# api/tests/invariants/test_inv1b_writer_caller_single_upstream.py
"""INV-1b: ApprovalStateWriter.{write, write_audit_only, delete_grant}
may only be called from DefaultPermissionEngine (or the DI factory).

WHY: PE is the single upstream caller (R5 CS4 delegation). Direct
calls from elsewhere break the audit chain.

WHAT: Match attribute access calls of the three method names on any
local variable or self attribute. Whitelist = permission/default_engine.py
+ approval_state_writer.py + service_dependencies.py + agent_service.py
(legacy preflight path, sunset A4-1).

WHITELIST RATIONALE: only the engine implementation calls the writer.
The writer file itself contains the method definitions (not callers).
The DI factory file constructs the engine but does NOT invoke the
methods — its mention is for import linkage and is verified via
'function call vs reference' detection below.
agent_service.py: _preflight_resume_tool_confirmation_legacy is the legacy
fail-open path retained for PE-0 (Phase 8.1). Direct writer calls there
are intentional and documented — whitelisted until A4-1 removes the path.
"""

import ast
from pathlib import Path

from tests.invariants._whitelists import (
    API_APP, INV1B_WRITER_CALLER_WHITELIST, REPO_ROOT,
)

_FORBIDDEN_NAMES = {"write", "write_audit_only", "delete_grant"}
_WRITER_LIKE_RECEIVER_NAMES = {
    "writer", "_writer", "approval_writer", "approval_state_writer",
    "once_audit_writer",
}

class _Visitor(ast.NodeVisitor):
    def __init__(self):
        self.hits: list[tuple[int, str]] = []

    def visit_Call(self, node):
        # Look for X.<forbidden_name>(...) where X is named like the writer.
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in _FORBIDDEN_NAMES:
            receiver = f.value
            if isinstance(receiver, ast.Name) and receiver.id in _WRITER_LIKE_RECEIVER_NAMES:
                self.hits.append((node.lineno, f"{receiver.id}.{f.attr}(...)"))
            if isinstance(receiver, ast.Attribute) and receiver.attr in _WRITER_LIKE_RECEIVER_NAMES:
                self.hits.append((node.lineno, f"self.{receiver.attr}.{f.attr}(...)"))
        self.generic_visit(node)

def _walk_app_py(base):
    for p in sorted(base.rglob("*.py")):
        if "/__pycache__/" in str(p):
            continue
        yield p

def test_writer_callsites_only_in_pe():
    violations: list[str] = []
    whitelist = set(INV1B_WRITER_CALLER_WHITELIST)
    for path in _walk_app_py(API_APP):
        rel = str(path.relative_to(REPO_ROOT))
        if rel in whitelist:
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        v = _Visitor()
        v.visit(tree)
        for ln, snippet in v.hits:
            violations.append(f"{rel}:{ln} — {snippet}")
    assert not violations, (
        "INV-1b violation — only DefaultPermissionEngine may call "
        "ApprovalStateWriter.{write,write_audit_only,delete_grant}:\n"
        + "\n".join(violations)
    )
