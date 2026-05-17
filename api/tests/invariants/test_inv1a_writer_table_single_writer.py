# api/tests/invariants/test_inv1a_writer_table_single_writer.py
"""INV-1a: only ApprovalStateWriter may write approval_grants /
tool_approval_logs.

WHY: R5 CS4 single-writer contract — any other writer breaks audit log
correctness + grant lifecycle invariants.

WHAT: AST scan all .py files in api/app. Flag any:
  - session.add(ToolApprovalGrantModel(...))
  - session.execute(insert(ToolApprovalGrantModel))
  - session.execute(update(ToolApprovalGrantModel))
  - session.execute(delete(ToolApprovalGrantModel))
  - bulk_save_objects([ToolApprovalGrantModel(...)])
  - raw SQL strings containing 'approval_grants' or 'tool_approval_logs'
    as a write (INSERT/UPDATE/DELETE)

WHITELIST RATIONALE: only db_approval_grant_repository.py +
db_tool_approval_log_repository.py are the legitimate ORM-write
modules (the ApprovalStateWriter wraps both, one per table). Any
other file matching these patterns is a violation.
"""

import ast
from pathlib import Path

from tests.invariants._whitelists import (
    API_APP, INV1A_TABLE_WRITER_WHITELIST, REPO_ROOT,
)

_TARGET_ORMS = {"ToolApprovalGrantModel", "ToolApprovalLogModel"}
_TABLE_NAMES = {"approval_grants", "tool_approval_logs"}
_WRITE_VERBS = ("insert", "update", "delete", "bulk_save_objects")

class _Visitor(ast.NodeVisitor):
    def __init__(self, source: str, path: Path):
        self.source = source
        self.path = path
        self.hits: list[tuple[int, str]] = []

    def visit_Call(self, node):
        # session.add(<TargetOrm>(...))
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add":
            for arg in node.args:
                if (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name)
                        and arg.func.id in _TARGET_ORMS):
                    self.hits.append((node.lineno, f"session.add({arg.func.id}(...))"))
        # session.execute(<verb>(<TargetOrm>...))
        if isinstance(node.func, ast.Attribute) and node.func.attr == "execute":
            if node.args:
                inner = node.args[0]
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                    if inner.func.id in _WRITE_VERBS:
                        # Inspect args for ORM target
                        for a in inner.args:
                            if isinstance(a, ast.Name) and a.id in _TARGET_ORMS:
                                self.hits.append(
                                    (node.lineno, f"execute({inner.func.id}({a.id}))")
                                )
        # bulk_save_objects([TargetOrm(...)])
        if isinstance(node.func, ast.Attribute) and node.func.attr == "bulk_save_objects":
            for a in node.args:
                if isinstance(a, (ast.List, ast.Tuple)):
                    for el in a.elts:
                        if (isinstance(el, ast.Call) and isinstance(el.func, ast.Name)
                                and el.func.id in _TARGET_ORMS):
                            self.hits.append(
                                (node.lineno, f"bulk_save_objects([{el.func.id}(...)])")
                            )
        self.generic_visit(node)

def _walk_app_py(base: Path):
    for p in sorted(base.rglob("*.py")):
        if "/__pycache__/" in str(p):
            continue
        yield p

def test_no_unauthorized_orm_writes():
    violations: list[str] = []
    whitelist = set(INV1A_TABLE_WRITER_WHITELIST)
    for path in _walk_app_py(API_APP):
        rel = str(path.relative_to(REPO_ROOT))
        if rel in whitelist:
            continue
        src = path.read_text()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        visitor = _Visitor(src, path)
        visitor.visit(tree)
        for ln, snippet in visitor.hits:
            violations.append(f"{rel}:{ln} — {snippet}")
    assert not violations, (
        "INV-1a violation — only db_approval_grant_repository.py + "
        "db_tool_approval_log_repository.py may write "
        "approval_grants / tool_approval_logs:\n" + "\n".join(violations)
    )

def test_no_raw_sql_writes_to_target_tables():
    """Semantic grep for raw SQL text that mutates the two tables."""
    import re

    pattern = re.compile(
        r"(INSERT\s+INTO|UPDATE|DELETE\s+FROM)\s+(approval_grants|tool_approval_logs)\b",
        re.IGNORECASE,
    )
    violations: list[str] = []
    whitelist = set(INV1A_TABLE_WRITER_WHITELIST)
    for path in _walk_app_py(API_APP):
        rel = str(path.relative_to(REPO_ROOT))
        if rel in whitelist:
            continue
        src = path.read_text()
        for m in pattern.finditer(src):
            ln = src[: m.start()].count("\n") + 1
            violations.append(f"{rel}:{ln} — {m.group(0)}")
    assert not violations, (
        "INV-1a violation — raw SQL writes to approval tables outside writer:\n"
        + "\n".join(violations)
    )
