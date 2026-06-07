# api/tests/invariants/test_inv4_ssm_single_writer.py
"""INV-4-hard (A4-1): the SSM subpackage is the SOLE CALLER of the three
session-status repo mutators (Gate A), and db_session_repository.py is the SOLE
module that issues a direct SQLAlchemy-Core `update(SessionModel).values(status=)`
/ `SessionModel.status = ...` / `setattr(SessionModel, "status", ...)` write
(Gate B).

Gate A mirrors INV-2/INV-3 (receiver-agnostic `f.attr in NAMES`, NO alias
resolver) — it catches `uow.session.X`, `self._uow.session.X`, `self._repo.X`,
bare `repo.X`, and `DBSessionRepository(...).X`. The three names are
session-repo-exclusive, so receiver-agnostic has zero false-positive surface.

Gate B is SessionModel-scoped + literal-`status=`-key so it never flags
`update(SessionModel).values(subagent_control_plane=...)`, `.values(**dict)`,
`SessionModel.status == ...` (where-clause), or `*.status = ...` on other models.

A best-effort raw-SQL/ORM-attr regex (test_inv4_soft_session_status_audit.py) is
retained as defense — NOT a hard guarantee (see spec §9 residual blind spots).
"""

import ast

from tests.invariants._whitelists import (
    API_APP,
    INV4_SESSION_STATUS_MUTATOR_NAMES,
    REPO_ROOT,
)

_EXEMPT_FILES = {
    "api/app/infrastructure/repositories/db_session_repository.py",
    "api/app/domain/repositories/session_repository.py",
}
_EXEMPT_PREFIX = "api/app/domain/services/session/"


def _is_exempt(rel: str) -> bool:
    return rel in _EXEMPT_FILES or rel.startswith(_EXEMPT_PREFIX)


# ───────────────────────────── Gate A ──────────────────────────────────── #
class _GateAVisitor(ast.NodeVisitor):
    """Flag any call `<anything>.update_status/update_to_terminal/transition_status(...)`."""

    def __init__(self) -> None:
        self.hits: list[tuple[int, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in INV4_SESSION_STATUS_MUTATOR_NAMES:
            self.hits.append((node.lineno, f"...{f.attr}(...)"))
        self.generic_visit(node)


# ───────────────────────────── Gate B ──────────────────────────────────── #
def _chain_reaches_update_sessionmodel(node: ast.AST) -> bool:
    """True iff the receiver chain of a `.values(...)` call traces back to
    `update(SessionModel)` (e.g. `update(SessionModel).where(...).values(...)`)."""
    cur = node
    while isinstance(cur, ast.Call):
        f = cur.func
        if isinstance(f, ast.Name) and f.id == "update":
            return bool(
                cur.args
                and isinstance(cur.args[0], ast.Name)
                and cur.args[0].id == "SessionModel"
            )
        if isinstance(f, ast.Attribute):
            cur = f.value  # walk down .where(...).values(...) etc.
            continue
        return False
    return False


def _values_has_literal_status_key(node: ast.Call) -> bool:
    for kw in node.keywords:
        # keyword `status=...`
        if kw.arg == "status":
            return True
        # `**{"status": ...}` STATIC-dict unpack (kw.arg is None). A static dict
        # literal is catchable; dynamic `**payload` (a Name) stays a documented
        # residual (see test_gate_b_known_residuals_documented).
        if kw.arg is None and isinstance(kw.value, ast.Dict):
            for k in kw.value.keys:
                if isinstance(k, ast.Constant) and k.value == "status":
                    return True
    # positional dict literal with a constant "status" key
    for arg in node.args:
        if isinstance(arg, ast.Dict):
            for k in arg.keys:
                if isinstance(k, ast.Constant) and k.value == "status":
                    return True
    return False


def _is_sessionmodel_status_attr(target: ast.AST) -> bool:
    return (
        isinstance(target, ast.Attribute)
        and target.attr == "status"
        and isinstance(target.value, ast.Name)
        and target.value.id == "SessionModel"
    )


class _GateBVisitor(ast.NodeVisitor):
    """Flag direct SessionModel.status writes outside the exempt repo."""

    def __init__(self) -> None:
        self.hits: list[tuple[int, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        # form (i): update(SessionModel)[...].values(status=<literal>)
        if (
            isinstance(f, ast.Attribute)
            and f.attr == "values"
            and _chain_reaches_update_sessionmodel(node)
            and _values_has_literal_status_key(node)
        ):
            self.hits.append((node.lineno, "update(SessionModel).values(status=...)"))
        # form (iii): setattr(SessionModel, "status", ...)
        if (
            isinstance(f, ast.Name)
            and f.id == "setattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "status"
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "SessionModel"
        ):
            self.hits.append((node.lineno, 'setattr(SessionModel, "status", ...)'))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        # form (ii): SessionModel.status = ...  (assignment TARGET only)
        for t in node.targets:
            if _is_sessionmodel_status_attr(t):
                self.hits.append((node.lineno, "SessionModel.status = ..."))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if _is_sessionmodel_status_attr(node.target):
            self.hits.append((node.lineno, "SessionModel.status: ... = ..."))
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if _is_sessionmodel_status_attr(node.target):
            self.hits.append((node.lineno, "SessionModel.status += ..."))
        self.generic_visit(node)


def _scan(visitor_cls) -> list[str]:
    violations: list[str] = []
    for path in sorted(API_APP.rglob("*.py")):
        if "/__pycache__/" in str(path):
            continue
        rel = str(path.relative_to(REPO_ROOT))
        if _is_exempt(rel):
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        v = visitor_cls()
        v.visit(tree)
        for ln, snippet in v.hits:
            violations.append(f"{rel}:{ln} — {snippet}")
    return violations


# ───────────────────────────── live gates ──────────────────────────────── #
def test_inv4_gate_a_method_calls():
    """No non-exempt CALL to the three status mutators (INV-4-hard)."""
    violations = _scan(_GateAVisitor)
    assert not violations, (
        "INV-4-hard Gate A: status mutator called outside the SSM subpackage / "
        "exempt repo. Route through ssm.set_mode / ssm.terminate:\n"
        + "\n".join(violations)
    )


def test_inv4_gate_b_direct_core_writes():
    """No non-exempt direct SessionModel.status Core/ORM write (green day-one —
    all such writes live in the exempt repo)."""
    violations = _scan(_GateBVisitor)
    assert not violations, (
        "INV-4-hard Gate B: direct SessionModel.status write outside "
        "db_session_repository.py:\n" + "\n".join(violations)
    )


# ───────────────────────────── meta-tests ──────────────────────────────── #
def _hits(visitor_cls, src: str) -> list[str]:
    v = visitor_cls()
    v.visit(ast.parse(src))
    return [s for _, s in v.hits]


def test_gate_a_meta_positive():
    assert _hits(_GateAVisitor, "await self._repo.update_to_terminal(s, x, r)")
    assert _hits(_GateAVisitor, "await repo.update_status(s, x)")
    assert _hits(_GateAVisitor, "await uow.session.update_status(s, x)")
    assert _hits(_GateAVisitor, "await DBSessionRepository(db).transition_status(s)")


def test_gate_a_meta_negative():
    assert not _hits(_GateAVisitor, "await supervisor.terminate(s, u, r)")
    assert not _hits(_GateAVisitor, "await ssm.set_mode(s, x, r, session_repo=repo)")
    assert not _hits(_GateAVisitor, "await ssm.terminate(s, x, r, session_repo=repo)")
    assert not _hits(_GateAVisitor, "await repo.update_title(s, t)")


def test_gate_b_meta_positive():
    assert _hits(_GateBVisitor, "update(SessionModel).values(status=X.value)")
    assert _hits(
        _GateBVisitor,
        "update(SessionModel).where(SessionModel.id == s).values(status='running')",
    )
    assert _hits(_GateBVisitor, "SessionModel.status = 'running'")
    assert _hits(_GateBVisitor, 'setattr(SessionModel, "status", x)')
    assert _hits(_GateBVisitor, "update(SessionModel).values({'status': 'running'})")
    # static-dict unpack `**{...}` (kw.arg is None but value is a Dict literal)
    assert _hits(_GateBVisitor, "update(SessionModel).values(**{'status': 'running'})")


def test_gate_b_meta_negative():
    assert not _hits(_GateBVisitor, "update(SessionModel).values(subagent_control_plane='legacy')")
    assert not _hits(_GateBVisitor, "stmt.where(SessionModel.status == 'finishing')")
    assert not _hits(_GateBVisitor, "user.status = 'active'")
    assert not _hits(_GateBVisitor, "row.status = 'active'")
    assert not _hits(_GateBVisitor, 'setattr(user, "status", x)')
    assert not _hits(_GateBVisitor, "update(OtherModel).values(status='x')")
    assert not _hits(_GateBVisitor, "update(SessionModel).values(**dynamic_dict)")


def test_gate_b_known_residuals_documented():
    """DOCUMENTED residual blind spots (spec §9). Gate B deliberately uses NO
    alias/data-flow resolver (mirrors INV-2/INV-3, not CS4), so these slip it.
    Gate A + the regex are the backstops; this test pins the limitation so a
    future reader does not assume Gate B is exhaustive.
    """
    # (a) variable-broken chain: .values() called on a local `stmt`, not on
    #     update(SessionModel) directly. This exact style exists in
    #     subagent_research_service.py:323/328 (for subagent_control_plane).
    broken_chain = "stmt = update(SessionModel)\nstmt = stmt.values(status='running')"
    assert not _hits(_GateBVisitor, broken_chain)
    # (b) dynamic setattr with a VARIABLE field name (the legacy save() backdoor
    #     in SessionModel.update_from_domain) — closed structurally by Task 2's
    #     neutralization + behavioral test, NOT by Gate B.
    assert not _hits(_GateBVisitor, "setattr(self, field, value)")
