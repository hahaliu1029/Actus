"""C3 spec §13.3 + R3 P2.2 — AST CI gate.

Non-whitelisted callers of ``.destroy()`` / ``.suspend()`` in
``api/app/application/`` must reference ``_should_skip_mailbox_lifecycle``
somewhere in the same function body (or have the module-level import).
The helper is the single point of truth for the M1 single-writer invariant
(spec §11.4) — any new caller of suspend/destroy that bypasses it could
race the MailboxSupervisor on a mailbox-plane session.

This is a syntactic check, not a control-flow proof; runtime safety is
provided by the helper + caller pattern together. A reviewer auditing a
flagged site can grep for the helper reference and verify the surrounding
``if`` branch.

**Whitelisted files** (own the destroy/suspend path or hold a different
contract):

- ``sandbox_lifecycle_service.py`` — single-writer for ``SandboxBinding``
  state transitions (I3); the helper protects callers of *this* service,
  not the service itself.
- ``mailbox_supervisor.py`` — owns mailbox-plane destroy (M1).
- ``supervisor_registry.py`` — supervisor lifecycle which transitively
  calls destroy through MailboxSupervisor.
- ``mailbox_skip_helper.py`` — the helper module itself.
- ``skill_creator_service.py`` — uses a *standalone* ``Sandbox`` external
  port for transient skill generation; not bound to a Session row, so
  the mailbox-plane invariant does not apply.

**Whitelisted functions**:

- ``reconcile_orphans`` — explicitly whitelisted per spec §13.3.
- ``sweep_terminal_coordinator_active_sandboxes`` — startup leak-sweep
  (C2 coordinator-cancel Part B) for the orphaned ACTIVE sandbox of
  *terminal* (COMPLETED/TIMED_OUT) mailbox-plane coordinator children
  whose root ``MailboxSupervisor`` was already killed on user-stop
  (before consuming ``CANCEL_ACK``). M1 protects *in-flight* bindings;
  these children are terminal with no live supervisor to race, so the
  restart-bounded destroy is safe (``destroy()``'s per-session lock
  serializes any pathological respawn race idempotently). Consulting the
  helper would be wrong here: the sweep's query selects only ``subagent``
  + ``mailbox``-plane rows — exactly what ``_should_skip_mailbox_lifecycle``
  returns ``True`` for — so it would skip every row and re-leak the
  containers it exists to reap.
- ``delete_session`` — user-initiated permanent session deletion: the row
  is going away, so M1 (which protects in-flight bindings) does not
  apply; the supervisor for a deleted root is stopped upstream of this
  call via ``_maybe_stop_supervisor_for_session``.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Tuple, Union


_WHITELIST_FILES = {
    "api/app/application/services/sandbox_lifecycle_service.py",
    "api/app/application/services/mailbox_supervisor.py",
    "api/app/application/services/supervisor_registry.py",
    "api/app/application/services/skill_creator_service.py",
    "api/app/domain/services/mailbox_skip_helper.py",
}

_WHITELIST_FUNCTION_NAMES = {
    "reconcile_orphans",
    "sweep_terminal_coordinator_active_sandboxes",
    "delete_session",
    # ``AgentService.shutdown`` calls ``self._task_cls.destroy()`` — the
    # Task class-level teardown, NOT the SandboxLifecycleService destroy.
    # The AST check is name-only and cannot disambiguate; whitelisting the
    # function is safer than weakening the matcher.
    "shutdown",
}


FuncDef = Union[ast.FunctionDef, ast.AsyncFunctionDef]


def _find_call_sites(tree: ast.AST) -> list[Tuple[ast.Call, FuncDef]]:
    """Yield ``(call_node, enclosing_func)`` for every ``.destroy(`` /
    ``.suspend(`` attribute call inside a function body."""
    out: list[Tuple[ast.Call, FuncDef]] = []

    class _V(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[FuncDef] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            self.stack.append(node)
            self.generic_visit(node)
            self.stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
            self.stack.append(node)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in ("destroy", "suspend")
                and self.stack
            ):
                out.append((node, self.stack[-1]))
            self.generic_visit(node)

    _V().visit(tree)
    return out


def _references_skip_helper(func: FuncDef) -> bool:
    """``True`` if the function body mentions ``_should_skip_mailbox_lifecycle``
    in any form (call, attribute, name, or import).

    Covers ``from … import _should_skip_mailbox_lifecycle`` inside a
    function body as well as a module-level import propagated via
    ``ast.Name`` references.
    """
    for node in ast.walk(func):
        if isinstance(node, ast.Name) and node.id == "_should_skip_mailbox_lifecycle":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "_should_skip_mailbox_lifecycle":
            return True
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "_should_skip_mailbox_lifecycle":
                    return True
    return False


def test_application_layer_callers_have_mailbox_skip_helper_reference() -> None:
    """codex r1 [R1-2, HIGH TEST] — earlier rounds let a module-level
    ``from … import _should_skip_mailbox_lifecycle`` satisfy the gate for
    every function in the file. That hid the real
    ``SubagentResearchService.run_research`` violation (one suspend in the
    finally block, no helper reference inside the function body, but the
    module imported the helper for OTHER sites).

    The check is now strictly per-function-body: each
    ``.destroy()`` / ``.suspend()`` call must have a
    ``_should_skip_mailbox_lifecycle`` reference inside the SAME enclosing
    function, or the function must be in ``_WHITELIST_FUNCTION_NAMES``,
    or the file must be in ``_WHITELIST_FILES``.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    app_dir = repo_root / "api" / "app" / "application"
    assert app_dir.is_dir(), f"expected {app_dir} to exist"

    violations: list[str] = []
    for py_file in app_dir.rglob("*.py"):
        rel = py_file.relative_to(repo_root).as_posix()
        if rel in _WHITELIST_FILES:
            continue
        source = py_file.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError as exc:  # pragma: no cover — defensive
            violations.append(f"{rel}: failed to parse ({exc!s})")
            continue

        for call, func in _find_call_sites(tree):
            if func.name in _WHITELIST_FUNCTION_NAMES:
                continue
            if _references_skip_helper(func):
                continue
            attr = call.func.attr if isinstance(call.func, ast.Attribute) else "?"
            violations.append(
                f"{rel}:{call.lineno} — `.{attr}(...)` in function "
                f"`{func.name}` without `_should_skip_mailbox_lifecycle` "
                f"reference in the SAME function body."
            )

    assert not violations, (
        "AST gate violations (C3 spec §13.3 — every application/ suspend/destroy "
        "site must reference _should_skip_mailbox_lifecycle in its own function "
        "body, or the function/file must be whitelisted):\n"
        + "\n".join(violations)
    )


def test_module_level_import_alone_is_insufficient(tmp_path) -> None:
    """codex r1 [R1-2, HIGH TEST] regression — the gate must reject a file
    whose ``.suspend()`` call is in a function that doesn't itself reference
    the helper, even when the module-top imports the helper for unrelated
    sites. Synthesize the violator pattern as a temp file fed to the same
    AST helpers used by the production gate.
    """
    bad = tmp_path / "fake_module.py"
    bad.write_text(
        "from app.domain.services.mailbox_skip_helper import _should_skip_mailbox_lifecycle\n"
        "\n"
        "class Svc:\n"
        "    async def some_method(self, sandbox):\n"
        "        await sandbox.suspend('s')  # unguarded\n"
        "\n"
        "    async def other_method(self, session, sandbox):\n"
        "        if _should_skip_mailbox_lifecycle(session):\n"
        "            return\n"
        "        await sandbox.suspend('s')  # guarded\n",
        encoding="utf-8",
    )

    tree = ast.parse(bad.read_text(encoding="utf-8"), filename=str(bad))
    flagged: list[str] = []
    for call, func in _find_call_sites(tree):
        if not _references_skip_helper(func):
            flagged.append(f"{func.name}:{call.lineno}")

    # ``some_method`` must be flagged; ``other_method`` must not.
    assert any(v.startswith("some_method") for v in flagged), (
        f"expected some_method to be flagged; got {flagged}"
    )
    assert not any(v.startswith("other_method") for v in flagged), (
        f"other_method has the helper reference and must NOT be flagged; got {flagged}"
    )
