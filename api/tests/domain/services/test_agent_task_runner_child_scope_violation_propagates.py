"""C2 PR-2 §5.4 — ChildScopeViolation must propagate past AgentTaskRunner's outer catch-all.

Without this, PR-4's CoordinatorChildRunner finalizer cannot distinguish typed
scope violations from generic runner crashes; the violation gets converted to
ErrorEvent + COMPLETED(FAILED) which loses the typed decision payload.
"""
import inspect
import re


def test_agent_task_runner_reraises_child_scope_violation():
    """AST-grep: AgentTaskRunner.run loop must `raise` for ChildScopeViolation BEFORE the broad `except Exception`."""
    from app.domain.services import agent_task_runner

    src = inspect.getsource(agent_task_runner)
    # The relevant catch-all is identified by the Chinese log message.
    crash_idx = src.find("AgentTaskRunner出错")
    assert crash_idx != -1, "AgentTaskRunner outer catch-all marker not found"
    preceding = src[:crash_idx]
    last_reraise = preceding.rfind("except ChildScopeViolation")
    assert last_reraise != -1, (
        "ChildScopeViolation must be re-raised BEFORE the broad runner catch-all; "
        "otherwise PR-4 finalizer cannot see the typed scope deny."
    )
    block_tail = src[last_reraise: last_reraise + 600]
    assert re.search(
        r"except ChildScopeViolation[^:]*:\s*(?:#[^\n]*\n\s*)*raise\b",
        block_tail,
    ), (
        f"ChildScopeViolation except block must `raise`; got: {block_tail!r}"
    )


def test_child_scope_violation_imported_in_agent_task_runner():
    """agent_task_runner must have ChildScopeViolation bound at module scope."""
    from app.domain.services import agent_task_runner
    from app.domain.services.permission.child_scope_violation import ChildScopeViolation

    assert hasattr(agent_task_runner, "ChildScopeViolation"), (
        "agent_task_runner must bind ChildScopeViolation at module scope"
    )
    assert agent_task_runner.ChildScopeViolation is ChildScopeViolation, (
        "agent_task_runner.ChildScopeViolation must reference the canonical class"
    )
