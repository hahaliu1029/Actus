"""C2 PR-2 §5.4 — ChildScopeViolation must propagate past AgentTaskRunner's outer catch-all.

Without this, PR-4's CoordinatorChildRunner finalizer cannot distinguish typed
scope violations from generic runner crashes; the violation gets converted to
ErrorEvent + COMPLETED(FAILED) which loses the typed decision payload.
"""
import inspect
import re


def test_agent_task_runner_reraises_child_scope_violation():
    """AST-grep: AgentTaskRunner.run loop must stash the typed violation on the
    task and then `raise` it, BEFORE the broad `except Exception`."""
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
    # Bound the block to THIS except arm (up to the next except at the same
    # 12-space indent) instead of a fragile fixed-size window — the arm grew
    # when the C2b §6-lim2 best-effort terminal-status write landed between the
    # stash and the re-raise, which would push `raise` past a fixed window.
    after = src[last_reraise:]
    next_arm = after.find("\n            except ", 1)
    block = after if next_arm == -1 else after[:next_arm]
    stash_i = block.find("set_child_scope_violation")
    assert stash_i != -1, (
        "ChildScopeViolation except block must stash the typed violation on the "
        "task (set_child_scope_violation) before re-raising — else RedisStreamTask "
        "swallows it and the adapter returns generic FAILED, not NEEDS_AUTHORIZATION."
    )
    # Match a `raise` STATEMENT (line-anchored) AFTER the stash — not a
    # "raise"/"re-raise" substring that may appear in a comment between the
    # stash and the real re-raise.
    assert re.search(r"\n\s+raise\b", block[stash_i:]), (
        "ChildScopeViolation except block must `raise` AFTER stashing the violation."
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
