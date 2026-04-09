"""Test that the input loop break condition is correct."""
import pytest


def test_break_condition_in_source_code():
    """P0-1: Verify the loop break condition uses `is_empty()` without negation.

    The bug: `if not await task.input_stream.is_empty(): break`
    breaks when queue is NOT empty (has messages), discarding them.
    The fix: `if await task.input_stream.is_empty(): break`
    breaks when queue IS empty (all messages processed).
    """
    import inspect
    from app.domain.services.agent_task_runner import AgentTaskRunner
    source = inspect.getsource(AgentTaskRunner.invoke)

    lines = source.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "is_empty()" in stripped and i + 1 < len(lines) and "break" in lines[i + 1].strip():
            assert "not" not in stripped, (
                f"Break condition still inverted: '{stripped}'. "
                f"Should be `if await task.input_stream.is_empty():` without `not`."
            )
            return


def test_session_delete_branch_exists_in_cancel_handler():
    """P0-3 structural: CancelledError handler must mention session_delete."""
    import inspect
    from app.domain.services.agent_task_runner import AgentTaskRunner
    source = inspect.getsource(AgentTaskRunner.invoke)
    assert "session_delete" in source, \
        "invoke() CancelledError handler must check for session_delete cancel reason"


def test_session_delete_branch_does_not_write_done_or_completed():
    """P0-3 structural guard: session_delete branch must raise before writing DoneEvent/COMPLETED.

    Note: This is a source-level structural guard, not a behavioral test.
    It verifies the current code shape but cannot prevent future indirect side effects.
    Full behavioral coverage for session_delete requires integration testing with DB/Redis (manual checklist in Task 12).
    """
    import inspect
    from app.domain.services.agent_task_runner import AgentTaskRunner
    source = inspect.getsource(AgentTaskRunner.invoke)

    cancel_idx = source.index("CancelledError")
    cancel_block = source[cancel_idx:]
    delete_idx = cancel_block.index("session_delete")
    after_delete = cancel_block[delete_idx:]

    raise_pos = after_delete.index("raise")
    has_put_before_raise = "_put_and_add_event" in after_delete[:raise_pos]
    has_status_before_raise = "update_status" in after_delete[:raise_pos]

    assert not has_put_before_raise, \
        "session_delete must raise BEFORE any _put_and_add_event call"
    assert not has_status_before_raise, \
        "session_delete must raise BEFORE any update_status call"
