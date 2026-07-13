"""Behavioral tests for _run_postprocess_or_cancel and _do_postprocess.

These tests construct a minimal AgentTaskRunner with mocked dependencies
and exercise the real methods to verify runtime behavior, not just source structure.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.domain.models.event import FinishingEvent, DoneEvent, ErrorEvent, MessageEvent
from app.domain.models.session import SessionStatus


def _make_runner_with_mocks():
    """Construct a minimal AgentTaskRunner with only the deps needed for postprocess tests.

    Since AgentTaskRunner.__init__ has many params, we patch it to skip init
    and set only the attributes _do_postprocess/_run_postprocess_or_cancel need.
    """
    from app.domain.services.agent_task_runner import AgentTaskRunner

    runner = object.__new__(AgentTaskRunner)
    # _do_postprocess needs: self._flow, self._memory_flusher, self._uow, self._session_id
    runner._flow = MagicMock()
    runner._flow._deferred_final_state = {"messages": [MagicMock()]}
    runner._flow._deferred_summaries = []
    runner._flow._persist_after_graph = AsyncMock()
    runner._flow._pending_flush_batch = None
    runner._flow.summary_llm = MagicMock()
    runner._memory_flusher = MagicMock()
    runner._memory_flusher.submit = MagicMock()
    # UoW mock for latest_message + unread updates
    runner._session_id = "test-session-id"
    mock_uow = MagicMock()
    mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
    mock_uow.__aexit__ = AsyncMock(return_value=False)
    mock_uow.session = MagicMock()
    mock_uow.session.update_latest_message = AsyncMock()
    mock_uow.session.increment_unread_message_count = AsyncMock()
    runner._uow = mock_uow
    # _put_and_add_event mock
    runner._events_log = []
    async def mock_put(task, event, persist=True):
        runner._events_log.append((event, persist))
    runner._put_and_add_event = mock_put
    return runner


def _make_mock_task(messages_in_queue=0):
    """Create a mock Task with controllable input_stream."""
    task = MagicMock()
    call_count = [0]
    async def is_empty():
        call_count[0] += 1
        return call_count[0] > messages_in_queue
    task.input_stream = MagicMock()
    task.input_stream.is_empty = is_empty
    return task


# --- _run_postprocess_or_cancel behavioral tests ---

@pytest.mark.anyio
async def test_postprocess_completes_normally_returns_false():
    """When postprocess finishes and no new messages, returns False."""
    runner = _make_runner_with_mocks()
    task = _make_mock_task(messages_in_queue=0)

    # Patch _do_postprocess to complete instantly
    runner._do_postprocess = AsyncMock()

    result = await runner._run_postprocess_or_cancel(task)
    assert result is False, "Should return False when postprocess completes without interruption"
    runner._do_postprocess.assert_awaited_once()


@pytest.mark.anyio
async def test_postprocess_cancelled_on_new_message_returns_true():
    """When input_stream has a message during postprocess, cancel and return True."""
    runner = _make_runner_with_mocks()
    # input_stream returns not-empty immediately
    task = MagicMock()
    task.input_stream = MagicMock()
    task.input_stream.is_empty = AsyncMock(return_value=False)

    # _do_postprocess hangs until cancelled
    async def hang_forever(task):
        await asyncio.sleep(999)
    runner._do_postprocess = hang_forever

    result = await runner._run_postprocess_or_cancel(task)
    assert result is True, "Should return True when new message cancels postprocess"


@pytest.mark.anyio
async def test_drain_check_catches_late_message():
    """After postprocess completes, drain check detects a late-arriving message."""
    runner = _make_runner_with_mocks()
    runner._do_postprocess = AsyncMock()  # completes instantly

    # is_empty: True during polling (no early cancel), then False at drain check
    call_count = [0]
    async def is_empty_with_late_message():
        call_count[0] += 1
        if call_count[0] <= 1:
            return True  # polling: empty
        return False  # drain check: message arrived
    task = MagicMock()
    task.input_stream = MagicMock()
    task.input_stream.is_empty = is_empty_with_late_message

    result = await runner._run_postprocess_or_cancel(task)
    assert result is True, "Drain check must catch late message and return True"


@pytest.mark.anyio
async def test_postprocess_exception_propagates():
    """_do_postprocess exception must propagate through _run_postprocess_or_cancel."""
    runner = _make_runner_with_mocks()
    task = _make_mock_task(messages_in_queue=0)

    async def failing_postprocess(t):
        raise RuntimeError("persist failed")
    runner._do_postprocess = failing_postprocess

    with pytest.raises(RuntimeError, match="persist failed"):
        await runner._run_postprocess_or_cancel(task)


# --- _do_postprocess behavioral tests ---

@pytest.mark.anyio
async def test_do_postprocess_calls_persist_then_flush_then_summary():
    """_do_postprocess must call persist, submit flush, then run summary in order."""
    runner = _make_runner_with_mocks()
    runner._flow._pending_flush_batch = MagicMock()
    task = MagicMock()

    call_order = []
    async def mock_persist(state, summaries):
        call_order.append("persist")
    runner._flow._persist_after_graph = mock_persist

    original_submit = runner._memory_flusher.submit
    def mock_submit(batch):
        call_order.append("flush")
    runner._memory_flusher.submit = mock_submit

    with patch(
        "app.domain.services.agent_task_runner.run_background_summary",
        new_callable=AsyncMock,
    ) as mock_summary:
        mock_summary.return_value = "summary text"
        async def side_effect(msgs, llm, on_event, lang="zh", **kwargs):
            # **kwargs tolerates the cost-handler ``callbacks=`` plumbing
            # added in B4 M0 — keeps these mocks robust to future signature
            # extensions on ``run_background_summary``.
            call_order.append("summary")
            return "summary text"
        mock_summary.side_effect = side_effect

        await runner._do_postprocess(task)

    assert call_order == ["persist", "flush", "summary"], \
        f"Expected [persist, flush, summary], got {call_order}"


@pytest.mark.anyio
async def test_do_postprocess_timeout_skips_all_llm_postprocessing():
    """The graph watchdog is the whole-run deadline.  After it fires, FINISHING
    may persist state but must not start conversation/background summary LLMs.
    """
    runner = _make_runner_with_mocks()
    runner._was_timed_out = True
    task = MagicMock()

    with patch(
        "app.domain.services.agent_task_runner.run_background_summary",
        new_callable=AsyncMock,
    ) as mock_summary:
        await runner._do_postprocess(task)

    runner._flow._persist_after_graph.assert_awaited_once_with(
        runner._flow._deferred_final_state,
        runner._flow._deferred_summaries,
        allow_llm=False,
    )
    mock_summary.assert_not_awaited()


@pytest.mark.anyio
async def test_do_postprocess_summary_failure_is_silent():
    """Phase 3 summary failure must not propagate — only logged."""
    runner = _make_runner_with_mocks()
    task = MagicMock()

    with patch(
        "app.domain.services.agent_task_runner.run_background_summary",
        new_callable=AsyncMock,
        side_effect=RuntimeError("LLM down"),
    ):
        # Should NOT raise
        await runner._do_postprocess(task)


@pytest.mark.anyio
async def test_do_postprocess_persist_failure_propagates():
    """Phase 1 persist failure must propagate to caller."""
    runner = _make_runner_with_mocks()
    runner._flow._persist_after_graph = AsyncMock(side_effect=RuntimeError("DB down"))
    task = MagicMock()

    with pytest.raises(RuntimeError, match="DB down"):
        await runner._do_postprocess(task)


@pytest.mark.anyio
async def test_do_postprocess_summary_partial_persist_false():
    """Summary partial=True events must be emitted with persist=False."""
    runner = _make_runner_with_mocks()
    task = MagicMock()

    with patch(
        "app.domain.services.agent_task_runner.run_background_summary",
        new_callable=AsyncMock,
    ) as mock_summary:
        async def emit_events(msgs, llm, on_event, lang="zh", **kwargs):
            await on_event(MessageEvent(role="assistant", message="hi", partial=True))
            await on_event(MessageEvent(role="assistant", message="hi world", partial=False))
        mock_summary.side_effect = emit_events

        await runner._do_postprocess(task)

    # Check persist flags
    partials = [(e, p) for e, p in runner._events_log if isinstance(e, MessageEvent)]
    assert partials[0][1] is False, "partial=True must have persist=False"
    assert partials[1][1] is True, "partial=False must have persist=True"


@pytest.mark.anyio
async def test_do_postprocess_summary_updates_latest_message_and_unread():
    """Final summary (partial=False) must update latest_message and increment unread count."""
    runner = _make_runner_with_mocks()
    task = MagicMock()

    with patch(
        "app.domain.services.agent_task_runner.run_background_summary",
        new_callable=AsyncMock,
    ) as mock_summary:
        async def emit_final(msgs, llm, on_event, lang="zh", **kwargs):
            await on_event(MessageEvent(role="assistant", message="Final summary", partial=False))
        mock_summary.side_effect = emit_final

        await runner._do_postprocess(task)

    # Verify latest_message was updated with the final summary text
    runner._uow.session.update_latest_message.assert_awaited_once()
    call_args = runner._uow.session.update_latest_message.call_args
    assert call_args[0][0] == "test-session-id"
    assert call_args[0][1] == "Final summary"

    # Verify unread count was incremented
    runner._uow.session.increment_unread_message_count.assert_awaited_once_with("test-session-id")


@pytest.mark.anyio
async def test_do_postprocess_partial_summary_does_not_update_latest_message():
    """Partial summary (partial=True) must NOT update latest_message or unread."""
    runner = _make_runner_with_mocks()
    task = MagicMock()

    with patch(
        "app.domain.services.agent_task_runner.run_background_summary",
        new_callable=AsyncMock,
    ) as mock_summary:
        async def emit_partial_only(msgs, llm, on_event, lang="zh", **kwargs):
            await on_event(MessageEvent(role="assistant", message="Partial...", partial=True))
        mock_summary.side_effect = emit_partial_only

        await runner._do_postprocess(task)

    # latest_message must NOT be called for partial
    runner._uow.session.update_latest_message.assert_not_awaited()
    runner._uow.session.increment_unread_message_count.assert_not_awaited()


def test_run_flow_no_sync_flush():
    """_run_flow must not contain synchronous flush submission."""
    import inspect
    from app.domain.services.agent_task_runner import AgentTaskRunner
    source = inspect.getsource(AgentTaskRunner._run_flow)
    assert "memory_flusher" not in source or "submit" not in source, \
        "_run_flow must not contain sync flush submit (moved to _do_postprocess)"
