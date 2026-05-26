"""Integration tests for Task 10: Full gate → batch → runner → flusher chain.

Verifies the end-to-end flush pipeline:
1. AgentTaskRunner.__init__ has memory_flusher parameter
2. Runner reads _pending_flush_batch from flow and calls flusher.submit()
3. When memory_flusher is None, no error even if batch exists
4. Full gate → chunk → batch pipeline with flush_enabled=True
5. Gate + chunking metadata (step_title, message_types, tool_names, created_at)
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.domain.models.app_config import AgentConfig, MemoryConfig
from app.domain.models.memory_chunk import FlushBatch, RawChunk
from app.domain.models.plan import ExecutionStatus, Plan, Step

from tests.conftest import TEST_USER_ID_FIXED


# PR-4+8: _evaluate_flush_gate became async so callers can await the LLM
# gate + Redis quota checks. Module-level anyio mark + backend fixture lets
# the gate/chunking tests run under asyncio; sync tests in this file are
# unaffected (anyio mark is a no-op on non-async test functions).


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ─── Helper: Capturing Flusher ────────────────────────────────────────────────


class _CapturingFlusher:
    """Test double that captures submitted FlushBatches."""

    def __init__(self) -> None:
        self.batches: list[FlushBatch] = []

    def submit(self, batch: FlushBatch) -> None:
        self.batches.append(batch)


# ─── Helper: Flow factory ─────────────────────────────────────────────────────


def _make_flow(**overrides):
    """Create a PlannerReActFlow with all required kwargs."""
    from app.domain.services.flows.planner_react import PlannerReActFlow

    kwargs = {
        "llm": MagicMock(),
        "agent_config": AgentConfig(
            memory=MemoryConfig(
                flush_enabled=True,
                flush_min_steps=1,
                flush_min_new_tokens=500,
            )
        ),
        "session_id": "s1",
        "user_id": TEST_USER_ID_FIXED,
        "uow_factory": MagicMock(),
        "browser": MagicMock(),
        "sandbox": MagicMock(),
        "search_engine": MagicMock(),
        "mcp_tool": MagicMock(),
        "a2a_tool": MagicMock(),
        "skill_tool": MagicMock(),
    }
    kwargs.update(overrides)
    return PlannerReActFlow(**kwargs)


def _make_plan_with_completed_steps(num_steps: int) -> Plan:
    """Create a Plan with the given number of completed steps."""
    steps = [
        Step(
            id=f"s{i}",
            description=f"Step {i}",
            status=ExecutionStatus.COMPLETED,
            result=f"Result {i}",
            success=True,
        )
        for i in range(num_steps)
    ]
    return Plan(
        title="Test Plan",
        goal="Test goal",
        steps=steps,
        status=ExecutionStatus.COMPLETED,
    )


def _make_rich_messages(count: int, content: str = "测试内容" * 200) -> list:
    """Create alternating Human/AI messages with enough content to pass the gate."""
    msgs = []
    for i in range(count):
        if i % 2 == 0:
            msgs.append(HumanMessage(content=f"{content} {i}"))
        else:
            msgs.append(AIMessage(content=f"response {i} {content}"))
    return msgs


# ─── Test 1: Runner attribute exists ─────────────────────────────────────────


class TestRunnerHasMemoryFlusherParam:
    """AgentTaskRunner.__init__ must declare memory_flusher parameter."""

    def test_memory_flusher_in_signature(self) -> None:
        """memory_flusher should appear in __init__ parameter list."""
        from app.domain.services.agent_task_runner import AgentTaskRunner

        sig = inspect.signature(AgentTaskRunner.__init__)
        assert "memory_flusher" in sig.parameters, (
            "AgentTaskRunner.__init__ must accept memory_flusher parameter"
        )

    def test_memory_flusher_defaults_to_none(self) -> None:
        """memory_flusher should default to None (optional param)."""
        from app.domain.services.agent_task_runner import AgentTaskRunner

        sig = inspect.signature(AgentTaskRunner.__init__)
        param = sig.parameters["memory_flusher"]
        assert param.default is None, (
            f"memory_flusher default should be None, got {param.default!r}"
        )


# ─── Test 2: Runner submits batch from flow ───────────────────────────────────


class TestRunnerSubmitsBatchFromFlow:
    """Runner reads _pending_flush_batch from flow and calls flusher.submit()."""

    def test_runner_submits_batch_when_flusher_set(self) -> None:
        """Simulate runner pattern: getattr + submit on a capturing flusher."""
        flusher = _CapturingFlusher()
        flow = _make_flow()

        # Manually set a pending batch (as flow would after _evaluate_flush_gate)
        batch = FlushBatch(
            session_id="s1",
            user_id=TEST_USER_ID_FIXED,
            from_cursor=0,
            target_cursor=5,
            chunks=(
                RawChunk(
                    content="test content",
                    session_id="s1",
                    user_id=TEST_USER_ID_FIXED,
                    source="conversation",
                    metadata={"turn_index": 0, "message_types": ["HumanMessage"], "created_at": "2024-01-01T00:00:00"},
                    content_hash="abc123",
                ),
            ),
        )
        flow._pending_flush_batch = batch

        # Replicate the runner pattern from agent_task_runner._run_flow
        flush_batch = getattr(flow, "_pending_flush_batch", None)
        if flush_batch and flusher:
            flusher.submit(flush_batch)

        assert len(flusher.batches) == 1
        assert flusher.batches[0] is batch

    def test_runner_does_not_submit_when_no_batch(self) -> None:
        """If _pending_flush_batch is None, flusher.submit is never called."""
        flusher = _CapturingFlusher()
        flow = _make_flow()

        # No batch set — default is None
        flush_batch = getattr(flow, "_pending_flush_batch", None)
        if flush_batch and flusher:
            flusher.submit(flush_batch)

        assert len(flusher.batches) == 0


# ─── Test 3: Flusher None guard ───────────────────────────────────────────────


class TestFlusherNoneGuard:
    """Runner guard: flusher=None + batch exists → short-circuit, no error.

    This tests the runner's `if flush_batch and flusher:` guard in isolation.
    The flow fixture uses default flush_enabled=True, but the test manually
    injects a batch and sets flusher=None to verify the guard itself.

    C5.1 system invariant: when flush_enabled=True, flusher is always present
    at runtime (guaranteed by lifespan). This test verifies the guard's safety
    net, not a normal production path.
    """

    def test_no_error_when_flusher_none_and_batch_exists(self) -> None:
        """Runner guard: `if flush_batch and flusher` short-circuits on None flusher."""
        flow = _make_flow()
        flusher = None  # intentionally None

        batch = FlushBatch(
            session_id="s1",
            user_id=TEST_USER_ID_FIXED,
            from_cursor=0,
            target_cursor=3,
            chunks=(),
        )
        flow._pending_flush_batch = batch

        # This is the exact pattern used in agent_task_runner._run_flow
        flush_batch = getattr(flow, "_pending_flush_batch", None)
        # Should not raise — flusher is None so submit is never called
        if flush_batch and flusher:
            flusher.submit(flush_batch)  # type: ignore[union-attr]

        # Confirm no exception was raised and flusher was never invoked
        assert flusher is None  # no mutation expected

    def test_flow_has_pending_flush_batch_attribute(self) -> None:
        """PlannerReActFlow must expose _pending_flush_batch attribute."""
        flow = _make_flow()
        assert hasattr(flow, "_pending_flush_batch"), (
            "PlannerReActFlow must have _pending_flush_batch attribute"
        )


# ─── Test 4: Full gate → chunk → batch pipeline ───────────────────────────────


class TestFullGateChunkBatchPipeline:
    """End-to-end: _evaluate_flush_gate produces a correct FlushBatch."""

    @pytest.mark.anyio
    async def test_gate_produces_batch_with_correct_cursors(self) -> None:
        """flush_enabled=True, sufficient steps + tokens → batch with correct cursors."""
        flow = _make_flow(session_id="s1", user_id=TEST_USER_ID_FIXED)
        flow._flush_cursor = 0

        msgs = _make_rich_messages(10)
        plan = _make_plan_with_completed_steps(3)
        await flow._evaluate_flush_gate(msgs, plan)

        batch = flow._pending_flush_batch
        assert batch is not None, "Gate should pass and produce a FlushBatch"
        assert isinstance(batch, FlushBatch)
        assert batch.session_id == "s1"
        assert batch.user_id == TEST_USER_ID_FIXED
        assert batch.from_cursor == 0
        assert batch.target_cursor == len(msgs)

    @pytest.mark.anyio
    async def test_gate_batch_has_chunks(self) -> None:
        """Gate-produced batch must contain at least one chunk."""
        flow = _make_flow()
        flow._flush_cursor = 0

        msgs = _make_rich_messages(10)
        plan = _make_plan_with_completed_steps(2)
        await flow._evaluate_flush_gate(msgs, plan)

        batch = flow._pending_flush_batch
        assert batch is not None
        assert len(batch.chunks) > 0

    @pytest.mark.anyio
    async def test_gate_batch_chunks_are_raw_chunks(self) -> None:
        """Each chunk in the batch must be a RawChunk instance."""
        flow = _make_flow(session_id="sess42", user_id="user42")
        flow._flush_cursor = 0

        msgs = _make_rich_messages(8)
        plan = _make_plan_with_completed_steps(1)
        await flow._evaluate_flush_gate(msgs, plan)

        batch = flow._pending_flush_batch
        assert batch is not None
        for chunk in batch.chunks:
            assert isinstance(chunk, RawChunk)
            assert chunk.session_id == "sess42"
            assert chunk.user_id == "user42"
            assert chunk.content  # non-empty
            assert chunk.content_hash  # non-empty

    @pytest.mark.anyio
    async def test_gate_disabled_produces_no_batch(self) -> None:
        """flush_enabled=False → _pending_flush_batch stays None."""
        flow = _make_flow(
            agent_config=AgentConfig(memory=MemoryConfig(flush_enabled=False))
        )
        flow._flush_cursor = 0

        msgs = _make_rich_messages(10)
        plan = _make_plan_with_completed_steps(5)
        await flow._evaluate_flush_gate(msgs, plan)

        assert flow._pending_flush_batch is None

    @pytest.mark.anyio
    async def test_gate_min_steps_not_met_produces_no_batch(self) -> None:
        """flush_min_steps=5 but only 2 completed steps → no batch."""
        flow = _make_flow(
            agent_config=AgentConfig(
                memory=MemoryConfig(
                    flush_enabled=True,
                    flush_min_steps=5,
                    flush_min_new_tokens=500,
                )
            )
        )
        flow._flush_cursor = 0

        msgs = _make_rich_messages(10)
        plan = _make_plan_with_completed_steps(2)
        await flow._evaluate_flush_gate(msgs, plan)

        assert flow._pending_flush_batch is None

    @pytest.mark.anyio
    async def test_gate_updates_cursor_on_batch_production(self) -> None:
        """After _evaluate_flush_gate, the cursor is stored in the batch's target_cursor."""
        flow = _make_flow()
        flow._flush_cursor = 0

        msgs = _make_rich_messages(6)
        plan = _make_plan_with_completed_steps(2)
        await flow._evaluate_flush_gate(msgs, plan)

        batch = flow._pending_flush_batch
        assert batch is not None
        assert batch.target_cursor == 6
        # C5.1: cursor should be advanced after batch creation
        assert flow._flush_cursor == 6  # == target_cursor == len(msgs)

    @pytest.mark.anyio
    async def test_flusher_receives_batch_after_gate(self) -> None:
        """Full chain: gate passes → capturing flusher receives the batch."""
        flusher = _CapturingFlusher()
        flow = _make_flow(session_id="chain-s", user_id="chain-u")
        flow._flush_cursor = 0

        msgs = _make_rich_messages(8)
        plan = _make_plan_with_completed_steps(2)
        await flow._evaluate_flush_gate(msgs, plan)

        # Simulate runner submission
        flush_batch = getattr(flow, "_pending_flush_batch", None)
        if flush_batch and flusher:
            flusher.submit(flush_batch)

        assert len(flusher.batches) == 1
        submitted = flusher.batches[0]
        assert submitted.session_id == "chain-s"
        assert submitted.user_id == "chain-u"
        assert submitted.from_cursor == 0
        assert submitted.target_cursor == len(msgs)


# ─── Test 5: Gate + chunking metadata ────────────────────────────────────────


class TestChunkingMetadata:
    """Verify chunks contain required metadata fields after full pipeline."""

    @pytest.mark.anyio
    async def test_chunks_have_message_types(self) -> None:
        """Each chunk metadata must include message_types."""
        flow = _make_flow()
        flow._flush_cursor = 0

        msgs = _make_rich_messages(6)
        plan = _make_plan_with_completed_steps(1)
        await flow._evaluate_flush_gate(msgs, plan)

        batch = flow._pending_flush_batch
        assert batch is not None
        for chunk in batch.chunks:
            assert "message_types" in chunk.metadata, (
                f"chunk.metadata missing 'message_types': {chunk.metadata}"
            )

    @pytest.mark.anyio
    async def test_chunks_have_created_at(self) -> None:
        """Each chunk metadata must include created_at."""
        flow = _make_flow()
        flow._flush_cursor = 0

        msgs = _make_rich_messages(6)
        plan = _make_plan_with_completed_steps(1)
        await flow._evaluate_flush_gate(msgs, plan)

        batch = flow._pending_flush_batch
        assert batch is not None
        for chunk in batch.chunks:
            assert "created_at" in chunk.metadata, (
                f"chunk.metadata missing 'created_at': {chunk.metadata}"
            )

    @pytest.mark.anyio
    async def test_chunks_tool_names_only_present_when_nonempty(self) -> None:
        """tool_names is only added to metadata when non-empty (conditional inclusion).

        Chunks from pure HumanMessage/AIMessage groups do not have tool_names.
        Chunks from tool-call groups do have tool_names.
        """
        flow = _make_flow()
        flow._flush_cursor = 0

        # Simple messages without tool calls — tool_names should NOT be present
        msgs = _make_rich_messages(6)
        plan = _make_plan_with_completed_steps(1)
        await flow._evaluate_flush_gate(msgs, plan)

        batch = flow._pending_flush_batch
        assert batch is not None
        for chunk in batch.chunks:
            # tool_names is absent OR present but non-empty (never an empty list)
            if "tool_names" in chunk.metadata:
                assert chunk.metadata["tool_names"], (
                    "tool_names key present but empty — should only appear when non-empty"
                )

    @pytest.mark.anyio
    async def test_chunks_have_step_title_with_plan(self) -> None:
        """When plan has completed steps, step_title is added to chunk metadata.

        step_title is conditionally included — only when non-empty. With completed
        steps that have descriptions, at least some chunks should carry it.
        """
        flow = _make_flow()
        flow._flush_cursor = 0

        msgs = _make_rich_messages(8)
        plan = _make_plan_with_completed_steps(3)
        await flow._evaluate_flush_gate(msgs, plan)

        batch = flow._pending_flush_batch
        assert batch is not None
        # step_title is conditionally included (only when non-empty)
        # With completed steps that have descriptions, at least some chunks should have it
        chunks_with_step_title = [
            chunk for chunk in batch.chunks if "step_title" in chunk.metadata
        ]
        assert len(chunks_with_step_title) > 0, (
            "Expected at least some chunks to have 'step_title' when plan has completed steps"
        )

    @pytest.mark.anyio
    async def test_chunks_with_tool_messages_have_tool_names(self) -> None:
        """Chunks produced from AIMessage+ToolMessage groups include tool_names."""
        flow = _make_flow(
            agent_config=AgentConfig(
                memory=MemoryConfig(
                    flush_enabled=True,
                    flush_min_steps=1,
                    flush_min_new_tokens=500,
                )
            )
        )
        flow._flush_cursor = 0

        # Build messages that include tool calls
        tool_content = "search results " * 100
        msgs = [
            HumanMessage(content="Search for Python docs " * 30),
            AIMessage(
                content="Let me search " * 30,
                additional_kwargs={
                    "tool_calls": [
                        {
                            "id": "tc1",
                            "function": {"name": "web_search", "arguments": "{}"},
                            "type": "function",
                        }
                    ]
                },
            ),
            ToolMessage(content=tool_content, tool_call_id="tc1", name="web_search"),
            AIMessage(content="Here are the results " * 30),
            HumanMessage(content="Thank you " * 30),
            AIMessage(content="You are welcome " * 30),
        ]
        plan = _make_plan_with_completed_steps(2)
        await flow._evaluate_flush_gate(msgs, plan)

        batch = flow._pending_flush_batch
        assert batch is not None
        # At least one chunk should have tool names populated
        all_tool_names: set[str] = set()
        for chunk in batch.chunks:
            tool_names = chunk.metadata.get("tool_names", [])
            all_tool_names.update(tool_names)

        assert "web_search" in all_tool_names, (
            f"Expected 'web_search' in tool_names across chunks; got {all_tool_names}"
        )


# ─── Test 6: Persist-level cursor validation (C5.1) ────────────────────────


class TestPersistCursorAdvancement:
    """Verify _persist_after_graph_inner saves the advanced flush_cursor."""

    @pytest.mark.anyio
    async def test_persist_saves_advanced_cursor(self) -> None:
        """After _evaluate_flush_gate advances cursor, Memory(flush_cursor=...)
        in _persist_after_graph_inner must use the new value, not the old one.

        Guards against someone reverting the persist path to use a stale cursor.
        Pattern ref: test_flush_cursor_loading.py:20-80.
        """
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock

        mock_session_repo = AsyncMock()
        mock_session_repo.save_memory = AsyncMock()
        mock_session_repo.save_summary = AsyncMock()
        mock_session_repo.get_summary = AsyncMock(return_value=[])

        @asynccontextmanager
        async def uow_factory():
            uow = MagicMock()
            uow.session = mock_session_repo
            yield uow

        flow = _make_flow(uow_factory=uow_factory)
        flow._flush_cursor = 0

        msgs = _make_rich_messages(6)
        plan = _make_plan_with_completed_steps(2)
        final = {
            "messages": msgs,
            "plan": plan,
            "should_interrupt": False,
        }

        await flow._persist_after_graph_inner(final, summaries=[])

        # Assert save_memory was called with Memory whose flush_cursor == 6
        mock_session_repo.save_memory.assert_awaited()
        saved_memory = mock_session_repo.save_memory.call_args[0][2]
        assert saved_memory.flush_cursor == 6, (
            f"Expected flush_cursor=6 (advanced), got {saved_memory.flush_cursor}"
        )
