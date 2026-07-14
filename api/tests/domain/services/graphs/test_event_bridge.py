"""Tests for GraphEventBridge."""

import pytest
from app.domain.models.event import MessageEvent, DoneEvent

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def _make_astream_graph(chunks: list[dict]):
    """Create a fake graph whose astream yields the given chunks."""

    class FakeGraph:
        async def astream(self, input_state, config=None, **kwargs):
            for chunk in chunks:
                yield chunk

    return FakeGraph()


class TestGraphEventBridge:
    async def test_yields_events_from_graph_result(self):
        from app.domain.services.graphs.event_bridge import GraphEventBridge

        msg_event = MessageEvent(role="assistant", message="hello")
        done_event = DoneEvent()

        graph = await _make_astream_graph([
            {"planner_node": {"events": [msg_event], "flow_status": "executing"}},
            {"summarizer_node": {"events": [done_event], "flow_status": "completed"}},
        ])

        bridge = GraphEventBridge()
        events = []
        async for event in bridge.run(graph, {"message": "test"}):
            events.append(event)

        assert len(events) == 2
        assert isinstance(events[0], MessageEvent)
        assert isinstance(events[1], DoneEvent)

    async def test_returns_final_state(self):
        from app.domain.services.graphs.event_bridge import GraphEventBridge

        graph = await _make_astream_graph([
            {"summarizer_node": {"events": [], "flow_status": "completed", "plan": None}},
        ])

        bridge = GraphEventBridge()
        async for _ in bridge.run(graph, {}):
            pass

        assert bridge.final_state["flow_status"] == "completed"

    async def test_empty_events(self):
        from app.domain.services.graphs.event_bridge import GraphEventBridge

        graph = await _make_astream_graph([
            {"node": {"events": []}},
        ])

        bridge = GraphEventBridge()
        events = []
        async for event in bridge.run(graph, {}):
            events.append(event)

        assert events == []

    async def test_queue_events_from_executor(self):
        """Events pushed via event_queue by executor_node are yielded."""
        import asyncio
        from app.domain.services.graphs.event_bridge import GraphEventBridge

        msg_event = MessageEvent(role="assistant", message="from queue")

        class QueuePushGraph:
            async def astream(self, input_state, config=None, **kwargs):
                # Simulate executor_node pushing to queue
                queue = config["configurable"]["event_queue"]
                await queue.put(msg_event)
                yield {"executor_node": {"events": [], "flow_status": "done"}}

        bridge = GraphEventBridge()
        events = []
        async for event in bridge.run(QueuePushGraph(), {}):
            events.append(event)

        assert len(events) == 1
        assert events[0].message == "from queue"

    async def test_wait_event_with_interrupt_no_cleanup_exception(self):
        """When WaitEvent is consumed and generator is closed, cleanup exception
        from _drive_graph should be suppressed (not propagate to caller).

        This prevents WAITING → COMPLETED overwrite in agent_task_runner.
        """
        import asyncio
        from app.domain.services.graphs.event_bridge import GraphEventBridge
        from app.domain.models.event import WaitEvent

        wait_event = WaitEvent()

        class InterruptGraph:
            async def astream(self, input_state, config=None, **kwargs):
                # executor_node pushes WaitEvent and returns should_interrupt
                queue = config["configurable"]["event_queue"]
                await queue.put(wait_event)
                yield {"executor_node": {
                    "events": [],
                    "should_interrupt": True,
                    "flow_status": "executing",
                }}
                # Simulate interrupt_node failing (e.g. no checkpointer)
                raise RuntimeError("interrupt() failed: no checkpointer")

        bridge = GraphEventBridge()
        events = []
        # Simulate agent_task_runner: consume until WaitEvent, then close
        gen = bridge.run(InterruptGraph(), {})
        try:
            async for event in gen:
                events.append(event)
                if isinstance(event, WaitEvent):
                    break  # agent_task_runner would `return` here
        finally:
            await gen.aclose()

        # WaitEvent should have been received
        assert len(events) == 1
        assert isinstance(events[0], WaitEvent)

        # Should NOT raise — cleanup suppressed the RuntimeError
        # (If it raised, agent_task_runner's except handler would overwrite WAITING)

    async def test_wait_event_early_close_drains_interrupt_state(self):
        """HITL close must let the graph publish its interrupt state first."""
        import asyncio

        from app.domain.models.event import WaitEvent
        from app.domain.services.graphs.event_bridge import GraphEventBridge

        settled = asyncio.Event()
        release = asyncio.Event()

        class GatedInterruptGraph:
            async def astream(self, input_state, config=None, **kwargs):
                await config["configurable"]["event_queue"].put(WaitEvent())
                await release.wait()
                settled.set()
                yield {
                    "executor_node": {
                        "events": [],
                        "should_interrupt": True,
                        "flow_status": "executing",
                    }
                }

        bridge = GraphEventBridge()
        generator = bridge.run(
            GatedInterruptGraph(),
            {},
            config={"configurable": {"execution_watchdog": None}},
        )
        assert isinstance(await anext(generator), WaitEvent)

        close_task = asyncio.create_task(generator.aclose())
        close_turn = asyncio.Event()
        asyncio.get_running_loop().call_soon(close_turn.set)
        await close_turn.wait()
        assert close_task.done() is False

        release.set()
        await asyncio.wait_for(close_task, timeout=1.0)

        assert settled.is_set()
        assert bridge.final_state["should_interrupt"] is True
        assert bridge.was_interrupted is True

    async def test_cancelling_hitl_close_propagates_and_cleans_graph_task(self):
        """Parent cancellation during HITL drain must not be swallowed."""
        import asyncio

        from app.domain.models.event import WaitEvent
        from app.domain.services.graphs.event_bridge import GraphEventBridge

        graph_cancelled = asyncio.Event()
        graph_task = None

        class BlockingInterruptGraph:
            async def astream(self, input_state, config=None, **kwargs):
                nonlocal graph_task
                graph_task = asyncio.current_task()
                await config["configurable"]["event_queue"].put(WaitEvent())
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    graph_cancelled.set()
                    raise
                yield {"node": {"events": []}}  # pragma: no cover

        bridge = GraphEventBridge()
        generator = bridge.run(
            BlockingInterruptGraph(),
            {},
            config={"configurable": {"execution_watchdog": None}},
        )
        assert isinstance(await anext(generator), WaitEvent)

        close_task = asyncio.create_task(generator.aclose())
        close_turn = asyncio.Event()
        asyncio.get_running_loop().call_soon(close_turn.set)
        await close_turn.wait()
        assert close_task.done() is False

        close_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close_task

        assert graph_cancelled.is_set()
        assert graph_task is not None
        assert graph_task.done()
        assert graph_task.cancelled()

    def test_hitl_control_event_classification_is_typed(self):
        from app.domain.models.event import (
            ControlAction,
            ControlEvent,
            ControlScope,
            ToolConfirmationEvent,
            WaitEvent,
        )
        from app.domain.services.graphs.event_bridge import _is_hitl_control_event

        confirmation = ToolConfirmationEvent(
            tool_call_id="tc-1",
            tool_name="shell_execute",
            tool_args={},
            risk_level="high",
            risk_reason="test",
            matched_patterns=[],
            timeout_seconds=300,
        )

        assert _is_hitl_control_event(WaitEvent()) is True
        assert _is_hitl_control_event(confirmation) is True
        assert _is_hitl_control_event(
            ControlEvent(action=ControlAction.REQUESTED, scope=ControlScope.SHELL)
        ) is True
        assert _is_hitl_control_event(
            ControlEvent(action=ControlAction.STARTED)
        ) is False
        assert _is_hitl_control_event(
            MessageEvent(role="assistant", message="ordinary")
        ) is False

    async def test_normal_exit_propagates_drive_graph_error(self):
        """In normal (non-cleanup) path, _drive_graph errors should propagate."""
        from app.domain.services.graphs.event_bridge import GraphEventBridge

        class ErrorGraph:
            async def astream(self, input_state, config=None, **kwargs):
                yield {"node": {"events": [], "data": "ok"}}
                raise RuntimeError("graph execution failed")

        bridge = GraphEventBridge()
        with pytest.raises(RuntimeError, match="graph execution failed"):
            async for _ in bridge.run(ErrorGraph(), {}):
                pass

    async def test_watchdog_none_early_close_cancels_live_graph_task(self):
        """Closing after a queue event must not wait forever for a silent graph."""
        import asyncio
        from contextlib import suppress

        from app.domain.services.graphs.event_bridge import GraphEventBridge

        cancelled = asyncio.Event()
        graph_task = None

        class EventThenForeverGraph:
            async def astream(self, input_state, config=None, **kwargs):
                nonlocal graph_task
                graph_task = asyncio.current_task()
                await config["configurable"]["event_queue"].put(
                    MessageEvent(role="assistant", message="first")
                )
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
                yield {"node": {"events": []}}  # pragma: no cover

        bridge = GraphEventBridge()
        generator = bridge.run(
            EventThenForeverGraph(),
            {},
            config={"configurable": {"execution_watchdog": None}},
        )
        first = await anext(generator)
        assert isinstance(first, MessageEvent)

        close_task = asyncio.create_task(generator.aclose())
        try:
            # Shield prevents the test timeout itself from cancelling the
            # bridge and hiding a missing production-side task.cancel().
            await asyncio.wait_for(asyncio.shield(close_task), timeout=0.1)
        finally:
            if not close_task.done():
                close_task.cancel()
                with suppress(asyncio.CancelledError):
                    await close_task

        assert cancelled.is_set()
        assert graph_task is not None
        assert graph_task.done()

    async def test_was_interrupted_with_checkpointer(self):
        """was_interrupted should be True when graph has pending next nodes."""
        from app.domain.services.graphs.event_bridge import GraphEventBridge
        from app.domain.models.event import WaitEvent
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import StateGraph, START, END
        from langgraph.types import interrupt
        from typing_extensions import TypedDict

        class SimpleState(TypedDict):
            value: str

        def node_a(state: SimpleState):
            return {"value": "from_a"}

        def node_b(state: SimpleState):
            interrupt("need input")
            return {"value": "from_b"}

        checkpointer = MemorySaver()
        g = StateGraph(SimpleState)
        g.add_node("a", node_a)
        g.add_node("b", node_b)
        g.add_edge(START, "a")
        g.add_edge("a", "b")
        g.add_edge("b", END)
        graph = g.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "test-was-interrupted"}}
        bridge = GraphEventBridge()
        async for _ in bridge.run(graph, {"value": ""}, config=config):
            pass

        assert bridge.was_interrupted is True


class TestGraphEventBridgeWatchdog:
    """D5: Watchdog integration tests.

    Covers:
    - Idle timeout → HealthEvent(DEGRADED) → recovery hint injection
    - Total timeout with total < idle_timeout → HARD_TERMINATE (the shortened-wait path)
    - HARD_TERMINATE must NOT leak CancelledError to the caller
    - total_timeout_seconds=0 = unlimited (no spurious termination)
    """

    async def test_idle_timeout_emits_degraded_and_sets_hint(self):
        import asyncio
        from app.domain.models.event import HealthEvent, HealthStatus
        from app.domain.services.execution_watchdog import (
            ExecutionControl,
            ExecutionWatchdog,
        )
        from app.domain.services.graphs.event_bridge import GraphEventBridge

        watchdog = ExecutionWatchdog(
            total_timeout_seconds=100,
            idle_timeout_seconds=0.05,
        )
        control = ExecutionControl()

        class SilentThenEventGraph:
            """Graph that stays silent long enough to trigger idle, then emits."""

            async def astream(self, input_state, config=None, **kwargs):
                # Hold so the bridge's wait_for times out at least once
                await asyncio.sleep(0.15)
                yield {
                    "executor_node": {
                        "events": [MessageEvent(role="assistant", message="back")],
                        "flow_status": "completed",
                    }
                }

        config = {
            "configurable": {
                "execution_watchdog": watchdog,
                "execution_control": control,
            }
        }
        bridge = GraphEventBridge()
        events = []
        async for event in bridge.run(SilentThenEventGraph(), {}, config=config):
            events.append(event)

        degraded = [
            e
            for e in events
            if isinstance(e, HealthEvent) and e.status == HealthStatus.DEGRADED
        ]
        assert len(degraded) >= 1
        # Recovery hint must have been injected onto the control object
        # (bridge sets it on SOFT_RECOVER).
        # The hint is consumed by llm_node in real use, but for this test
        # we only assert it was set — consumption is tested elsewhere.
        # (In this test the graph finishes before another llm cycle, so the
        # hint stays on control after DEGRADED is emitted.)
        assert control.idle_recovery_hint is not None

    async def test_total_timeout_smaller_than_idle_triggers_terminate(self):
        """Regression: when total_timeout < idle_timeout, the bridge must still
        fire HARD_TERMINATE at total_timeout instead of waiting for the full
        idle tick. This is the 'total < idle' scenario called out by review.
        """
        import asyncio
        import time
        from app.domain.models.event import HealthEvent, HealthStatus
        from app.domain.services.execution_watchdog import (
            ExecutionControl,
            ExecutionWatchdog,
        )
        from app.domain.services.graphs.event_bridge import GraphEventBridge

        watchdog = ExecutionWatchdog(
            total_timeout_seconds=0.1,      # 100ms hard cap
            idle_timeout_seconds=10.0,       # 10s idle — MUCH larger
        )
        control = ExecutionControl()

        class ForeverSilentGraph:
            async def astream(self, input_state, config=None, **kwargs):
                # Never yield anything so the queue stays empty.
                await asyncio.sleep(30)
                yield {"node": {"events": []}}  # unreachable

        config = {
            "configurable": {
                "execution_watchdog": watchdog,
                "execution_control": control,
            }
        }
        bridge = GraphEventBridge()
        events = []
        started = time.monotonic()
        async for event in bridge.run(ForeverSilentGraph(), {}, config=config):
            events.append(event)
            if isinstance(event, HealthEvent) and event.status == HealthStatus.TERMINATING:
                break  # terminating reached — sanity short-circuit
        elapsed = time.monotonic() - started

        # Must terminate quickly (within ~1s, well under the 10s idle tick)
        assert elapsed < 1.0, f"Bridge waited {elapsed:.2f}s — exceeded total cap"
        # Must have emitted the terminating health event
        terminating = [
            e
            for e in events
            if isinstance(e, HealthEvent) and e.status == HealthStatus.TERMINATING
        ]
        assert len(terminating) == 1
        # Control flag must be set so cooperative exit works
        assert control.should_terminate is True

    async def test_hard_terminate_does_not_leak_cancelled_error(self):
        """Regression: HARD_TERMINATE must not propagate CancelledError from
        task.cancel() to the caller. Previously _normal_exit = True ran after
        break, and await task raised CancelledError unhandled.
        """
        import asyncio
        from app.domain.services.execution_watchdog import (
            ExecutionControl,
            ExecutionWatchdog,
        )
        from app.domain.services.graphs.event_bridge import GraphEventBridge

        watchdog = ExecutionWatchdog(
            total_timeout_seconds=0.05,
            idle_timeout_seconds=10.0,
        )
        control = ExecutionControl()

        class ForeverSilentGraph:
            async def astream(self, input_state, config=None, **kwargs):
                await asyncio.sleep(30)
                yield {"node": {"events": []}}

        config = {
            "configurable": {
                "execution_watchdog": watchdog,
                "execution_control": control,
            }
        }
        bridge = GraphEventBridge()

        # If CancelledError leaks, this async for block raises and the test fails.
        events = []
        try:
            async for event in bridge.run(ForeverSilentGraph(), {}, config=config):
                events.append(event)
        except asyncio.CancelledError:
            pytest.fail("CancelledError leaked from bridge.run() on HARD_TERMINATE")

        # Ensure we actually reached HARD_TERMINATE, not something else.
        from app.domain.models.event import HealthEvent, HealthStatus
        terminating = [
            e
            for e in events
            if isinstance(e, HealthEvent) and e.status == HealthStatus.TERMINATING
        ]
        assert len(terminating) == 1

    async def test_total_timeout_zero_is_unlimited(self):
        """Regression: total_timeout_seconds=0 must mean 'no hard cap', not
        'expire immediately'."""
        import asyncio
        from app.domain.services.execution_watchdog import (
            ExecutionControl,
            ExecutionWatchdog,
        )
        from app.domain.services.graphs.event_bridge import GraphEventBridge

        watchdog = ExecutionWatchdog(
            total_timeout_seconds=0,         # unlimited
            idle_timeout_seconds=1.0,
        )
        control = ExecutionControl()

        class QuickGraph:
            async def astream(self, input_state, config=None, **kwargs):
                yield {
                    "executor_node": {
                        "events": [MessageEvent(role="assistant", message="ok")],
                        "flow_status": "completed",
                    }
                }

        config = {
            "configurable": {
                "execution_watchdog": watchdog,
                "execution_control": control,
            }
        }
        bridge = GraphEventBridge()
        events = []
        async for event in bridge.run(QuickGraph(), {}, config=config):
            events.append(event)

        # Must NOT have terminated — normal completion
        from app.domain.models.event import HealthEvent, HealthStatus
        terminating = [
            e
            for e in events
            if isinstance(e, HealthEvent) and e.status == HealthStatus.TERMINATING
        ]
        assert terminating == []
        assert control.should_terminate is False
        # Normal event flowed through
        assert any(isinstance(e, MessageEvent) for e in events)

    async def test_wait_guard_uses_the_same_watchdog_and_reaches_graph_config(self):
        from unittest.mock import MagicMock

        from app.application.services.coordinator_wait_guard import CoordinatorWaitGuard
        from app.domain.services.execution_watchdog import ExecutionWatchdog
        from app.domain.services.graphs.event_bridge import GraphEventBridge

        watchdog = ExecutionWatchdog(total_timeout_seconds=0, idle_timeout_seconds=10)
        factory = MagicMock(side_effect=CoordinatorWaitGuard)
        captured = {}

        class CapturingGraph:
            async def astream(self, input_state, config=None, **kwargs):
                captured.update(config["configurable"])
                yield {"node": {"events": []}}

        config = {
            "configurable": {
                "execution_watchdog": watchdog,
                "coordinator_wait_guard_factory": factory,
            }
        }
        bridge = GraphEventBridge()
        async for _ in bridge.run(CapturingGraph(), {}, config=config):
            pass

        guard = captured["coordinator_wait_guard"]
        assert isinstance(guard, CoordinatorWaitGuard)
        assert guard.watchdog is watchdog
        factory.assert_called_once_with(watchdog=watchdog)

    async def test_none_watchdog_does_not_create_wait_guard_or_start_monitor(self):
        from unittest.mock import MagicMock

        from app.domain.services.graphs.event_bridge import GraphEventBridge

        factory = MagicMock(side_effect=AssertionError("must not create guard"))
        captured = {}

        class QuickGraph:
            async def astream(self, input_state, config=None, **kwargs):
                captured.update(config["configurable"])
                yield {
                    "node": {
                        "events": [MessageEvent(role="assistant", message="ok")]
                    }
                }

        config = {
            "configurable": {
                "execution_watchdog": None,
                "coordinator_wait_guard_factory": factory,
            }
        }
        bridge = GraphEventBridge()
        events = []
        async for event in bridge.run(QuickGraph(), {}, config=config):
            events.append(event)

        factory.assert_not_called()
        assert "coordinator_wait_guard" not in captured
        assert len(events) == 1
