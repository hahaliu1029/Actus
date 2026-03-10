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
        async def astream(self, input_state, config=None):
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
            async def astream(self, input_state, config=None):
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
