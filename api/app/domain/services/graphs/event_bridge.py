"""Bridge between LangGraph execution and Actus Event stream.

Uses an asyncio.Queue so that events from both the main graph nodes and
the nested react_graph are yielded to the frontend in real-time.

Nodes that receive an ``event_queue`` via LangGraph config can push events
directly; other nodes' events are picked up from astream output.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator

from app.domain.models.event import BaseEvent

logger = logging.getLogger(__name__)


class GraphEventBridge:
    """Runs a LangGraph and streams events in real-time via an async queue."""

    def __init__(self) -> None:
        self._final_state: dict[str, Any] = {}

    @property
    def final_state(self) -> dict[str, Any]:
        """The full graph output state after execution."""
        return self._final_state

    async def run(
        self,
        graph: Any,
        input_state: dict[str, Any],
    ) -> AsyncGenerator[BaseEvent, None]:
        """Stream the graph, yielding events as each node produces them.

        Events reach the caller via two paths:

        1. **Queue path** — nodes that receive ``event_queue`` in config push
           events directly (used by ``executor_node`` for react sub-graph).
        2. **State path** — nodes that only return ``{"events": [...]}`` have
           their events picked up here from the ``astream`` output.
        """
        queue: asyncio.Queue[BaseEvent | None] = asyncio.Queue()

        async def _drive_graph() -> None:
            """Run the graph and forward state-path events to the queue."""
            try:
                async for chunk in graph.astream(
                    input_state,
                    config={"configurable": {"event_queue": queue}},
                ):
                    for _node_name, node_output in chunk.items():
                        if not isinstance(node_output, dict):
                            continue
                        self._final_state.update(node_output)
                        # Emit events that were NOT already pushed via queue
                        # (nodes using queue return events=[])
                        for evt in node_output.get("events") or []:
                            if isinstance(evt, BaseEvent):
                                await queue.put(evt)
            except Exception:
                logger.exception("GraphEventBridge: graph execution error")
                raise
            finally:
                await queue.put(None)  # sentinel

        task = asyncio.create_task(_drive_graph())

        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            await task
