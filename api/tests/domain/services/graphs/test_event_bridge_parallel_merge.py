"""PR-9b-A INV-A8 — GraphEventBridge merges event_queue into the
parallel-subgraph cfg.

Regression-locks the behavior at:
- ``event_bridge.py:74-79`` (the merge seam: ``merged_config["configurable"]
  ["event_queue"] = queue`` + per-key update of caller-supplied configurable)
- ``main_graph.py:129`` (the pass-through: ``config={"configurable": cfg}``
  from ``_run_parallel_backend`` into the parallel subgraph)

A6 already wired ``_run_parallel_backend`` (main_graph.py:219-224) to raise
``RuntimeError`` when ``cfg.get("event_queue")`` is None. If a future
refactor silently drops the merge here, coordinator dispatch dies loud
inside the parallel subgraph instead of failing silent.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_event_queue_visible_in_parallel_subgraph_inner_cfg() -> None:
    """Drive GraphEventBridge end-to-end and confirm that a node receives
    ``configurable["event_queue"]`` set to a real ``asyncio.Queue``.

    This mirrors the production seam where ``main_graph._run_parallel_backend``
    reads ``cfg.get("event_queue")`` to close over the bridge-owned queue
    before invoking the parallel subgraph (``subgraph.ainvoke(
    config={"configurable": cfg})`` at main_graph.py:129).
    """
    from app.domain.services.graphs.event_bridge import GraphEventBridge

    inner_seen: dict[str, Any] = {}

    class ParallelSubgraphFakeGraph:
        """Mimics the real outer→inner cfg pass-through.

        The bridge's merged ``configurable`` reaches ``astream`` via the
        ``config`` kwarg. The real ``main_graph._run_parallel_backend``
        then re-uses that ``configurable`` when invoking the parallel
        subgraph. Here we simply capture what the bridge handed us so we
        can assert the merge happened.
        """

        async def astream(
            self,
            input_state: Any,
            config: dict[str, Any] | None = None,
            **kwargs: Any,
        ):
            configurable = (config or {}).get("configurable") or {}
            # Snapshot what _run_parallel_backend would see (cfg = configurable).
            inner_seen["configurable"] = configurable
            inner_seen["event_queue"] = configurable.get("event_queue")
            # Preserve caller-supplied keys so we can verify they survived
            # the per-key merge in event_bridge.py:77-81.
            inner_seen["thread_id"] = configurable.get("thread_id")
            yield {"node": {"events": [], "flow_status": "done"}}

    # Caller-supplied configurable: bridge must preserve these AND inject
    # ``event_queue`` alongside them.
    caller_config = {"configurable": {"thread_id": "test-parallel-merge"}}

    bridge = GraphEventBridge()
    events = []
    async for event in bridge.run(
        ParallelSubgraphFakeGraph(),
        {"message": "drive bridge"},
        config=caller_config,
    ):
        events.append(event)

    # Core invariant: event_queue must reach the inner cfg.
    assert "event_queue" in inner_seen["configurable"], (
        "event_queue not merged into inner configurable — INV-A8 regression "
        "at event_bridge.py:74-79. A6 hard-fails _run_parallel_backend "
        "(main_graph.py:219-224) when this key is absent."
    )
    assert inner_seen["event_queue"] is not None, (
        "event_queue merged as None — must be the bridge-owned asyncio.Queue"
    )
    # The merged value must be a real asyncio.Queue, never a sentinel/dummy
    # (A6 expects ``.put_nowait`` behavior at main_graph.py:234).
    assert isinstance(inner_seen["event_queue"], asyncio.Queue), (
        "event_queue must be a real asyncio.Queue (A6 calls .put_nowait on "
        f"it at main_graph.py:234); got {type(inner_seen['event_queue'])!r}"
    )
    # Caller-supplied configurable keys must survive the merge
    # (event_bridge.py:77-81 does a per-key update, not a wholesale replace).
    assert inner_seen["thread_id"] == "test-parallel-merge", (
        "caller-supplied configurable.thread_id was dropped by the merge — "
        "INV-A8 regression at event_bridge.py:77-81"
    )
