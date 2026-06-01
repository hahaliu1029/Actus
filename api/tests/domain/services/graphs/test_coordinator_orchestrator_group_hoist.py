import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.application.services.coordinator_run_orchestrator import (
    CoordinatorRunOrchestrator,
)

pytestmark = pytest.mark.anyio


async def test_run_skips_subscribe_when_group_precreated_but_keeps_destroy():
    """[INV-F4.3] With observer_group_precreated=True, run() does NOT subscribe
    (group already exists) but still destroys on completion."""
    subscriber = MagicMock()
    subscriber.subscribe = AsyncMock()
    subscriber.destroy_group = AsyncMock()
    async def _consume(**kwargs):
        if False:
            yield {}
    subscriber.consume = _consume
    orch = CoordinatorRunOrchestrator(
        publisher=MagicMock(), parent_session_id="p", coordinator_run_id="run-1",
        mailbox_subscriber=subscriber,
    )
    cancel = asyncio.Event()
    cancel.set()  # short-circuit the watcher so run() returns promptly
    await orch.run(
        coordinator_run_id="run-1", root_session_id="root-1",
        work_units_pending=[], child_session_ids={}, cancel_event=cancel,
        timeout_seconds=1.0, observer_group_precreated=True,
    )
    subscriber.subscribe.assert_not_called()       # hoisted to dispatch
    subscriber.destroy_group.assert_awaited()      # orchestrator still cleans up


def test_dispatch_hoists_orchestrator_group_before_start_and_publish():
    """[INV-F4.1] In dispatch_node, the orchestrator group-create
    (consumer_group=orchestrator_group) is hoisted BEFORE the first
    runner_starter.start(...) and before publisher.publish(...)."""
    import pathlib, re
    src = pathlib.Path(
        "app/domain/services/graphs/parallel_execution_subgraph.py"
    ).read_text()

    def _pos(pat):
        m = re.search(pat, src, re.MULTILINE)
        assert m, f"pattern not found: {pat}"
        return m.start()

    i_orch = _pos(r"^\s*consumer_group=orchestrator_group")
    i_start = _pos(r"^\s*await runner_starter\.start\(")
    i_publish = _pos(r"^\s*await publisher\.publish\(")
    assert i_orch < i_start < i_publish, (
        "orchestrator group-create must be hoisted before runner_starter.start "
        "and publisher.publish (INV-F4.1 G4-min)"
    )
