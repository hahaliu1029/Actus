"""C2 PR-8 §13 Task 8.4 — integration: 5 coordinator events on event_queue.

End-to-end-style coverage: exercise the actual subgraph dispatch + reducer
nodes wired to an asyncio.Queue (the production event_queue surface) and
assert the 5 lineage-tagged events arrive in the right order. SSE-side
reconnect is already covered by ``redis_event_recovery`` infra; this file
pins the emit-side contract that feeds it.

NB: This file lives under tests/integration/ per the C2 PR-8 spec
(``tests/integration/test_sse_coordinator_lineage.py``) but does NOT
require Postgres/Redis — collaborators are mocked. We override the
auto-use ``_migrate`` fixture so collection doesn't drag in alembic.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(scope="module", autouse=True)
def _migrate():  # noqa: PT004
    """Override the parent integration conftest's autouse migration so
    this DB-free file does not require a running Postgres."""
    yield

from app.application.services.patch_reducer_service import (
    ReducerDiagnostics,
    ReducerOutput,
)
from app.domain.models.event import (
    BaseEvent,
    CoordinatorDispatchEvent,
    CoordinatorReduceEvent,
    CoordinatorWorkerSpawnedEvent,
    ExecutionStatePayload,
    MessageEvent,
    PendingExecutionEvent,
)
from app.domain.models.mailbox_envelope import (
    CostAggregate,
    ResultReadyOutcome,
)
from app.domain.models.patch_apply_plan import GroupOutcome
from app.domain.models.work_unit import WorkUnit, WorkUnitRequest
from app.domain.models.session import Session, SessionStatus
from app.domain.services.graphs.parallel_execution_subgraph import (
    WorkerResult,
    dispatch_node,
    reducer_node,
)
from app.interfaces.endpoints import session_routes
from app.interfaces.schemas.session import ChatRequest


def _mk_session(sid: str) -> MagicMock:
    m = MagicMock()
    m.id = sid
    return m


def _dispatch_config(*, event_queue: asyncio.Queue) -> dict:
    rehydrate = AsyncMock()
    rehydrate.detect_existing_run = AsyncMock(return_value=None)
    session_service = AsyncMock()
    session_service.peek_coordinator_attempt = AsyncMock(return_value=None)
    session_service.bump_coordinator_attempt = AsyncMock(return_value=1)
    session_service.create_session_with_parent = AsyncMock(
        side_effect=[_mk_session("c1"), _mk_session("c2")],
    )
    runner_starter = AsyncMock()
    publisher = AsyncMock()
    artifact = AsyncMock()
    artifact.put_content_addressed_bytes = AsyncMock(
        return_value="minio://manifest-ref",
    )
    parent_sandbox = AsyncMock()
    orchestrator = AsyncMock()
    orchestrator_factory = MagicMock()
    orchestrator_factory.build = MagicMock(return_value=orchestrator)
    subscriber = AsyncMock()
    return {
        "configurable": {
            "rehydrate_service": rehydrate,
            "session_service": session_service,
            "child_runner_starter": runner_starter,
            "mailbox_publisher": publisher,
            "mailbox_subscriber": subscriber,
            "artifact_storage": artifact,
            "parent_sandbox": parent_sandbox,
            "orchestrator_factory": orchestrator_factory,
            "cancel_event": asyncio.Event(),
            "event_queue": event_queue,
        }
    }


@pytest.mark.integration
@pytest.mark.anyio
async def test_dispatch_and_reducer_flow_emits_lineage_events_on_queue() -> None:
    """Full happy-path: dispatch → reducer drains 1 + N + 1 = N+2 events."""
    queue: asyncio.Queue = asyncio.Queue()
    config = _dispatch_config(event_queue=queue)
    state = {
        "coordinator_run_id": None,
        "step_id": "step-integ",
        "work_unit_requests": [
            WorkUnitRequest(
                objective="explore X", phase="exploration",
                allowed_tools=["file_read"],
            ),
            WorkUnitRequest(
                objective="explore Y", phase="exploration",
                allowed_tools=["file_read"],
            ),
        ],
        "work_units": [],
        "parent_session_id": "p-integ",
        "user_id": "u-integ",
        "root_session_id": "root-integ",
        "child_session_ids": {},
        "orchestrator_task": None,
        "worker_results": [],
        "apply_plan": None,
        "group_outcome": None,
        "step_result_candidate": None,
    }

    dispatch_cmd = await dispatch_node(state, config)
    # Dispatch produced N WorkUnit objects in state-update.
    enriched_units: list[WorkUnit] = dispatch_cmd.update["work_units"]
    coordinator_run_id = dispatch_cmd.update["coordinator_run_id"]

    # Now wire a reducer that returns SUCCESS.
    reducer = AsyncMock()
    reducer.reduce = AsyncMock(return_value=ReducerOutput(
        apply_plan=None,
        group_outcome=GroupOutcome.SUCCESS,
        step_result_candidate="done",
        diagnostics=ReducerDiagnostics(),
    ))
    reducer_config = {
        "configurable": {
            "patch_reducer_service": reducer,
            "event_queue": queue,
        },
    }
    reducer_state = {
        "coordinator_run_id": coordinator_run_id,
        "root_session_id": "root-integ",
        "parent_session_id": "p-integ",
        "user_id": "u-integ",
        "work_units": enriched_units,
        "worker_results": [
            WorkerResult(
                work_unit_id=enriched_units[0].work_unit_id,
                child_session_id="c1",
                outcome=ResultReadyOutcome.SUCCESS,
                cost_summary=CostAggregate(),
                summary="ok",
                patch_manifest=None,
                needs_authorization_details=None,
            ),
            WorkerResult(
                work_unit_id=enriched_units[1].work_unit_id,
                child_session_id="c2",
                outcome=ResultReadyOutcome.SUCCESS,
                cost_summary=CostAggregate(),
                summary="ok",
                patch_manifest=None,
                needs_authorization_details=None,
            ),
        ],
        "quota_acquired": False,
    }
    await reducer_node(reducer_state, reducer_config)

    # Drain queue and validate the 4-event sequence.
    drained = []
    while not queue.empty():
        drained.append(queue.get_nowait())
    assert len(drained) == 4

    # First event is the dispatch.
    assert isinstance(drained[0], CoordinatorDispatchEvent)
    assert drained[0].work_unit_count == 2
    assert drained[0].root_session_id == "root-integ"
    assert drained[0].parent_session_id == "p-integ"

    # Next two are the per-WorkUnit spawn events.
    spawned = drained[1:3]
    assert all(isinstance(ev, CoordinatorWorkerSpawnedEvent) for ev in spawned)
    assert {ev.child_session_id for ev in spawned} == {"c1", "c2"}

    # Last is the reducer event.
    assert isinstance(drained[3], CoordinatorReduceEvent)
    assert drained[3].group_outcome == GroupOutcome.SUCCESS
    assert drained[3].coordinator_run_id == coordinator_run_id
    assert drained[3].root_session_id == "root-integ"
    assert drained[3].parent_session_id == "p-integ"


class _ReconnectHarness:
    def __init__(self, window: str) -> None:
        self.window = window
        self.reached = asyncio.Event()
        self.release = asyncio.Event()
        self.degraded = asyncio.Event()
        self.events: list[BaseEvent] = []
        self.done = False
        self.task: asyncio.Task[None] | None = None
        self.config: dict[str, Any] | None = None
        self.session = Session(
            id="p-reconnect", user_id="u-reconnect", status=SessionStatus.RUNNING,
            execution_mode="foreground", execution_phase="running",
        )
        self.owner: str | None = None
        self.chat_calls: list[str | None] = []
        self.counts = {"start": 0, "dispatch": 0, "reduce": 0,
                       "promote": 0, "resume": 0}
        self.lineage: list[tuple[str, frozenset[str], frozenset[str]]] = []

    def _emit(self, event: BaseEvent) -> None:
        event.id = f"1000-{len(self.events)}"
        self.events.append(event)

    async def get_session(self, session_id: str, **kwargs: Any) -> Session:
        return self.session

    async def _emit_event(self, session_id: str, event: BaseEvent) -> str:
        self._emit(event)
        self.degraded.set()
        return event.id

    async def chat(self, **kwargs: Any) -> AsyncGenerator[BaseEvent, None]:
        latest = kwargs.get("latest_event_id")
        self.chat_calls.append(latest)
        if kwargs.get("message"):
            self._emit(MessageEvent(role="user", message=kwargs["message"]))
            if self.task is None:
                self.counts["start"] += 1
                self.task = asyncio.create_task(self._run())
        cursor = next((i + 1 for i, event in enumerate(self.events)
                       if event.id == latest), 0)
        while True:
            while cursor < len(self.events):
                event = self.events[cursor]
                cursor += 1
                yield event
            if self.done:
                return
            await asyncio.sleep(0)

    @asynccontextmanager
    async def subscriber_scope(self, *, session_id: str, connection_id: str):
        self.owner = connection_id
        try:
            yield SimpleNamespace(is_conflict=False, current_owner=None)
        finally:
            self.owner = None

    async def can_auto_degrade_after_disconnect(self, **kwargs: Any) -> bool:
        return self.owner is None

    @asynccontextmanager
    async def mode_transition_fence(self, *, session_id: str):
        yield

    def new_auto_degrade_cleanup_expiry(self) -> datetime:
        return datetime.now(timezone.utc)

    async def promote(self, **kwargs: Any) -> int:
        self.counts["promote"] += 1
        revision = self.session.execution_revision + 1
        pending = PendingExecutionEvent(payload=ExecutionStatePayload(
            execution_mode="background",
            execution_phase="running",
            transition_reason="auto_degrade_sse_disconnect",
            background_reason="auto_degrade",
            expires_at=kwargs["expires_at"],
            retry_budget_remaining=3,
            execution_revision=revision,
        ))
        self.session = self.session.model_copy(update={
            "execution_mode": "background", "background_reason": "auto_degrade",
            "expires_at": kwargs["expires_at"],
            "execution_revision": revision,
            "pending_execution_event": pending,
        })
        return 3

    async def resume(self, **kwargs: Any) -> None:
        self.counts["resume"] += 1
        if self.session.execution_mode != "background":
            return
        revision = self.session.execution_revision + 1
        pending = PendingExecutionEvent(payload=ExecutionStatePayload(
            execution_mode="foreground",
            execution_phase="running",
            transition_reason="auto_degrade_sse_reconnect",
            retry_budget_remaining=3,
            execution_revision=revision,
        ))
        self.session = self.session.model_copy(update={
            "execution_mode": "foreground", "background_reason": None,
            "expires_at": None,
            "execution_revision": revision,
            "pending_execution_event": pending,
        })

    async def clear_pending_execution_event(
        self, *, session_id: str, execution_revision: int,
    ) -> bool:
        if self.session.execution_revision != execution_revision:
            return False
        self.session = self.session.model_copy(update={"pending_execution_event": None})
        return True

    async def _pause(self, name: str) -> None:
        if self.window == name:
            self.reached.set()
            await self.release.wait()

    def _drain(self, queue: asyncio.Queue[BaseEvent]) -> None:
        while not queue.empty():
            event = queue.get_nowait()
            if isinstance(event, CoordinatorDispatchEvent):
                self.counts["dispatch"] += 1
            elif isinstance(event, CoordinatorReduceEvent):
                self.counts["reduce"] += 1
            self._emit(event)

    async def _run(self) -> None:
        queue: asyncio.Queue[BaseEvent] = asyncio.Queue()
        self.config = _dispatch_config(event_queue=queue)
        state = {
            "step_id": "step-reconnect",
            "work_unit_requests": [WorkUnitRequest(
                objective="explore", phase="exploration", allowed_tools=["file_read"],
            )],
            "parent_session_id": self.session.id,
            "user_id": self.session.user_id, "root_session_id": "root-reconnect",
        }
        try:
            dispatched = await dispatch_node(state, self.config)
            units: list[WorkUnit] = dispatched.update["work_units"]
            run_id = dispatched.update["coordinator_run_id"]
            children: dict[str, str] = dispatched.update["child_session_ids"]
            snapshot = (run_id, frozenset(u.work_unit_id for u in units),
                        frozenset(children.values()))
            self.lineage.append(snapshot)
            self._drain(queue)
            await self._pause("dispatch")
            results = [WorkerResult(
                work_unit_id=u.work_unit_id,
                child_session_id=children[u.work_unit_id],
                outcome=ResultReadyOutcome.SUCCESS, cost_summary=CostAggregate(),
                summary="ok", patch_manifest=None,
                needs_authorization_details=None,
            ) for u in units]
            self.lineage.append((run_id,
                                 frozenset(r.work_unit_id for r in results),
                                 frozenset(r.child_session_id for r in results)))
            await self._pause("child_terminal")
            reducer = AsyncMock()
            reducer.reduce.return_value = ReducerOutput(
                apply_plan=None, group_outcome=GroupOutcome.SUCCESS,
                step_result_candidate="done", diagnostics=ReducerDiagnostics(),
            )
            await reducer_node({
                "coordinator_run_id": run_id, "root_session_id": "root-reconnect",
                "parent_session_id": self.session.id, "user_id": self.session.user_id,
                "work_units": units, "worker_results": results,
                "quota_acquired": False,
            }, {"configurable": {"patch_reducer_service": reducer,
                                  "event_queue": queue}})
            self._drain(queue)
            self.lineage.append(snapshot)
            await self._pause("reduce")
        finally:
            self.done = True


@pytest.mark.integration
@pytest.mark.anyio
@pytest.mark.parametrize("disconnect_window", ["dispatch", "child_terminal", "reduce"])
async def test_sse_disconnect_reconnect_reattaches_same_coordinator_run(
    disconnect_window: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _ReconnectHarness(disconnect_window)

    async def _acquire_connection_limit(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            start_heartbeat=lambda: None,
            release=AsyncMock(),
        )

    monkeypatch.setattr(
        session_routes,
        "acquire_connection_limit",
        _acquire_connection_limit,
    )

    async def _connect(*, connection_id: str, request: ChatRequest):
        return await session_routes.chat(
            session_id="p-reconnect",
            request=request,
            fastapi_request=SimpleNamespace(
                headers={"X-Connection-Id": connection_id},
            ),
            current_user=SimpleNamespace(id="u-reconnect", is_admin=lambda: False),
            agent_service=harness,
            session_service=harness,
            supervisor=harness,
            redis_client=object(),
        )

    first = await _connect(
        connection_id="conn-1",
        request=ChatRequest(message="start coordinator"),
    )
    first_frame = await first.body_iterator.__anext__()
    await asyncio.wait_for(harness.reached.wait(), timeout=1)

    assert first.client_close_handler_callable is not None
    await first.client_close_handler_callable({"type": "http.disconnect"})
    await first.body_iterator.aclose()
    await asyncio.wait_for(harness.degraded.wait(), timeout=1)
    assert harness.session.execution_mode == "background"

    reconnect = await _connect(
        connection_id="conn-2",
        request=ChatRequest(event_id=first_frame.id),
    )
    assert harness.session.execution_mode == "foreground"
    harness.release.set()
    async for _frame in reconnect.body_iterator:
        pass
    assert harness.task is not None
    await asyncio.wait_for(harness.task, timeout=1)

    assert harness.chat_calls == [None, first_frame.id]
    assert harness.counts == {
        "start": 1, "dispatch": 1, "reduce": 1, "promote": 1, "resume": 2,
    }
    assert harness.lineage[0] == harness.lineage[1] == harness.lineage[2]

    assert harness.config is not None
    configurable = harness.config["configurable"]
    assert configurable["session_service"].bump_coordinator_attempt.await_count == 1
    assert configurable["session_service"].create_session_with_parent.await_count == 1
    assert configurable["child_runner_starter"].start.await_count == 1
