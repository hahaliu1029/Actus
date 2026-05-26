"""C2 PR-3 §7.5 — parallel_execution_subgraph.dispatch_node unit tests.

Mocks all external collaborators (session_repo, rehydrate_service, parent_sandbox,
artifact_storage, session_service, runner_starter, mailbox_publisher, orchestrator_factory).
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.domain.models.work_unit import ProposedPath, WorkUnitRequest
from app.domain.services.graphs.parallel_execution_subgraph import dispatch_node


def _mk_session(sid: str) -> MagicMock:
    m = MagicMock()
    m.id = sid
    return m


def _base_state(work_unit_requests=None) -> dict:
    return {
        "coordinator_run_id": None,
        "step_id": "step-abc",
        "work_unit_requests": work_unit_requests or [
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
        "parent_session_id": "parent1",
        "user_id": "u1",
        "root_session_id": "root1",
        "child_session_ids": {},
        "orchestrator_task": None,
        "worker_results": [],
        "apply_plan": None,
        "group_outcome": None,
        "step_result_candidate": None,
    }


def _base_config(*, peek_returns: int | None = None) -> dict:
    rehydrate = AsyncMock()
    rehydrate.detect_existing_run = AsyncMock(return_value=None)
    session_service = AsyncMock()
    # [r1 P0-2 fix] peek/bump live on SessionService now (UoW-bounded).
    session_service.peek_coordinator_attempt = AsyncMock(return_value=peek_returns)
    session_service.bump_coordinator_attempt = AsyncMock(return_value=1)
    session_service.create_session_with_parent = AsyncMock(
        side_effect=[_mk_session("c1"), _mk_session("c2")],
    )
    runner_starter = AsyncMock()
    runner_starter.start = AsyncMock()
    publisher = AsyncMock()
    artifact = AsyncMock()
    artifact.put_content_addressed_bytes = AsyncMock(return_value="minio://manifest-ref")
    parent_sandbox = AsyncMock()
    parent_sandbox.compute_digest = AsyncMock(return_value="sha256_abc")
    parent_sandbox.read_file = AsyncMock(return_value=b"content")
    orchestrator = AsyncMock()
    orchestrator.run = AsyncMock()
    orchestrator_factory = MagicMock()
    orchestrator_factory.build = MagicMock(return_value=orchestrator)
    # [r1 P0-1 fix] subscriber is required for waiter-group pre-creation.
    subscriber = AsyncMock()
    subscriber.subscribe = AsyncMock()
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
        }
    }


@pytest.mark.anyio
async def test_first_time_dispatch_bumps_and_spawns_n_children() -> None:
    config = _base_config(peek_returns=None)
    state = _base_state()

    cmd = await dispatch_node(state, config)

    cfg = config["configurable"]
    cfg["session_service"].bump_coordinator_attempt.assert_awaited_once()
    assert cfg["session_service"].create_session_with_parent.await_count == 2
    assert cfg["mailbox_publisher"].publish.await_count == 2
    assert cfg["child_runner_starter"].start.await_count == 2
    cfg["orchestrator_factory"].build.assert_called_once()
    assert len(cmd.goto) == 2


@pytest.mark.anyio
async def test_peek_returning_none_triggers_bump() -> None:
    config = _base_config(peek_returns=None)
    state = _base_state()
    await dispatch_node(state, config)
    config["configurable"]["session_service"].peek_coordinator_attempt.assert_awaited_once()
    config["configurable"]["session_service"].bump_coordinator_attempt.assert_awaited_once()


@pytest.mark.anyio
async def test_peek_returning_attempt_with_no_rehydrate_falls_back_to_bump() -> None:
    """peek == 2 but rehydrate returns None → must bump fresh attempt."""
    config = _base_config(peek_returns=2)
    config["configurable"]["session_service"].bump_coordinator_attempt = AsyncMock(return_value=3)
    state = _base_state()
    await dispatch_node(state, config)
    config["configurable"]["rehydrate_service"].detect_existing_run.assert_awaited_once()
    config["configurable"]["session_service"].bump_coordinator_attempt.assert_awaited_once()


@pytest.mark.anyio
async def test_peek_then_rehydrate_found_skips_bump_and_spawn() -> None:
    """Crash-recovery path: peek == 2, rehydrate returns existing → reuse, no bump."""
    config = _base_config(peek_returns=2)
    rehydrate = config["configurable"]["rehydrate_service"]
    existing = MagicMock()
    existing.child_session_ids = {"wu_id_unused": "c1"}
    existing.pending = ["wu_id_unused"]
    rehydrate.detect_existing_run = AsyncMock(return_value=existing)

    state = _base_state()
    cmd = await dispatch_node(state, config)

    cfg = config["configurable"]
    cfg["session_service"].bump_coordinator_attempt.assert_not_awaited()
    cfg["session_service"].create_session_with_parent.assert_not_called()
    cfg["mailbox_publisher"].publish.assert_not_called()
    assert len(cmd.goto) == 1


@pytest.mark.anyio
async def test_dispatch_seeds_modify_lease_with_digest_and_content() -> None:
    config = _base_config(peek_returns=None)
    config["configurable"]["session_service"].create_session_with_parent = AsyncMock(
        side_effect=[_mk_session("c1")],
    )
    state = _base_state(work_unit_requests=[
        WorkUnitRequest(
            objective="patch X", phase="write",
            allowed_tools=["file_write"],
            proposed_paths=[ProposedPath(path="api/foo.py", op="modify")],
        ),
    ])
    await dispatch_node(state, config)
    cfg = config["configurable"]
    cfg["parent_sandbox"].compute_digest.assert_awaited_once_with("api/foo.py")
    cfg["parent_sandbox"].read_file.assert_awaited_once_with("api/foo.py")
    assert cfg["artifact_storage"].put_content_addressed_bytes.await_count >= 2


@pytest.mark.anyio
async def test_dispatch_skips_seed_upload_for_add_op() -> None:
    """op=add must keep base_digest + seed_content_ref None per PathLease invariant."""
    config = _base_config(peek_returns=None)
    config["configurable"]["session_service"].create_session_with_parent = AsyncMock(
        side_effect=[_mk_session("c1")],
    )
    state = _base_state(work_unit_requests=[
        WorkUnitRequest(
            objective="new file", phase="write",
            allowed_tools=["file_write"],
            proposed_paths=[ProposedPath(path="api/new.py", op="add")],
        ),
    ])
    await dispatch_node(state, config)
    cfg = config["configurable"]
    cfg["parent_sandbox"].compute_digest.assert_not_awaited()
    cfg["parent_sandbox"].read_file.assert_not_awaited()


@pytest.mark.anyio
async def test_dispatch_raises_when_modify_target_missing_in_parent_sandbox() -> None:
    config = _base_config(peek_returns=None)
    config["configurable"]["parent_sandbox"].compute_digest = AsyncMock(return_value=None)
    state = _base_state(work_unit_requests=[
        WorkUnitRequest(
            objective="patch missing", phase="write",
            allowed_tools=["file_write"],
            proposed_paths=[ProposedPath(path="api/missing.py", op="modify")],
        ),
    ])
    with pytest.raises(ValueError) as ei:
        await dispatch_node(state, config)
    assert "missing.py" in str(ei.value)


@pytest.mark.anyio
async def test_waiter_consumer_group_pre_created_before_runner_start() -> None:
    """[r1 P0-1 fix] Pre-creating the waiter group BEFORE runner.start closes
    the fast-publish race where a child could emit RESULT_READY before the
    lazy waiter.subscribe ever runs."""
    config = _base_config(peek_returns=None)
    state = _base_state()
    await dispatch_node(state, config)
    cfg = config["configurable"]
    subscribe_calls = cfg["mailbox_subscriber"].subscribe.await_args_list
    # 2 work units → 2 waiter groups pre-created.
    assert len(subscribe_calls) == 2
    groups = {call.kwargs["consumer_group"] for call in subscribe_calls}
    assert groups == {"coordinator:waiter:c1", "coordinator:waiter:c2"}
    for call in subscribe_calls:
        assert call.kwargs["stream_key"] == "actus:child:root1:mailbox"


@pytest.mark.anyio
async def test_subscriber_subscribe_runs_before_runner_start() -> None:
    """Pre-creation must happen BEFORE runner.start. Use a callable that records ordering."""
    config = _base_config(peek_returns=None)
    order: list[str] = []
    subscriber = config["configurable"]["mailbox_subscriber"]
    runner_starter = config["configurable"]["child_runner_starter"]

    async def record_subscribe(**kwargs):
        order.append(f"subscribe:{kwargs['consumer_group']}")

    async def record_start(**kwargs):
        order.append(f"start:{kwargs['child_session_id']}")

    subscriber.subscribe = AsyncMock(side_effect=record_subscribe)
    runner_starter.start = AsyncMock(side_effect=record_start)

    state = _base_state()
    await dispatch_node(state, config)
    # Both subscribe events must precede any start event.
    first_start_idx = next(i for i, e in enumerate(order) if e.startswith("start:"))
    last_subscribe_idx = max(i for i, e in enumerate(order) if e.startswith("subscribe:"))
    assert last_subscribe_idx < first_start_idx, (
        f"runner started before all waiter groups subscribed: order={order}"
    )


@pytest.mark.anyio
async def test_dispatch_fails_fast_when_mailbox_subscriber_missing_from_cfg() -> None:
    """[r2 P1-2 fix] mailbox_subscriber is REQUIRED. Earlier ``cfg.get`` silently
    skipped pre-creation, re-opening the fast-publish race when DI forgot to
    inject the subscriber. The fix uses ``cfg[...]`` so missing DI raises KeyError."""
    config = _base_config(peek_returns=None)
    # Strip mailbox_subscriber to simulate forgotten DI.
    del config["configurable"]["mailbox_subscriber"]
    state = _base_state()
    with pytest.raises(KeyError) as ei:
        await dispatch_node(state, config)
    assert "mailbox_subscriber" in str(ei.value)


@pytest.mark.anyio
async def test_bump_invoked_via_session_service_not_repository() -> None:
    """[r1 P0-2 fix] peek/bump go through SessionService so the JSONB UPDATE
    commits via UoW BEFORE child sessions are created. dispatch_node MUST NOT
    read a raw ``session_repository`` for this work."""
    config = _base_config(peek_returns=None)
    state = _base_state()
    await dispatch_node(state, config)
    # session_service was the one called.
    config["configurable"]["session_service"].bump_coordinator_attempt.assert_awaited_once()
    # session_repository is intentionally NOT in cfg anymore.
    assert "session_repository" not in config["configurable"]


@pytest.mark.anyio
async def test_rehydrate_pre_creates_waiter_group_for_each_pending_child() -> None:
    """[r3 P1-1 fix] _rehydrate_dispatch must pre-create the waiter consumer
    group for each pending child BEFORE returning Send to worker_node. After
    pod restart the previous waiter group is gone and the lazy
    waiter.subscribe inside worker_node.await_terminal would race the
    fast-publishing child (id=$ excludes already-buffered RESULT_READY)."""
    config = _base_config(peek_returns=3)
    existing = MagicMock()
    existing.child_session_ids = {"wu1": "c1", "wu2": "c2"}
    existing.pending = ["wu1", "wu2"]
    config["configurable"]["rehydrate_service"].detect_existing_run = AsyncMock(
        return_value=existing,
    )
    state = _base_state()
    await dispatch_node(state, config)

    subscriber = config["configurable"]["mailbox_subscriber"]
    subscribe_calls = subscriber.subscribe.await_args_list
    groups = {call.kwargs["consumer_group"] for call in subscribe_calls}
    # 2 pending children → 2 waiter groups pre-created on rehydrate path.
    assert {"coordinator:waiter:c1", "coordinator:waiter:c2"}.issubset(groups), (
        f"rehydrate did not pre-create waiter groups; got {groups}"
    )


@pytest.mark.anyio
async def test_rehydrate_fails_fast_when_mailbox_subscriber_missing_from_cfg() -> None:
    """[r3 P1-1 fix] rehydrate path also fails fast on missing subscriber DI."""
    config = _base_config(peek_returns=2)
    existing = MagicMock()
    existing.child_session_ids = {"wu1": "c1"}
    existing.pending = ["wu1"]
    config["configurable"]["rehydrate_service"].detect_existing_run = AsyncMock(
        return_value=existing,
    )
    del config["configurable"]["mailbox_subscriber"]
    state = _base_state()
    with pytest.raises(KeyError) as ei:
        await dispatch_node(state, config)
    assert "mailbox_subscriber" in str(ei.value)


@pytest.mark.anyio
async def test_orchestrator_factory_build_receives_parent_session_id() -> None:
    """[r3 P1-2 fix] dispatch must pass parent_session_id to
    orchestrator_factory.build(...) so the orchestrator's CANCEL_REQUEST
    envelopes carry the real parent (not '')."""
    config = _base_config(peek_returns=None)
    state = _base_state()
    await dispatch_node(state, config)
    factory = config["configurable"]["orchestrator_factory"]
    factory.build.assert_called_once()
    build_kwargs = factory.build.call_args.kwargs
    assert build_kwargs["parent_session_id"] == "parent1"
    assert build_kwargs["root_session_id"] == "root1"
    assert build_kwargs["coordinator_run_id"].startswith("parent1:")


@pytest.mark.anyio
async def test_rehydrate_subscribes_with_start_id_zero_to_read_backlog() -> None:
    """[r4 P1-1 fix] rehydrate subscribe must use start_id='0' so the new
    consumer group reads terminal envelopes already buffered in the stream
    BEFORE rehydrate ran. id='$' would skip them and waiter would timeout."""
    config = _base_config(peek_returns=3)
    existing = MagicMock()
    existing.child_session_ids = {"wu1": "c1"}
    existing.pending = ["wu1"]
    config["configurable"]["rehydrate_service"].detect_existing_run = AsyncMock(
        return_value=existing,
    )
    state = _base_state()
    await dispatch_node(state, config)
    sub = config["configurable"]["mailbox_subscriber"]
    rehydrate_subscribe_call = sub.subscribe.await_args_list[0]
    assert rehydrate_subscribe_call.kwargs.get("start_id") == "0", (
        "rehydrate must pass start_id='0' to consume already-buffered terminals; "
        f"got kwargs={rehydrate_subscribe_call.kwargs}"
    )


@pytest.mark.anyio
async def test_first_time_dispatch_subscribes_with_default_dollar_start_id() -> None:
    """[r4 P1-1 fix] First-time dispatch keeps default start_id='$' (no
    pre-buffered terminals possible — child hasn't been started)."""
    config = _base_config(peek_returns=None)
    state = _base_state()
    await dispatch_node(state, config)
    sub = config["configurable"]["mailbox_subscriber"]
    for call in sub.subscribe.await_args_list:
        # Default '$' applies — either kwarg absent or explicitly '$'.
        assert call.kwargs.get("start_id", "$") == "$"


@pytest.mark.anyio
async def test_runner_starter_receives_parent_session_id() -> None:
    """[r4 P1-2 fix] runner_starter.start must receive parent_session_id so
    PR-4 finalizers can build envelopes without parsing coordinator_run_id."""
    config = _base_config(peek_returns=None)
    state = _base_state()
    await dispatch_node(state, config)
    runner = config["configurable"]["child_runner_starter"]
    for call in runner.start.await_args_list:
        assert call.kwargs["parent_session_id"] == "parent1", (
            f"runner.start missing parent_session_id; got {call.kwargs}"
        )


@pytest.mark.anyio
async def test_create_session_with_parent_receives_coordinator_lineage_kwargs() -> None:
    """[r6 P2-3 fix] dispatch_node must pass coordinator_run_id + work_unit_id
    to create_session_with_parent so the partial unique index
    ``ux_sessions_coordinator_wu`` actually has fields to enforce. Without
    these kwargs the idempotency column protection is silently void."""
    config = _base_config(peek_returns=None)
    state = _base_state()
    await dispatch_node(state, config)
    create = config["configurable"]["session_service"].create_session_with_parent
    assert create.await_count == 2
    for call in create.await_args_list:
        assert call.kwargs.get("coordinator_run_id"), (
            f"create_session_with_parent missing coordinator_run_id: {call.kwargs}"
        )
        assert call.kwargs.get("work_unit_id"), (
            f"create_session_with_parent missing work_unit_id: {call.kwargs}"
        )
        # coordinator_run_id has the parent:hash16:a{n} shape.
        assert call.kwargs["coordinator_run_id"].startswith("parent1:")
        # work_unit_id has the hash16.a{n}.{i} shape.
        assert ".a1." in call.kwargs["work_unit_id"]


@pytest.mark.anyio
async def test_orchestrator_task_done_callback_silent_on_cancelled(caplog) -> None:
    """[r7 P2 fix] _log_orchestrator_task_done must be a no-op when the task
    was cancelled (vs. raised). Otherwise pod-shutdown cancellations would
    spam the error log with false-positive 'orchestrator died' entries."""
    import logging
    from app.domain.services.graphs.parallel_execution_subgraph import (
        _log_orchestrator_task_done,
    )

    async def long_running():
        await asyncio.sleep(60)

    task = asyncio.create_task(long_running())
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    caplog.set_level(logging.ERROR, logger="app.domain.services.graphs.parallel_execution_subgraph")
    # Must not raise + must not log.
    _log_orchestrator_task_done(task)
    assert not any(
        "orchestrator task died" in rec.message.lower()
        for rec in caplog.records
    ), f"cancelled task should be silent; got {[r.message for r in caplog.records]}"


@pytest.mark.anyio
async def test_orchestrator_task_done_callback_surfaces_unhandled_exception(caplog) -> None:
    """[r6 P1-1 fix] If orchestrator.run raises (e.g. r5 P1-2 all-publish-failed),
    the exception must be logged via done_callback — not silently swallowed."""
    import logging
    config = _base_config(peek_returns=None)
    boom_orchestrator = AsyncMock()
    boom_orchestrator.run = AsyncMock(side_effect=RuntimeError("orchestrator blew up"))
    config["configurable"]["orchestrator_factory"].build = MagicMock(return_value=boom_orchestrator)
    state = _base_state()
    caplog.set_level(logging.ERROR, logger="app.domain.services.graphs.parallel_execution_subgraph")
    await dispatch_node(state, config)
    # Let the background task fail.
    await asyncio.sleep(0.05)
    assert any(
        "orchestrator task died with unhandled exception" in rec.message.lower()
        for rec in caplog.records
    ), f"expected orchestrator fatal log; got {[r.message for r in caplog.records]}"


@pytest.mark.anyio
async def test_coordinator_run_id_format_is_session_hash_attempt() -> None:
    config = _base_config(peek_returns=None)
    config["configurable"]["session_service"].bump_coordinator_attempt = AsyncMock(return_value=7)
    state = _base_state()
    cmd = await dispatch_node(state, config)
    run_id = cmd.update["coordinator_run_id"]
    assert run_id.startswith("parent1:")
    assert run_id.endswith(":a7")
    parts = run_id.split(":")
    assert len(parts[1]) == 16
