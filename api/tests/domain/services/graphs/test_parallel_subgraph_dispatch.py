"""C2 PR-3 §7.5 — parallel_execution_subgraph.dispatch_node unit tests.

Mocks all external collaborators (session_repo, rehydrate_service, parent_sandbox,
artifact_storage, session_service, runner_starter, mailbox_publisher, orchestrator_factory).
"""
from __future__ import annotations

import asyncio
import hashlib
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.domain.models.path_validation import CoordinatorPathContractError
from app.domain.models.work_unit import ProposedPath, WorkUnitRequest
from app.domain.services.graphs.parallel_execution_subgraph import (
    _build_work_units_from_requests,
    dispatch_node,
)


def _expected_wu_ids(step_id: str, attempt_ix: int, count: int) -> list[str]:
    """Mirror ``_build_work_units_from_requests`` id derivation so rehydrate
    tests can populate ``existing.child_session_ids`` with the actual wu_ids
    dispatch_node will compute (post-codex R1 P1 fix: missing wu_ids in
    work_units now raise instead of silently falling through)."""
    hash16 = hashlib.sha256(step_id.encode("utf-8")).hexdigest()[:16]
    return [f"{hash16}.a{attempt_ix}.{i}" for i in range(count)]


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
    """Crash-recovery path: peek == 2, rehydrate returns existing → reuse, no bump.

    [codex R1 P1] existing.child_session_ids MUST cover every wu_id in
    work_units or _rehydrate_dispatch raises (missing-child contract).
    Use _expected_wu_ids to mirror the production hash derivation.
    """
    config = _base_config(peek_returns=2)
    rehydrate = config["configurable"]["rehydrate_service"]
    wu_ids = _expected_wu_ids("step-abc", attempt_ix=2, count=2)
    existing = MagicMock()
    existing.child_session_ids = {wu_ids[0]: "c1", wu_ids[1]: "c2"}
    existing.pending = wu_ids
    existing.terminal = {}
    existing.already_applied = None
    rehydrate.detect_existing_run = AsyncMock(return_value=existing)

    state = _base_state()
    cmd = await dispatch_node(state, config)

    cfg = config["configurable"]
    cfg["session_service"].bump_coordinator_attempt.assert_not_awaited()
    cfg["session_service"].create_session_with_parent.assert_not_called()
    cfg["mailbox_publisher"].publish.assert_not_called()
    # Both wu_ids are pending → 2 Send(worker_node) entries on goto.
    assert len(cmd.goto) == 2


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
    # 2 work units → 2 waiter groups + 2 cancel-listener groups (C2b budget
    # §3-9 R4#1) pre-created, PLUS the orchestrator's observer group hoisted
    # here (finish-core §5.4 G4-min, INV-F4.1).
    assert len(subscribe_calls) == 5
    groups = {call.kwargs["consumer_group"] for call in subscribe_calls}
    # Both waiter groups AND both listener groups present.
    assert {"coordinator:waiter:c1", "coordinator:waiter:c2"} <= groups
    assert {"coordinator:child:c1", "coordinator:child:c2"} <= groups
    # Exactly one ``coordinator:`` group that is neither a waiter nor a
    # cancel-listener group = the orchestrator observer group. Its run_id is
    # generated as ``parent:hash16:a{n}`` by _create_task, so assert
    # structurally rather than hardcoding the hash.
    orch_groups = {
        g for g in groups
        if g.startswith("coordinator:")
        and not g.startswith("coordinator:waiter:")
        and not g.startswith("coordinator:child:")
    }
    assert len(orch_groups) == 1, f"expected one orchestrator group, got {orch_groups}"
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
    fast-publishing child (id=$ excludes already-buffered RESULT_READY).

    [codex R1 P1] wu_ids must match _build_work_units_from_requests output.
    """
    config = _base_config(peek_returns=3)
    wu_ids = _expected_wu_ids("step-abc", attempt_ix=3, count=2)
    existing = MagicMock()
    existing.child_session_ids = {wu_ids[0]: "c1", wu_ids[1]: "c2"}
    existing.pending = wu_ids
    existing.terminal = {}
    existing.already_applied = None
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
    """[r3 P1-1 fix] rehydrate path also fails fast on missing subscriber DI.

    [codex R1 P1] wu_ids match the production derivation; use a
    single-request state so existing.child_session_ids stays minimal.
    """
    config = _base_config(peek_returns=2)
    wu_ids = _expected_wu_ids("step-abc", attempt_ix=2, count=1)
    existing = MagicMock()
    existing.child_session_ids = {wu_ids[0]: "c1"}
    existing.pending = wu_ids
    existing.terminal = {}
    existing.already_applied = None
    config["configurable"]["rehydrate_service"].detect_existing_run = AsyncMock(
        return_value=existing,
    )
    del config["configurable"]["mailbox_subscriber"]
    state = _base_state(work_unit_requests=[
        WorkUnitRequest(
            objective="explore Z", phase="exploration",
            allowed_tools=["file_read"],
        ),
    ])
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
    BEFORE rehydrate ran. id='$' would skip them and waiter would timeout.

    [codex R1 P1] wu_ids must match _build_work_units_from_requests so
    missing-child contract isn't tripped.
    """
    config = _base_config(peek_returns=3)
    wu_ids = _expected_wu_ids("step-abc", attempt_ix=3, count=1)
    existing = MagicMock()
    existing.child_session_ids = {wu_ids[0]: "c1"}
    existing.pending = wu_ids
    existing.terminal = {}
    existing.already_applied = None
    config["configurable"]["rehydrate_service"].detect_existing_run = AsyncMock(
        return_value=existing,
    )
    state = _base_state(work_unit_requests=[
        WorkUnitRequest(
            objective="explore W", phase="exploration",
            allowed_tools=["file_read"],
        ),
    ])
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


@pytest.mark.anyio
async def test_dispatch_threads_parent_sandbox_into_runner_starter_start() -> None:
    """PR-9b-A audit round-1 P1 (Fix 1) — ``_first_time_dispatch`` MUST forward
    ``cfg['parent_sandbox']`` into every ``runner_starter.start(...)`` call.

    ``DefaultCoordinatorChildRunnerStarter.start`` declares ``parent_sandbox``
    as a required kwarg (api/app/application/services/coordinator_child_runner_starter.py:124-135).
    A future drop of this thread would crash first flag-on dispatch with
    ``TypeError: start() missing 1 required keyword-only argument:
    'parent_sandbox'`` BEFORE any SPAWN_REQUEST publishes — this regression
    test fails fast at the call-site instead.
    """
    config = _base_config(peek_returns=None)
    parent_sandbox = config["configurable"]["parent_sandbox"]
    state = _base_state()

    await dispatch_node(state, config)

    starter = config["configurable"]["child_runner_starter"]
    assert starter.start.await_count == 2
    # Every start(...) call must carry the same parent_sandbox instance
    # that was published into cfg by PlannerReActFlow._build_config.
    for call in starter.start.await_args_list:
        kwargs = call.kwargs
        assert "parent_sandbox" in kwargs, (
            "runner_starter.start(...) call missing parent_sandbox kwarg; "
            f"got kwargs={list(kwargs.keys())}"
        )
        assert kwargs["parent_sandbox"] is parent_sandbox, (
            "runner_starter.start(...) parent_sandbox must be the per-run "
            "cfg['parent_sandbox'] (the planner's self._sandbox); "
            f"got {kwargs['parent_sandbox']!r} vs cfg's {parent_sandbox!r}"
        )


# ── C2b budget D9: per-child cancel events + rollback stop ───────────────────


@pytest.mark.anyio
async def test_dispatch_passes_distinct_per_child_cancel_events() -> None:
    """[spec §5-12 pairwise-distinct, R6#2] EVERY starter.start receives its
    OWN fresh Event — pairwise distinct AND distinct from the run-level
    cfg['cancel_event'] (which stays with the orchestrator). Kills the
    'create one child event outside the loop' mutation."""
    config = _base_config(peek_returns=None)
    state = _base_state()

    await dispatch_node(state, config)

    cfg = config["configurable"]
    start_calls = cfg["child_runner_starter"].start.await_args_list
    events = [c.kwargs["cancel_event"] for c in start_calls]
    assert len(events) == 2
    assert events[0] is not events[1], "child events must be pairwise distinct"
    for ev in events:
        assert isinstance(ev, asyncio.Event)
        assert ev is not cfg["cancel_event"], (
            "child must NOT share the run-level event (D9 split)"
        )
    # The orchestrator keeps observing the RUN-LEVEL event (parent cancel).
    orch = cfg["orchestrator_factory"].build.return_value
    assert orch.run.call_args.kwargs["cancel_event"] is cfg["cancel_event"]


@pytest.mark.anyio
async def test_dispatch_precreates_listener_groups() -> None:
    """[spec §3-9 R4#1] dispatch pre-creates the cancel-listener consumer
    group coordinator:child:{sid} (same loop as the waiter hoist) so a
    CANCEL_REQUEST published before the child's listener subscribes is
    retained as group backlog — the pre-subscribe race is closed."""
    config = _base_config(peek_returns=None)
    state = _base_state()

    await dispatch_node(state, config)

    sub = config["configurable"]["mailbox_subscriber"]
    groups = [c.kwargs["consumer_group"] for c in sub.subscribe.await_args_list]
    assert "coordinator:child:c1" in groups
    assert "coordinator:child:c2" in groups
    # Waiter groups still pre-created (unchanged behavior).
    assert "coordinator:waiter:c1" in groups
    assert "coordinator:waiter:c2" in groups


def _three_unit_state() -> dict:
    return _base_state(work_unit_requests=[
        WorkUnitRequest(
            objective=f"explore {i}", phase="exploration",
            allowed_tools=["file_read"],
        )
        for i in range(3)
    ])


def _config_with_quota_and_third_start_failing(call_log: list) -> dict:
    config = _base_config(peek_returns=None)
    cfg = config["configurable"]
    cfg["session_service"].create_session_with_parent = AsyncMock(
        side_effect=[_mk_session("c1"), _mk_session("c2"), _mk_session("c3")],
    )
    cfg["child_runner_starter"].start = AsyncMock(
        side_effect=[None, None, RuntimeError("spawn boom")],
    )
    # Production request_stop_started is SYNC — model it with MagicMock so
    # no stray un-awaited coroutine warnings.
    cfg["child_runner_starter"].request_stop_started = MagicMock(
        side_effect=lambda ids, **kw: call_log.append(("stop", tuple(ids))),
    )
    pq = AsyncMock()
    pq.acquire_coordinator_concurrency = AsyncMock(return_value=True)
    pq.release_coordinator_quotas = AsyncMock(
        side_effect=lambda **kw: call_log.append(("release",)),
    )
    cfg["probe_quota"] = pq
    limits = MagicMock()
    limits.max_work_units_per_run = 5
    limits.max_concurrent_coordinator_runs_per_user = 2
    cfg["coordinator_limits"] = limits
    return config


@pytest.mark.anyio
async def test_dispatch_rollback_stops_started_children() -> None:
    """[spec §5-14, R3#1+R4#2+R6#5] Third start raises → rollback calls
    request_stop_started with EXACTLY the two started ids (c3 never started),
    BEFORE quota release; the original exception propagates; pre-created
    listener groups are destroyed."""
    call_log: list = []
    config = _config_with_quota_and_third_start_failing(call_log)
    state = _three_unit_state()

    with pytest.raises(RuntimeError, match="spawn boom"):
        await dispatch_node(state, config)

    stops = [e for e in call_log if e[0] == "stop"]
    assert stops == [("stop", ("c1", "c2"))], (
        f"must stop exactly the STARTED children, in order; got {stops}"
    )
    # R6#5 order pin: stop strictly precedes quota release.
    assert call_log.index(("stop", ("c1", "c2"))) < call_log.index(("release",))

    # Listener groups (pre-created for all 3) torn down on rollback. No exact
    # destroy-count pin (R5#6: destroy is idempotent; double-destroy benign).
    sub = config["configurable"]["mailbox_subscriber"]
    destroyed = {
        c.kwargs["consumer_group"] for c in sub.destroy_group.await_args_list
    }
    for sid in ("c1", "c2", "c3"):
        assert f"coordinator:child:{sid}" in destroyed


@pytest.mark.anyio
async def test_rollback_stop_survives_release_failure() -> None:
    """[R6#5 半] release_coordinator_quotas raising must not skip the stop
    (it runs FIRST) nor mask the original dispatch exception."""
    call_log: list = []
    config = _config_with_quota_and_third_start_failing(call_log)
    cfg = config["configurable"]
    cfg["probe_quota"].release_coordinator_quotas = AsyncMock(
        side_effect=RuntimeError("redis down"),
    )
    state = _three_unit_state()

    with pytest.raises(RuntimeError, match="spawn boom"):  # NOT "redis down"
        await dispatch_node(state, config)

    assert [e for e in call_log if e[0] == "stop"] == [("stop", ("c1", "c2"))]


# ── C2b rollout WS1b §3.3: dispatch_started_monotonic stamp ──────────────────


@pytest.mark.anyio
async def test_first_time_dispatch_stamps_dispatch_started_monotonic() -> None:
    """[C2b rollout WS1b §3.3] _first_time_dispatch carries a float
    dispatch_started_monotonic in Command(update) — the reducer derives run
    duration from it AND gates run-level metrics on its presence."""
    config = _base_config(peek_returns=None)
    state = _base_state()
    cmd = await dispatch_node(state, config)
    assert "dispatch_started_monotonic" in cmd.update
    assert isinstance(cmd.update["dispatch_started_monotonic"], float)


@pytest.mark.anyio
async def test_rehydrate_dispatch_does_not_stamp_dispatch_started_monotonic() -> None:
    """[C2b rollout WS1b §3.4 double-count guard] _rehydrate_dispatch (crash
    recovery, same coordinator_run_id) must NOT set dispatch_started_monotonic
    — so the reducer records run-level metrics nothing on rehydrate (no
    double-count of the monotonic run_cost_usd counter)."""
    config = _base_config(peek_returns=2)
    rehydrate = config["configurable"]["rehydrate_service"]
    wu_ids = _expected_wu_ids("step-abc", attempt_ix=2, count=2)
    existing = MagicMock()
    existing.child_session_ids = {wu_ids[0]: "c1", wu_ids[1]: "c2"}
    existing.pending = wu_ids
    existing.terminal = {}
    existing.already_applied = None
    rehydrate.detect_existing_run = AsyncMock(return_value=existing)

    state = _base_state()
    cmd = await dispatch_node(state, config)
    assert "dispatch_started_monotonic" not in cmd.update


class TestBuildWorkUnitsPathContract:
    """[single-path contract — lease boundary] ``_build_work_units_from_requests``
    rejects a planner-proposed path whose workspace-relative form lacks a
    directory component, BEFORE any child spawns. The lease keeps the proposed
    path's ORIGINAL form (ChildScopeGate exact-match)."""

    @staticmethod
    def _req(path: str, op: str = "add") -> WorkUnitRequest:
        return WorkUnitRequest(
            objective="o", phase="write", allowed_tools=["file_write"],
            proposed_paths=[ProposedPath(path=path, op=op)],
        )

    def test_bare_relative_proposed_path_rejected(self) -> None:
        with pytest.raises(CoordinatorPathContractError, match="directory"):
            _build_work_units_from_requests([self._req("part_a.md")], "hash16", 1)

    def test_workspace_root_absolute_rejected(self) -> None:
        """``/home/ubuntu/part_a.md`` canonicalizes to bare ``part_a.md`` — the
        §14 live-repro path. Rejected at dispatch, never produces a bare manifest."""
        with pytest.raises(CoordinatorPathContractError, match="directory"):
            _build_work_units_from_requests(
                [self._req("/home/ubuntu/part_a.md")], "hash16", 1,
            )

    def test_directory_qualified_relative_accepted(self) -> None:
        units = _build_work_units_from_requests(
            [self._req("workspace/part_a.md")], "hash16", 1,
        )
        assert units[0].write_lease[0].path == "workspace/part_a.md"

    def test_directory_qualified_absolute_canonicalized_to_relative(self) -> None:
        units = _build_work_units_from_requests(
            [self._req("/home/ubuntu/sub/b.py")], "hash16", 1,
        )
        # [single-path contract] absolute lease allowed (N5) but CANONICALIZED to
        # the one workspace-relative form the manifest + ChildScopeGate agree on,
        # so it can't strand the run mid-flight at strict manifest validation.
        assert units[0].write_lease[0].path == "sub/b.py"

    def test_outside_workspace_rejected(self) -> None:
        with pytest.raises(CoordinatorPathContractError):
            _build_work_units_from_requests(
                [self._req("/etc/passwd", op="modify")], "hash16", 1,
            )

    def test_exploration_request_with_no_paths_unaffected(self) -> None:
        """Exploration requests carry no proposed_paths — nothing to validate."""
        req = WorkUnitRequest(
            objective="explore", phase="exploration", allowed_tools=["file_read"],
        )
        units = _build_work_units_from_requests([req], "hash16", 1)
        assert units[0].write_lease == []
