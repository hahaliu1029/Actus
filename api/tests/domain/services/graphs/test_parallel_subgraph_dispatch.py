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
from app.domain.models.work_unit import ProposedPath, ProposedTree, WorkUnitRequest
from app.domain.services.graphs.parallel_execution_subgraph import (
    _build_work_units_from_requests,
    _coerce_units_typed_only_if_flag_off,
    _reject_cross_unit_tree_overlap,
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


@pytest.mark.anyio  # this file marks each async test individually (no module-level pytestmark)
async def test_first_time_send_payload_carries_manifest_required() -> None:
    write_req = WorkUnitRequest(
        objective="write X",
        phase="write",
        allowed_tools=["file_write"],
        proposed_paths=[ProposedPath(path="workspace/a.py", op="add")],
    )
    explore_req = WorkUnitRequest(
        objective="explore Y",
        phase="exploration",
        allowed_tools=["file_read"],
    )
    state = _base_state(work_unit_requests=[write_req, explore_req])
    config = _base_config(peek_returns=None)
    cmd = await dispatch_node(state, config)
    # cmd.goto is a list of Send; each .arg dict must carry manifest_required.
    by_required = {
        send.arg["work_unit_id"]: send.arg["manifest_required"]
        for send in cmd.goto
    }
    # exactly one write-phase wu (manifest_required=True) + one exploration (False)
    assert sorted(by_required.values()) == [False, True]


@pytest.mark.anyio
async def test_rehydrate_send_payload_carries_manifest_required() -> None:
    """[S2 §3.2 R3-F1] The rehydrate dispatch builds its own Send payload
    (parallel_execution_subgraph.py:1236) — assert it ALSO carries
    manifest_required, keyed off the rehydrated WorkUnit's phase. Drives the
    rehydrate branch via detect_existing_run returning a pending run (mirrors
    the file's existing rehydrate tests, e.g.
    test_peek_then_rehydrate_found_skips_bump_and_spawn)."""
    # Two requests: index 0 = write-phase, index 1 = exploration. wu_ids are
    # derived positionally as f"{hash16}.a{attempt}.{i}" (research:
    # _build_work_units_from_requests :206), so wu_ids[0] is the write unit.
    write_req = WorkUnitRequest(
        objective="write X",
        phase="write",
        allowed_tools=["file_write"],
        proposed_paths=[ProposedPath(path="workspace/a.py", op="add")],
    )
    explore_req = WorkUnitRequest(
        objective="explore Y",
        phase="exploration",
        allowed_tools=["file_read"],
    )
    config = _base_config(peek_returns=3)
    wu_ids = _expected_wu_ids("step-abc", attempt_ix=3, count=2)
    existing = MagicMock()
    existing.child_session_ids = {wu_ids[0]: "c1", wu_ids[1]: "c2"}
    existing.pending = wu_ids
    existing.terminal = {}  # no terminal envelopes ⇒ all pending fan out as Sends
    existing.already_applied = None
    config["configurable"]["rehydrate_service"].detect_existing_run = AsyncMock(
        return_value=existing,
    )
    state = _base_state(work_unit_requests=[write_req, explore_req])
    cmd = await dispatch_node(state, config)
    # Rehydrate fan-out returns the pending Sends as cmd.goto.
    by_required = {
        send.arg["work_unit_id"]: send.arg["manifest_required"]
        for send in cmd.goto
    }
    assert by_required[wu_ids[0]] is True   # write-phase wu ⇒ manifest required
    assert by_required[wu_ids[1]] is False  # exploration wu ⇒ not required


@pytest.mark.anyio
async def test_rehydrate_unexpected_terminal_wu_id_fails_closed() -> None:
    """[codex PR-2 R1 P1] A persisted terminal envelope keyed by a wu_id NOT in
    the current attempt's work_units must fail-closed at the dispatcher BEFORE
    the builder — never flowing a stale SUCCESS into the reducer.

    ``CoordinatorRehydrateService.detect_existing_run`` builds ``terminal`` from
    EVERY persisted envelope for the (attempt-scoped) ``coordinator_run_id``
    with no filtering against the rebuilt ``work_units``. A wu_id in
    ``terminal`` but outside ``expected_wu_ids`` means the persisted run shape
    no longer matches the current plan (state ``work_unit_requests`` changed /
    unit count shrank between dispatch and rehydrate). Left unguarded,
    ``_build_pre_results_from_terminal`` computes ``manifest_required`` from
    ``work_units_by_id.get(wu_id)`` → ``None`` for the unknown id → NO
    demotion, and a self-consistent SUCCESS manifest would pass the reducer's
    lineage cross-check (matched against the worker_result's own wu_id, not the
    expected set) and inject out-of-plan writes into the apply plan. Mirror the
    Step 6 (missing) / limbo guards: raise so the orchestrator surfaces an
    operator-actionable error instead of a silent fail-open.
    """
    config = _base_config(peek_returns=3)
    wu_ids = _expected_wu_ids("step-abc", attempt_ix=3, count=2)
    existing = MagicMock()
    # Both expected children present & pending ⇒ no missing-child, no limbo.
    existing.child_session_ids = {wu_ids[0]: "c1", wu_ids[1]: "c2"}
    existing.pending = wu_ids
    # ... plus ONE stale terminal keyed by a wu_id NOT in the current plan.
    stale = MagicMock()
    stale.envelope_type = "RESULT_READY"
    stale.payload = {"outcome": "success"}
    stale.child_session_id = "c-stale"
    existing.terminal = {"deadbeefdeadbeef.a3.9": stale}
    existing.already_applied = None
    config["configurable"]["rehydrate_service"].detect_existing_run = AsyncMock(
        return_value=existing,
    )
    state = _base_state()
    with pytest.raises(RuntimeError, match="unexpected terminal"):
        await dispatch_node(state, config)


@pytest.mark.anyio
async def test_rehydrate_unexpected_pending_wu_id_fails_closed() -> None:
    """[codex PR-2 R1 P1 — opus-review follow-up] Symmetric to the terminal
    guard: an out-of-plan wu_id that is still PENDING (a child row exists for a
    wu_id NOT in the current attempt's work_units) must ALSO fail-closed.

    Without the union guard, Step 8 (``pending_sends``) would ``Send`` the
    out-of-plan id to ``worker_node`` with
    ``manifest_required = _phase_by_wu_id.get(wu_id) == "write"`` → ``None ==
    "write"`` → ``False`` ⇒ no write-phase demotion, and the child's
    self-consistent SUCCESS manifest would then fold into the apply plan via the
    reducer's wu_id-self-referential lineage check — the identical out-of-plan
    write injection the terminal guard closes. The guard rejects
    ``(terminal ∪ pending) - expected_wu_ids`` so BOTH routes fail loudly.
    """
    config = _base_config(peek_returns=3)
    wu_ids = _expected_wu_ids("step-abc", attempt_ix=3, count=2)
    oob = "cafebabecafebabe.a3.7"  # out-of-plan: not produced by this attempt
    existing = MagicMock()
    # Realistic shape: pending ⊆ children. Expected children present ⇒ no
    # missing-child; nothing in terminal & all in pending ⇒ no limbo.
    existing.child_session_ids = {wu_ids[0]: "c1", wu_ids[1]: "c2", oob: "c3"}
    existing.pending = [wu_ids[0], wu_ids[1], oob]
    existing.terminal = {}
    existing.already_applied = None
    config["configurable"]["rehydrate_service"].detect_existing_run = AsyncMock(
        return_value=existing,
    )
    state = _base_state()
    with pytest.raises(RuntimeError, match="unexpected terminal/pending"):
        await dispatch_node(state, config)


@pytest.mark.anyio
async def test_rehydrate_terminal_child_session_mismatch_fails_closed() -> None:
    """[codex PR-2 R2 P1] A terminal envelope for an EXPECTED wu_id whose
    child_session_id differs from the current child must fail-closed.

    Terminal records are keyed by work_unit_id only, and (coordinator_run_id,
    work_unit_id) is NOT yet unique on the sessions table (partial-unique
    deferred to PR-7 — see the missing-child guard). A stale/duplicate child row
    can therefore surface a terminal envelope produced by a DIFFERENT child than
    the current ``child_session_ids[wu_id]`` (the newest child by created_at).
    Left unguarded, ``_build_pre_results_from_terminal`` would turn that stale
    envelope into the WorkerResult for the expected wu_id (and ``pending`` skips
    the REAL current child), so a stale-child SUCCESS manifest folds into the
    apply plan while the live child's work is dropped — the reducer's lineage
    check validates run_id + wu_id but NOT child_session_id. Fail closed.
    """
    config = _base_config(peek_returns=3)
    wu_ids = _expected_wu_ids("step-abc", attempt_ix=3, count=2)
    existing = MagicMock()
    # Current children for both expected wu_ids ⇒ no missing-child.
    existing.child_session_ids = {wu_ids[0]: "c1-current", wu_ids[1]: "c2"}
    existing.pending = [wu_ids[1]]  # wu1 still pending; wu0 has a terminal
    # wu0's terminal was produced by a STALE child (≠ "c1-current").
    stale = MagicMock()
    stale.envelope_type = "RESULT_READY"
    stale.payload = {"outcome": "success"}
    stale.child_session_id = "c1-STALE"
    existing.terminal = {wu_ids[0]: stale}  # in-plan id, wrong child
    existing.already_applied = None
    config["configurable"]["rehydrate_service"].detect_existing_run = AsyncMock(
        return_value=existing,
    )
    state = _base_state()
    with pytest.raises(RuntimeError, match="produced by child"):
        await dispatch_node(state, config)




# ── S2 PR-3 Task 3.6: tree-lease build/enrich + overlap + flag-off coercion ──


def _tree_req(prefix: str) -> WorkUnitRequest:
    return WorkUnitRequest(
        objective="o", phase="write", allowed_tools=["file_write"],
        proposed_paths=[],
        proposed_trees=[ProposedTree(prefix=prefix, ops=frozenset({"add"}))],
    )


def _path_req(path: str, op: str = "add") -> WorkUnitRequest:
    return WorkUnitRequest(
        objective="o", phase="write", allowed_tools=["file_write"],
        proposed_paths=[ProposedPath(path=path, op=op)],
    )


class TestBuildWorkUnitsTreeLease:
    def test_build_sets_tree_lease_and_shell_mode(self):
        units = _build_work_units_from_requests([_tree_req("workspace")], "h16", 1)
        assert units[0].write_tree_lease[0].prefix == "workspace"
        assert units[0].shell_mode is True

    def test_build_canonicalizes_tree_prefix(self):
        units = _build_work_units_from_requests([_tree_req("./workspace/")], "h16", 1)
        assert units[0].write_tree_lease[0].prefix == "workspace"

    def test_build_path_only_keeps_shell_mode_false(self):
        units = _build_work_units_from_requests([_path_req("d/a.py")], "h16", 1)
        assert units[0].shell_mode is False
        assert units[0].write_tree_lease == []

    def test_build_bad_tree_prefix_rejected(self):
        # [deviation from plan-verbatim, justified by Task 3.2 design]
        # Task 3.2 places ``validate_coordinator_tree_prefix`` on the
        # ``ProposedTree.prefix`` field as a Pydantic ``AfterValidator`` (it
        # mirrors ``ProposedPath``). So a bad prefix like "/etc" is rejected at
        # ``ProposedTree`` CONSTRUCTION (inside ``_tree_req``) — BEFORE
        # ``_build_work_units_from_requests`` ever runs. Pydantic re-wraps the
        # ``CoordinatorPathContractError`` raised by the validator into a
        # ``pydantic.ValidationError`` (both are ``ValueError`` subclasses). The
        # test's intent — a bad tree prefix is rejected before any child spawns
        # — holds; it is just rejected one layer earlier than the plan's test
        # body assumed (raw ``CoordinatorPathContractError`` out of the build).
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            _build_work_units_from_requests([_tree_req("/etc")], "h16", 1)

    def test_build_shell_mode_request_with_paths_only_no_tree_lease(self):
        # P0: a legit shell-mode unit with ONLY exact file leases (shell_mode
        # requested positively, NO proposed_trees) must build shell_mode=True
        # with an EMPTY write_tree_lease. Before the fix the build derived
        # shell_mode=bool(tree_leases), so this unit could NEVER activate shell
        # mode (shell_mode stuck False). Tree leases are SUFFICIENT, not
        # NECESSARY, for shell mode.
        req = WorkUnitRequest(
            objective="o", phase="write", allowed_tools=["file_write"],
            proposed_paths=[ProposedPath(path="d/a.py", op="add")],
            proposed_trees=[],
            shell_mode=True,
        )
        units = _build_work_units_from_requests([req], "h16", 1)
        assert units[0].shell_mode is True
        assert units[0].write_tree_lease == []
        # the exact file lease survives so the built unit is a valid write unit.
        assert [l.path for l in units[0].write_lease] == ["d/a.py"]


class TestEnrichmentPreservesShellSignals:
    """[S2 §3.3/§3.5 — P0-1] The Step-4 enrichment loop in dispatch rebuilds
    every WorkUnit. It MUST carry shell_mode + write_tree_lease through the
    rebuild, otherwise the signals are stripped before _serialize_spawn_manifest
    and never reach the child (PR-4/PR-5 stay dead even flag-on). We observe the
    rebuilt unit indirectly through the serialized manifest bytes uploaded for
    that unit (the enriched unit is what _serialize_spawn_manifest sees).
    """

    @pytest.mark.anyio
    async def test_dispatch_enriched_manifest_preserves_shell_mode_and_tree_lease(
        self, monkeypatch,
    ) -> None:
        import json as _json

        from app.domain.services.graphs import (
            parallel_execution_subgraph as _peg,
        )

        # Flag ON so the flag-off coercion (F27) does NOT strip the signals —
        # we are isolating the enrichment rebuild, not the coercion.
        monkeypatch.setenv("ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED", "true")

        config = _base_config(peek_returns=None)
        config["configurable"]["session_service"].create_session_with_parent = (
            AsyncMock(side_effect=[_mk_session("c1")])
        )
        # Capture every manifest upload (filename="manifest.json") so we can
        # decode the bytes the serializer produced for the enriched unit.
        manifests: list[bytes] = []

        async def _capture_put(*, prefix, content, filename=None):
            if filename == "manifest.json":
                manifests.append(content)
            return "minio://manifest-ref"

        config["configurable"]["artifact_storage"].put_content_addressed_bytes = (
            AsyncMock(side_effect=_capture_put)
        )

        state = _base_state(work_unit_requests=[
            WorkUnitRequest(
                objective="shell-write", phase="write",
                allowed_tools=["file_write"],
                proposed_paths=[],
                proposed_trees=[
                    ProposedTree(prefix="workspace", ops=frozenset({"add"})),
                ],
            ),
        ])
        await _peg.dispatch_node(state, config)

        assert manifests, "no manifest.json uploaded"
        data = _json.loads(manifests[0])
        # With the P0-1 bug present (enrichment omits the fields), these would be
        # shell_mode False / write_tree_lease [] -> this test FAILS.
        assert data["shell_mode"] is True
        assert data["write_tree_lease"] == [{"prefix": "workspace", "ops": ["add"]}]


class TestCrossUnitTreeOverlap:
    def test_file_lease_under_other_units_tree_rejected(self):
        # F24: unit A leases tree "workspace"; unit B leases file
        # "workspace/x.py" -> overlap -> reject.
        units = _build_work_units_from_requests(
            [_tree_req("workspace"), _path_req("workspace/x.py")], "h16", 1,
        )
        with pytest.raises(CoordinatorPathContractError, match="overlap"):
            _reject_cross_unit_tree_overlap(units)

    def test_tree_under_other_units_tree_rejected(self):
        units = _build_work_units_from_requests(
            [_tree_req("api"), _tree_req("api/gen")], "h16", 1,
        )
        with pytest.raises(CoordinatorPathContractError, match="overlap"):
            _reject_cross_unit_tree_overlap(units)

    def test_identical_tree_prefix_rejected(self):
        # F24 + §3.3/§159: two units leasing the SAME prefix is a double-grant.
        # tree_contains returns False for equal paths, so a bare
        # tree_contains(a, b) check would WRONGLY accept this — the overlap
        # helper must special-case equality.
        units = _build_work_units_from_requests(
            [_tree_req("workspace"), _tree_req("workspace")], "h16", 1,
        )
        with pytest.raises(CoordinatorPathContractError, match="overlap"):
            _reject_cross_unit_tree_overlap(units)

    def test_disjoint_trees_ok(self):
        units = _build_work_units_from_requests(
            [_tree_req("workspace"), _tree_req("api")], "h16", 1,
        )
        _reject_cross_unit_tree_overlap(units)  # no raise

    def test_same_unit_file_under_own_tree_ok(self):
        # A unit's own file lease under its own tree is not cross-unit overlap.
        unit_req = WorkUnitRequest(
            objective="o", phase="write", allowed_tools=["file_write"],
            proposed_paths=[ProposedPath(path="workspace/x.py", op="add")],
            proposed_trees=[ProposedTree(prefix="workspace", ops=frozenset({"add"}))],
        )
        units = _build_work_units_from_requests([unit_req], "h16", 1)
        _reject_cross_unit_tree_overlap(units)  # no raise


class TestFlagOffFailClosedCoercion:
    def test_flag_off_coerces_shell_mode_unit_to_typed_only(self, monkeypatch):
        # F27: master flag OFF -> any unit with shell_mode/tree lease is coerced
        # back to typed-only (shell_mode False, write_tree_lease []).
        monkeypatch.delenv("ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED", raising=False)
        units = _build_work_units_from_requests(
            [WorkUnitRequest(
                objective="o", phase="write", allowed_tools=["file_write"],
                proposed_paths=[ProposedPath(path="workspace/x.py", op="add")],
                proposed_trees=[ProposedTree(prefix="workspace", ops=frozenset({"add"}))],
            )],
            "h16", 1,
        )
        assert units[0].shell_mode is True  # built with the signal
        coerced = _coerce_units_typed_only_if_flag_off(units)
        assert coerced[0].shell_mode is False
        assert coerced[0].write_tree_lease == []
        # the path lease survives so the unit is still a valid write unit.
        assert [l.path for l in coerced[0].write_lease] == ["workspace/x.py"]

    def test_flag_off_tree_only_unit_hard_rejected(self, monkeypatch):
        # A tree-ONLY unit (no file lease) has no typed write to fall back to —
        # "coercing" it would still spawn a child for a flag-off shell payload.
        # Per spec §3.6/F27 ("hard-reject (or coerce typed-only)") it is HARD
        # REJECTED, NOT demoted to exploration.
        monkeypatch.delenv("ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED", raising=False)
        units = _build_work_units_from_requests([_tree_req("workspace")], "h16", 1)
        with pytest.raises(CoordinatorPathContractError, match="fail-closed"):
            _coerce_units_typed_only_if_flag_off(units)

    def test_flag_on_leaves_units_unchanged(self, monkeypatch):
        monkeypatch.setenv("ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED", "true")
        units = _build_work_units_from_requests([_tree_req("workspace")], "h16", 1)
        coerced = _coerce_units_typed_only_if_flag_off(units)
        assert coerced[0].shell_mode is True
        assert coerced[0].write_tree_lease[0].prefix == "workspace"


class TestDispatchNodeFlagOffFailClosed:
    """[S2 §3.6 F27 — wiring guard] The helper tests above prove the COERCION
    LOGIC. These prove the guard is actually WIRED INTO ``dispatch_node`` (both
    build branches), not merely defined: an implementer who builds the helpers
    but forgets to CALL them in ``dispatch_node`` would pass every helper test
    but fail here. We drive the real ``dispatch_node`` with the master flag OFF
    and a ``proposed_trees``-carrying request, then observe the units that
    ENTERED dispatch via the serialized manifest bytes (MIXED → coerced
    typed-only) and via the raised contract error (TREE-ONLY → hard-reject).
    Covers BOTH the first-time build branch (``:313`` → ``_first_time_dispatch``)
    and the rehydrate build branch (``:290`` → ``_rehydrate_dispatch``).
    """

    @pytest.mark.anyio
    async def test_first_time_dispatch_mixed_unit_coerced_typed_only(
        self, monkeypatch,
    ) -> None:
        import json as _json

        from app.domain.services.graphs import (
            parallel_execution_subgraph as _peg,
        )

        # Master flag OFF — the dispatch guard must coerce before enrichment.
        monkeypatch.delenv("ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED", raising=False)

        config = _base_config(peek_returns=None)
        config["configurable"]["session_service"].create_session_with_parent = (
            AsyncMock(side_effect=[_mk_session("c1")])
        )
        manifests: list[bytes] = []

        async def _capture_put(*, prefix, content, filename=None):
            if filename == "manifest.json":
                manifests.append(content)
            return "minio://manifest-ref"

        config["configurable"]["artifact_storage"].put_content_addressed_bytes = (
            AsyncMock(side_effect=_capture_put)
        )

        # MIXED unit: a real file lease + a tree lease / shell_mode. Flag OFF ⇒
        # the tree lease + shell_mode are stripped but the typed write survives.
        state = _base_state(work_unit_requests=[
            WorkUnitRequest(
                objective="mixed-write", phase="write",
                allowed_tools=["file_write"],
                proposed_paths=[ProposedPath(path="api/new.py", op="add")],
                proposed_trees=[
                    ProposedTree(prefix="workspace", ops=frozenset({"add"})),
                ],
            ),
        ])
        await _peg.dispatch_node(state, config)

        assert manifests, "no manifest.json uploaded"
        data = _json.loads(manifests[0])
        # The unit that ENTERED dispatch was coerced typed-only BEFORE serialize.
        assert data["shell_mode"] is False
        assert data["write_tree_lease"] == []
        # The typed write survives so the unit is still a valid write unit.
        assert data["write_lease"] == [
            {"path": "api/new.py", "op": "add",
             "base_digest": None, "seed_content_ref": None},
        ]

    @pytest.mark.anyio
    async def test_flag_off_overlapping_trees_disjoint_paths_coerced_not_rejected(
        self, monkeypatch,
    ) -> None:
        """[codex PR-3 R1 P1 — ordering] With the master flag OFF, two MIXED
        units whose typed file leases are DISJOINT but whose tree leases OVERLAP
        must be COERCED to typed-only (trees discarded) and dispatch — NOT
        rejected by the overlap guard. Coercion MUST run BEFORE overlap: under
        flag OFF the trees are stripped, so the (moot) tree overlap is irrelevant
        and only the disjoint typed leases remain. Overlap-first spuriously
        rejected this — an F27 dormancy violation that bites flag-OFF rollout
        once the planner emits trees.
        """
        import json as _json

        from app.domain.services.graphs import (
            parallel_execution_subgraph as _peg,
        )

        monkeypatch.delenv("ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED", raising=False)

        config = _base_config(peek_returns=None)  # default side_effect c1, c2
        manifests: list[bytes] = []

        async def _capture_put(*, prefix, content, filename=None):
            if filename == "manifest.json":
                manifests.append(content)
            return "minio://manifest-ref"

        config["configurable"]["artifact_storage"].put_content_addressed_bytes = (
            AsyncMock(side_effect=_capture_put)
        )

        # Two MIXED units: DISJOINT typed paths, OVERLAPPING tree prefixes.
        state = _base_state(work_unit_requests=[
            WorkUnitRequest(
                objective="mixed-a", phase="write", allowed_tools=["file_write"],
                proposed_paths=[ProposedPath(path="api/a.py", op="add")],
                proposed_trees=[
                    ProposedTree(prefix="workspace", ops=frozenset({"add"})),
                ],
            ),
            WorkUnitRequest(
                objective="mixed-b", phase="write", allowed_tools=["file_write"],
                proposed_paths=[ProposedPath(path="api/b.py", op="add")],
                proposed_trees=[
                    ProposedTree(prefix="workspace", ops=frozenset({"add"})),
                ],
            ),
        ])

        cmd = await _peg.dispatch_node(state, config)  # must NOT raise

        # Both children spawned — no spurious overlap rejection.
        assert config["configurable"][
            "session_service"
        ].create_session_with_parent.await_count == 2
        assert len(cmd.goto) == 2
        # Both units coerced typed-only (tree leases discarded under flag OFF).
        assert manifests, "no manifest.json uploaded"
        for raw in manifests:
            data = _json.loads(raw)
            assert data["shell_mode"] is False
            assert data["write_tree_lease"] == []

    @pytest.mark.anyio
    async def test_first_time_dispatch_tree_only_unit_hard_rejected(
        self, monkeypatch,
    ) -> None:
        from app.domain.models.path_validation import CoordinatorPathContractError
        from app.domain.services.graphs import (
            parallel_execution_subgraph as _peg,
        )

        monkeypatch.delenv("ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED", raising=False)

        config = _base_config(peek_returns=None)
        config["configurable"]["session_service"].create_session_with_parent = (
            AsyncMock(side_effect=[_mk_session("c1")])
        )

        # TREE-ONLY unit (no file lease) + flag OFF ⇒ dispatch_node must raise,
        # proving the guard is wired into the first-time build branch.
        state = _base_state(work_unit_requests=[
            WorkUnitRequest(
                objective="shell-only", phase="write",
                allowed_tools=["file_write"], proposed_paths=[],
                proposed_trees=[
                    ProposedTree(prefix="workspace", ops=frozenset({"add"})),
                ],
            ),
        ])
        with pytest.raises(CoordinatorPathContractError, match="fail-closed"):
            await _peg.dispatch_node(state, config)
        # No child was spawned (hard-reject happens before dispatch).
        config["configurable"][
            "session_service"
        ].create_session_with_parent.assert_not_called()

    @pytest.mark.anyio
    async def test_rehydrate_flag_off_overlapping_trees_disjoint_paths_coerced(
        self, monkeypatch,
    ) -> None:
        """[codex PR-3 R2 P2] Ordering regression LOCK on the REHYDRATE branch.
        The first-time branch is covered by
        ``test_flag_off_overlapping_trees_disjoint_paths_coerced_not_rejected``;
        the existing rehydrate test is tree-only-reject, which passes under
        EITHER order. Two MIXED units with disjoint typed paths + overlapping
        trees, flag OFF: coercion must strip the trees BEFORE overlap so the
        rehydrate dispatch proceeds (both pending children re-dispatched), NOT
        raise. Reverting the rehydrate branch to overlap-before-coerce makes
        this raise."""
        from app.domain.services.graphs import (
            parallel_execution_subgraph as _peg,
        )

        monkeypatch.delenv("ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED", raising=False)

        config = _base_config(peek_returns=2)
        wu_ids = _expected_wu_ids("step-abc", attempt_ix=2, count=2)
        existing = MagicMock()
        existing.child_session_ids = {wu_ids[0]: "c1", wu_ids[1]: "c2"}
        existing.pending = wu_ids
        existing.terminal = {}
        existing.already_applied = None
        config["configurable"]["rehydrate_service"].detect_existing_run = (
            AsyncMock(return_value=existing)
        )

        state = _base_state(work_unit_requests=[
            WorkUnitRequest(
                objective="mixed-a", phase="write", allowed_tools=["file_write"],
                proposed_paths=[ProposedPath(path="api/a.py", op="add")],
                proposed_trees=[
                    ProposedTree(prefix="workspace", ops=frozenset({"add"})),
                ],
            ),
            WorkUnitRequest(
                objective="mixed-b", phase="write", allowed_tools=["file_write"],
                proposed_paths=[ProposedPath(path="api/b.py", op="add")],
                proposed_trees=[
                    ProposedTree(prefix="workspace", ops=frozenset({"add"})),
                ],
            ),
        ])
        cmd = await _peg.dispatch_node(state, config)  # must NOT raise
        # Both pending children re-dispatched — no spurious overlap rejection.
        assert len(cmd.goto) == 2

    @pytest.mark.anyio
    async def test_rehydrate_dispatch_tree_only_unit_hard_rejected(
        self, monkeypatch,
    ) -> None:
        from app.domain.models.path_validation import CoordinatorPathContractError
        from app.domain.services.graphs import (
            parallel_execution_subgraph as _peg,
        )

        monkeypatch.delenv("ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED", raising=False)

        # Route the rehydrate build branch (:290): peek hits + detect returns an
        # existing run. The guard must run on the units built there too.
        config = _base_config(peek_returns=2)
        wu_ids = _expected_wu_ids("step-abc", attempt_ix=2, count=1)
        existing = MagicMock()
        existing.child_session_ids = {wu_ids[0]: "c1"}
        existing.pending = wu_ids
        existing.terminal = {}
        existing.already_applied = None
        config["configurable"]["rehydrate_service"].detect_existing_run = (
            AsyncMock(return_value=existing)
        )

        state = _base_state(work_unit_requests=[
            WorkUnitRequest(
                objective="shell-only", phase="write",
                allowed_tools=["file_write"], proposed_paths=[],
                proposed_trees=[
                    ProposedTree(prefix="workspace", ops=frozenset({"add"})),
                ],
            ),
        ])
        with pytest.raises(CoordinatorPathContractError, match="fail-closed"):
            await _peg.dispatch_node(state, config)
