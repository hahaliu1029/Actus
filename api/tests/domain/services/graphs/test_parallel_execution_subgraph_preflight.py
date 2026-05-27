"""[C2 PR-6 §14.3 #1] dispatch_node preflight cap + reducer quota-release tests.

Covers:
- ``_first_time_dispatch`` rejection paths for work_unit count cap,
  per-user concurrency cap, and descendants cap, including the
  concurrency-rollback-on-descendants-rejection ordering.
- Backward-compat: missing ``coordinator_limits`` / ``probe_quota`` /
  ``session_repository`` keys must NOT raise (caps silently skipped).
- ``reducer_node`` must release the user-concurrency slot acquired by
  dispatch on both normal return and reducer raise (try/finally).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.patch_reducer_service import (
    ReducerDiagnostics,
    ReducerOutput,
)
from app.domain.models.patch_apply_plan import GroupOutcome
from app.domain.models.work_unit import (
    PathLease,
    ProposedPath,
    WorkUnit,
    WorkUnitRequest,
)
from app.domain.services.coordinator_limits import CoordinatorLimits
from app.domain.services.graphs.parallel_execution_subgraph import (
    _first_time_dispatch,
    reducer_node,
)
from app.domain.services.subagent_limits import MAX_DESCENDANTS_PER_ROOT

pytestmark = pytest.mark.anyio


# ── shared fixtures ──────────────────────────────────────────────────────────


def _mk_session(sid: str) -> MagicMock:
    m = MagicMock()
    m.id = sid
    return m


def _mk_work_units(n: int) -> list[WorkUnit]:
    """Build N runtime WorkUnits with only ``add`` ops so the enrich-leases
    step is a no-op (no parent_sandbox digest / artifact upload required).

    Uses ``phase="write"`` because WorkUnit invariants require write_lease
    to be empty for ``phase="exploration"``. The dispatch_node enrich step
    treats ``op="add"`` as a no-op regardless of phase.
    """
    return [
        WorkUnit(
            work_unit_id=f"wu{i}",
            objective=f"objective {i}",
            phase="write",
            allowed_tools=["file_write"],
            write_lease=[PathLease(path=f"file{i}.py", op="add")],
            expected_result_schema=None,
        )
        for i in range(n)
    ]


def _base_state(work_units: list[WorkUnit]) -> dict:
    return {
        "coordinator_run_id": None,
        "step_id": "step-abc",
        "work_unit_requests": [
            WorkUnitRequest(
                objective=f"explore {i}", phase="write",
                allowed_tools=["file_write"],
                proposed_paths=[ProposedPath(path=f"file{i}.py", op="add")],
            )
            for i in range(len(work_units))
        ],
        "work_units": work_units,
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


def _full_dispatch_config(
    *,
    coordinator_limits: CoordinatorLimits | None = None,
    probe_quota: AsyncMock | MagicMock | None = None,
    session_repo: AsyncMock | MagicMock | None = None,
    n_children: int = 2,
) -> dict:
    """Build a config dict with every key ``_first_time_dispatch`` reads.

    Default ``coordinator_limits`` is ``None`` so backward-compat tests
    can opt out. Pass an instance to enable preflight checks.

    The descendants cap branch only fires when an explicit
    ``session_repository`` is passed via ``session_repo``; the default
    helper omits the key so legacy / partial-DI tests don't accidentally
    hit it.
    """
    session_service = AsyncMock()
    session_service.create_session_with_parent = AsyncMock(
        side_effect=[_mk_session(f"c{i}") for i in range(n_children)],
    )
    runner_starter = AsyncMock()
    runner_starter.start = AsyncMock()
    publisher = AsyncMock()
    publisher.publish = AsyncMock()
    artifact = AsyncMock()
    artifact.put_content_addressed_bytes = AsyncMock(return_value="minio://x")
    parent_sandbox = AsyncMock()
    parent_sandbox.compute_digest = AsyncMock(return_value="sha256_abc")
    parent_sandbox.read_file = AsyncMock(return_value=b"content")
    orchestrator = AsyncMock()
    orchestrator.run = AsyncMock()
    orchestrator_factory = MagicMock()
    orchestrator_factory.build = MagicMock(return_value=orchestrator)
    subscriber = AsyncMock()
    subscriber.subscribe = AsyncMock()

    cfg: dict = {
        "session_service": session_service,
        "child_runner_starter": runner_starter,
        "mailbox_publisher": publisher,
        "mailbox_subscriber": subscriber,
        "artifact_storage": artifact,
        "parent_sandbox": parent_sandbox,
        "orchestrator_factory": orchestrator_factory,
        "cancel_event": asyncio.Event(),
    }
    if coordinator_limits is not None:
        cfg["coordinator_limits"] = coordinator_limits
    if probe_quota is not None:
        cfg["probe_quota"] = probe_quota
    if session_repo is not None:
        cfg["session_repository"] = session_repo
    return {"configurable": cfg}


# ── preflight cap rejection paths ─────────────────────────────────────────────


class TestPreflightCaps:
    async def test_work_unit_count_exceeds_cap_raises(self) -> None:
        """work_units > max_work_units_per_run → ValueError.

        Side-effect assertion: rejection MUST happen before any session is
        created, any manifest uploaded, or any runner started.
        """
        limits = CoordinatorLimits(max_work_units_per_run=3)
        # probe_quota intentionally None: count check should fire first.
        config = _full_dispatch_config(coordinator_limits=limits, n_children=5)
        units = _mk_work_units(5)
        state = _base_state(units)

        with pytest.raises(ValueError) as ei:
            await _first_time_dispatch(state, config, "run1", units)

        assert "work_units=5" in str(ei.value)
        assert "cap 3" in str(ei.value)
        cfg = config["configurable"]
        cfg["session_service"].create_session_with_parent.assert_not_called()
        cfg["child_runner_starter"].start.assert_not_called()
        cfg["mailbox_publisher"].publish.assert_not_called()

    async def test_user_concurrency_acquire_failure_raises(self) -> None:
        """probe_quota.acquire_coordinator_concurrency=False → ValueError."""
        probe_quota = AsyncMock()
        probe_quota.acquire_coordinator_concurrency = AsyncMock(return_value=False)
        probe_quota.release_coordinator_quotas = AsyncMock()
        limits = CoordinatorLimits(
            max_work_units_per_run=5,
            max_concurrent_coordinator_runs_per_user=2,
        )
        config = _full_dispatch_config(
            coordinator_limits=limits, probe_quota=probe_quota,
        )
        units = _mk_work_units(2)
        state = _base_state(units)

        with pytest.raises(ValueError) as ei:
            await _first_time_dispatch(state, config, "run1", units)

        assert "concurrency cap" in str(ei.value)
        assert "u1" in str(ei.value)
        probe_quota.acquire_coordinator_concurrency.assert_awaited_once_with(
            user_id="u1", cap=2,
        )
        # Failed-acquire path MUST NOT call release (rollback) — that would
        # decrement someone else's slot.
        probe_quota.release_coordinator_quotas.assert_not_awaited()
        config["configurable"]["session_service"].create_session_with_parent.assert_not_called()

    async def test_descendants_would_exceed_cap_raises_and_rolls_back_concurrency(
        self,
    ) -> None:
        """count_descendants + requested > MAX_DESCENDANTS_PER_ROOT
        → release_coordinator_quotas called, then ValueError.

        Pins the ordering: the concurrency slot acquired in step 2 must be
        released BEFORE the ValueError propagates, so a rejected dispatch
        does not leak quota.
        """
        probe_quota = AsyncMock()
        probe_quota.acquire_coordinator_concurrency = AsyncMock(return_value=True)
        probe_quota.release_coordinator_quotas = AsyncMock()
        session_repo = AsyncMock()
        # existing=9, requested=2 → 11 > 10 cap
        session_repo.count_descendants = AsyncMock(return_value=9)

        limits = CoordinatorLimits(
            max_work_units_per_run=5,
            max_concurrent_coordinator_runs_per_user=2,
        )
        config = _full_dispatch_config(
            coordinator_limits=limits,
            probe_quota=probe_quota,
            session_repo=session_repo,
        )
        units = _mk_work_units(2)
        state = _base_state(units)

        with pytest.raises(ValueError) as ei:
            await _first_time_dispatch(state, config, "run1", units)

        assert "descendants cap" in str(ei.value)
        assert f"{MAX_DESCENDANTS_PER_ROOT}" in str(ei.value)
        assert "existing=9" in str(ei.value)
        assert "requested=2" in str(ei.value)

        # Concurrency was acquired then rolled back exactly once.
        probe_quota.acquire_coordinator_concurrency.assert_awaited_once()
        probe_quota.release_coordinator_quotas.assert_awaited_once_with(
            user_id="u1",
        )
        # No downstream side effects.
        config["configurable"]["session_service"].create_session_with_parent.assert_not_called()

    async def test_work_unit_count_exactly_at_cap_is_allowed(self) -> None:
        """[codex R2 P2-1 boundary] len(work_units) == max_work_units_per_run
        → NOT rejected (predicate is strict ``>``).

        Pins the cap predicate against off-by-one regressions: the
        intent is to reject *exceeding* the cap, so exactly-at-cap
        must pass cleanly through to downstream dispatch.
        """
        # Cap = 3, work_units = 3 → boundary case (must pass).
        limits = CoordinatorLimits(max_work_units_per_run=3)
        config = _full_dispatch_config(coordinator_limits=limits, n_children=3)
        units = _mk_work_units(3)
        state = _base_state(units)

        # Must NOT raise on the preflight count cap.
        await _first_time_dispatch(state, config, "run1", units)

        # Downstream proceeded (proves the count cap did not short-circuit).
        cfg = config["configurable"]
        assert (
            cfg["session_service"].create_session_with_parent.await_count == 3
        )

    async def test_descendants_exactly_at_cap_is_allowed(self) -> None:
        """[codex R2 P2-1 boundary] existing + requested == MAX_DESCENDANTS_PER_ROOT
        → NOT rejected (predicate is strict ``>``).

        Companion to ``test_descendants_would_exceed_cap_raises_and_rolls
        _back_concurrency``: exact-cap must pass so the concurrency slot
        is held (NOT rolled back) and downstream dispatch fires.
        """
        probe_quota = AsyncMock()
        probe_quota.acquire_coordinator_concurrency = AsyncMock(return_value=True)
        probe_quota.release_coordinator_quotas = AsyncMock()
        session_repo = AsyncMock()
        # existing + requested == MAX_DESCENDANTS_PER_ROOT → exact-cap.
        existing_count = MAX_DESCENDANTS_PER_ROOT - 2
        session_repo.count_descendants = AsyncMock(return_value=existing_count)

        limits = CoordinatorLimits(
            max_work_units_per_run=5,
            max_concurrent_coordinator_runs_per_user=2,
        )
        config = _full_dispatch_config(
            coordinator_limits=limits,
            probe_quota=probe_quota,
            session_repo=session_repo,
        )
        units = _mk_work_units(2)
        state = _base_state(units)

        # Boundary case: existing + 2 == cap → predicate ``>`` does NOT trip.
        await _first_time_dispatch(state, config, "run1", units)

        # Concurrency was acquired and (critically) NOT rolled back at
        # this layer. The reducer is responsible for the eventual
        # release on normal exit — dispatch only rolls back when it
        # explicitly rejects.
        probe_quota.acquire_coordinator_concurrency.assert_awaited_once()
        probe_quota.release_coordinator_quotas.assert_not_awaited()
        # Downstream proceeded — proves the descendants cap did not
        # short-circuit at boundary.
        cfg = config["configurable"]
        assert (
            cfg["session_service"].create_session_with_parent.await_count == 2
        )

    async def test_descendants_at_exact_cap_with_requested_zero_passes(
        self,
    ) -> None:
        """Boundary: existing == cap + requested == 0 → passes (no work to do)
        but the check is ``existing + requested > cap`` so equal is fine.

        We still need ``n_children=0`` for the dispatch to not crash on
        empty work_units, but since dispatch creates one session per work
        unit and we pass zero units, downstream is a no-op for sessions.
        """
        probe_quota = AsyncMock()
        probe_quota.acquire_coordinator_concurrency = AsyncMock(return_value=True)
        probe_quota.release_coordinator_quotas = AsyncMock()
        session_repo = AsyncMock()
        # existing=10, requested=0 → 10 == cap → passes.
        session_repo.count_descendants = AsyncMock(return_value=10)

        limits = CoordinatorLimits(max_work_units_per_run=5)
        config = _full_dispatch_config(
            coordinator_limits=limits,
            probe_quota=probe_quota,
            session_repo=session_repo,
            n_children=0,
        )
        units: list[WorkUnit] = []
        state = _base_state(units)

        # No raise — passes the cap check; downstream still runs (zero
        # children means zero side effects).
        await _first_time_dispatch(state, config, "run1", units)

        # Concurrency held (NOT rolled back) — reducer_node is responsible
        # for the release on the normal exit path.
        probe_quota.release_coordinator_quotas.assert_not_awaited()


# ── [codex R8 P2-1] dispatch-rollback waiter group destroy ────────────────────


class TestDispatchRollbackWaiterGroupCleanup:
    """[codex R8 P2-1] Pre-created waiter consumer groups MUST be destroyed
    when a later dispatch step fails so they don't leak in Redis.

    The pre-create loop at §7.5 calls ``subscriber.subscribe`` for every
    work unit BEFORE the subsequent steps (``runner_starter.start`` /
    ``publisher.publish`` / orchestrator launch). Earlier rounds added a
    try/except wrap that releases the concurrency slot on failure, but
    the pre-created waiter groups were not torn down → leaked groups
    accumulated in Redis until manual cleanup.
    """

    async def test_pre_created_waiter_groups_destroyed_on_dispatch_failure(
        self,
    ) -> None:
        """runner_starter.start raises → every pre-created waiter group
        is destroyed before the original exception propagates.

        Pins the cleanup contract: subscribe is called for each work
        unit during pre-create; when a later step fails, each previously
        subscribed (stream_key, consumer_group) pair MUST be passed to
        ``subscriber.destroy_group`` so Redis doesn't accumulate dead
        consumer groups.
        """
        probe_quota = AsyncMock()
        probe_quota.acquire_coordinator_concurrency = AsyncMock(return_value=True)
        probe_quota.release_coordinator_quotas = AsyncMock()
        limits = CoordinatorLimits(
            max_work_units_per_run=5,
            max_concurrent_coordinator_runs_per_user=2,
        )
        config = _full_dispatch_config(
            coordinator_limits=limits,
            probe_quota=probe_quota,
            n_children=2,
        )

        # Make the step AFTER the pre-create loop raise: runner_starter.start
        # is the first per-work-unit step after subscribe in §7.5.
        cfg = config["configurable"]
        cfg["child_runner_starter"].start = AsyncMock(
            side_effect=RuntimeError("runner start boom"),
        )

        # Track destroy_group calls.
        cfg["mailbox_subscriber"].destroy_group = AsyncMock()

        units = _mk_work_units(2)
        state = _base_state(units)

        with pytest.raises(RuntimeError, match="runner start boom"):
            await _first_time_dispatch(state, config, "run1", units)

        # Both pre-created waiter groups must have been destroyed.
        subscriber = cfg["mailbox_subscriber"]
        assert subscriber.subscribe.await_count == 2
        assert subscriber.destroy_group.await_count == 2

        # Verify each destroy call matches the corresponding subscribe.
        # child_session_ids are c0 and c1 (from _full_dispatch_config side
        # effect for n_children=2). root_session_id is "root1" per
        # _base_state.
        expected_calls = {
            ("actus:child:root1:mailbox", "coordinator:waiter:c0"),
            ("actus:child:root1:mailbox", "coordinator:waiter:c1"),
        }
        actual_calls = {
            (
                call.kwargs["stream_key"],
                call.kwargs["consumer_group"],
            )
            for call in subscriber.destroy_group.await_args_list
        }
        assert actual_calls == expected_calls

        # Concurrency slot was also released as part of rollback (Round 3
        # P1-3) — destroy_group cleanup is additive, not a replacement.
        probe_quota.release_coordinator_quotas.assert_awaited_once_with(
            user_id="u1",
        )

    async def test_destroy_group_failure_during_rollback_does_not_mask_original(
        self,
    ) -> None:
        """subscriber.destroy_group itself raises → the ORIGINAL dispatch
        exception still propagates (not the destroy_group exception).

        The destroy loop is best-effort by design: a Redis hiccup during
        rollback must not swallow / replace the real failure cause. We
        log + swallow inside the destroy loop so the trailing ``raise``
        in the except clause re-raises the dispatch exception.
        """
        probe_quota = AsyncMock()
        probe_quota.acquire_coordinator_concurrency = AsyncMock(return_value=True)
        probe_quota.release_coordinator_quotas = AsyncMock()
        limits = CoordinatorLimits(
            max_work_units_per_run=5,
            max_concurrent_coordinator_runs_per_user=2,
        )
        config = _full_dispatch_config(
            coordinator_limits=limits,
            probe_quota=probe_quota,
            n_children=2,
        )

        cfg = config["configurable"]
        cfg["child_runner_starter"].start = AsyncMock(
            side_effect=RuntimeError("runner start boom"),
        )
        # destroy_group itself fails — must NOT mask the original error.
        cfg["mailbox_subscriber"].destroy_group = AsyncMock(
            side_effect=RuntimeError("redis destroy boom"),
        )

        units = _mk_work_units(2)
        state = _base_state(units)

        # The original RuntimeError (runner start boom) MUST surface,
        # NOT the destroy_group exception.
        with pytest.raises(RuntimeError, match="runner start boom"):
            await _first_time_dispatch(state, config, "run1", units)

        # destroy_group was attempted for every pre-created group even
        # though the first attempt raised — the destroy loop catches per
        # iteration so a single bad group doesn't skip the rest.
        subscriber = cfg["mailbox_subscriber"]
        assert subscriber.destroy_group.await_count == 2


class TestBackwardCompat:
    async def test_no_coordinator_limits_in_cfg_skips_caps(self) -> None:
        """cfg without ``coordinator_limits`` → work_unit count cap skipped.

        With limits omitted, even an exceeded-cap-sized work_units list MUST
        proceed past the preflight block. Downstream side effects may still
        fire; we only assert that no preflight ValueError is raised.
        """
        config = _full_dispatch_config(coordinator_limits=None, n_children=10)
        units = _mk_work_units(10)
        state = _base_state(units)

        # Must not raise on the preflight cap; downstream invokes session
        # creation etc. Since side_effect is a list of length 10 and we
        # pass 10 units, this happens to flow cleanly through.
        await _first_time_dispatch(state, config, "run1", units)

        cfg = config["configurable"]
        # Downstream proceeded — proves the preflight didn't short-circuit.
        assert cfg["session_service"].create_session_with_parent.await_count == 10

    async def test_no_probe_quota_in_cfg_skips_concurrency_acquire(self) -> None:
        """cfg.get('probe_quota') is None → concurrency check skipped.

        ``coordinator_limits`` is still wired so the count cap fires if
        violated; concurrency check is silently bypassed.
        """
        limits = CoordinatorLimits(max_work_units_per_run=5)
        config = _full_dispatch_config(
            coordinator_limits=limits, probe_quota=None,
        )
        units = _mk_work_units(2)
        state = _base_state(units)

        # Must not raise; downstream proceeds.
        await _first_time_dispatch(state, config, "run1", units)
        assert (
            config["configurable"]["session_service"]
            .create_session_with_parent.await_count == 2
        )

    async def test_no_session_repository_in_cfg_skips_descendants_check(
        self,
    ) -> None:
        """cfg without ``session_repository`` → descendants check silently
        skipped.

        Only the explicit ``session_repository`` cfg key activates the
        descendants cap branch — there is no fallback to ``session_service``
        (production SessionService does not expose count_descendants
        publicly; a bare AsyncMock would otherwise spuriously fire the cap).
        """
        limits = CoordinatorLimits(max_work_units_per_run=5)
        probe_quota = AsyncMock()
        probe_quota.acquire_coordinator_concurrency = AsyncMock(return_value=True)
        probe_quota.release_coordinator_quotas = AsyncMock()
        # No session_repo passed.
        config = _full_dispatch_config(
            coordinator_limits=limits, probe_quota=probe_quota,
        )
        assert "session_repository" not in config["configurable"]

        units = _mk_work_units(2)
        state = _base_state(units)

        # Must not raise; downstream proceeds (descendants check skipped).
        await _first_time_dispatch(state, config, "run1", units)
        assert (
            config["configurable"]["session_service"]
            .create_session_with_parent.await_count == 2
        )
        # release_coordinator_quotas NOT called by dispatch — only by reducer
        # on the happy path. Dispatch only rolls back on descendants-cap
        # rejection.
        probe_quota.release_coordinator_quotas.assert_not_awaited()


# ── reducer quota-release semantics ──────────────────────────────────────────


def _wu(work_unit_id: str = "wu1") -> WorkUnit:
    return WorkUnit(
        work_unit_id=work_unit_id,
        objective="test",
        phase="write",
        allowed_tools=["file_write"],
        write_lease=[PathLease(path="x.py", op="add")],
        expected_result_schema=None,
    )


class TestReducerReleasesQuota:
    async def test_reducer_normal_return_releases_quota(self) -> None:
        """Happy path: reducer returns Command → probe_quota.release fires.

        [codex R2 P1-4] State carries ``quota_acquired=True`` (set by
        ``_first_time_dispatch`` on the first-time path) so the release
        gate fires. The rehydrate path test below pins the inverse.
        """
        reducer = AsyncMock()
        reducer.reduce = AsyncMock(return_value=ReducerOutput(
            apply_plan=None,
            group_outcome=GroupOutcome.SUCCESS,
            step_result_candidate="ok",
            diagnostics=ReducerDiagnostics(),
        ))
        probe_quota = AsyncMock()
        probe_quota.release_coordinator_quotas = AsyncMock()
        state = {
            "coordinator_run_id": "r1",
            "user_id": "u1",
            "work_units": [_wu("wu1")],
            "worker_results": [],
            "quota_acquired": True,
        }
        config = {"configurable": {
            "patch_reducer_service": reducer,
            "probe_quota": probe_quota,
        }}

        cmd = await reducer_node(state, config)

        # Normal completion still returns the Command.
        assert cmd.update["group_outcome"] == GroupOutcome.SUCCESS
        probe_quota.release_coordinator_quotas.assert_awaited_once_with(
            user_id="u1",
        )

    async def test_reducer_raise_still_releases_quota(self) -> None:
        """Exception path: reducer raises → release still fires, then exc
        propagates."""
        reducer = AsyncMock()
        reducer.reduce = AsyncMock(side_effect=RuntimeError("reduce boom"))
        probe_quota = AsyncMock()
        probe_quota.release_coordinator_quotas = AsyncMock()
        state = {
            "coordinator_run_id": "r1",
            "user_id": "u1",
            "work_units": [_wu("wu1")],
            "worker_results": [],
            "quota_acquired": True,
        }
        config = {"configurable": {
            "patch_reducer_service": reducer,
            "probe_quota": probe_quota,
        }}

        with pytest.raises(RuntimeError, match="reduce boom"):
            await reducer_node(state, config)

        # release ran even though reduce raised.
        probe_quota.release_coordinator_quotas.assert_awaited_once_with(
            user_id="u1",
        )

    async def test_reducer_release_failure_does_not_mask_normal_return(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """probe_quota.release_coordinator_quotas itself raising must NOT
        mask the reducer's Command return. Failure is logged."""
        import logging

        reducer = AsyncMock()
        reducer.reduce = AsyncMock(return_value=ReducerOutput(
            apply_plan=None,
            group_outcome=GroupOutcome.SUCCESS,
            step_result_candidate="ok",
            diagnostics=ReducerDiagnostics(),
        ))
        probe_quota = AsyncMock()
        probe_quota.release_coordinator_quotas = AsyncMock(
            side_effect=RuntimeError("redis down"),
        )
        state = {
            "coordinator_run_id": "r1",
            "user_id": "u1",
            "work_units": [_wu("wu1")],
            "worker_results": [],
            "quota_acquired": True,
        }
        config = {"configurable": {
            "patch_reducer_service": reducer,
            "probe_quota": probe_quota,
        }}
        caplog.set_level(
            logging.ERROR,
            logger="app.domain.services.graphs.parallel_execution_subgraph",
        )

        cmd = await reducer_node(state, config)

        # Reducer's Command still returned normally.
        assert cmd.update["group_outcome"] == GroupOutcome.SUCCESS
        # Failure logged at ERROR via logger.exception.
        assert any(
            "probe_quota release failed" in rec.message
            for rec in caplog.records
        ), f"expected probe_quota release fail log; got {[r.message for r in caplog.records]}"

    async def test_reducer_no_probe_quota_in_cfg_does_not_raise(self) -> None:
        """Backward compat: cfg without probe_quota → reducer flows normally
        without attempting release."""
        reducer = AsyncMock()
        reducer.reduce = AsyncMock(return_value=ReducerOutput(
            apply_plan=None,
            group_outcome=GroupOutcome.SUCCESS,
            step_result_candidate="ok",
            diagnostics=ReducerDiagnostics(),
        ))
        state = {
            "coordinator_run_id": "r1",
            "user_id": "u1",
            "work_units": [_wu("wu1")],
            "worker_results": [],
        }
        config = {"configurable": {"patch_reducer_service": reducer}}

        cmd = await reducer_node(state, config)

        assert cmd.update["group_outcome"] == GroupOutcome.SUCCESS

    async def test_reducer_no_user_id_in_state_skips_release(self) -> None:
        """If state lacks user_id, skip release (don't pass user_id=None
        to Redis)."""
        reducer = AsyncMock()
        reducer.reduce = AsyncMock(return_value=ReducerOutput(
            apply_plan=None,
            group_outcome=GroupOutcome.SUCCESS,
            step_result_candidate="ok",
            diagnostics=ReducerDiagnostics(),
        ))
        probe_quota = AsyncMock()
        probe_quota.release_coordinator_quotas = AsyncMock()
        state = {
            "coordinator_run_id": "r1",
            # no "user_id" key
            "work_units": [_wu("wu1")],
            "worker_results": [],
        }
        config = {"configurable": {
            "patch_reducer_service": reducer,
            "probe_quota": probe_quota,
        }}

        cmd = await reducer_node(state, config)
        assert cmd.update["group_outcome"] == GroupOutcome.SUCCESS
        probe_quota.release_coordinator_quotas.assert_not_awaited()

    async def test_reducer_missing_coordinator_run_id_still_releases(
        self,
    ) -> None:
        """The early-return on missing coordinator_run_id must still release
        the concurrency slot (otherwise a topology-bug session would leak
        quota)."""
        reducer = AsyncMock()
        probe_quota = AsyncMock()
        probe_quota.release_coordinator_quotas = AsyncMock()
        state = {
            "coordinator_run_id": None,
            "user_id": "u1",
            "work_units": [_wu("wu1")],
            "worker_results": [],
            "quota_acquired": True,
        }
        config = {"configurable": {
            "patch_reducer_service": reducer,
            "probe_quota": probe_quota,
        }}

        cmd = await reducer_node(state, config)
        assert "missing coordinator_run_id" in cmd.update["step_result_candidate"]
        reducer.reduce.assert_not_called()
        # try/finally still runs after the early return.
        probe_quota.release_coordinator_quotas.assert_awaited_once_with(
            user_id="u1",
        )

    async def test_reducer_rehydrate_path_does_not_release_quota(
        self,
    ) -> None:
        """[codex R2 P1-4 invariant] Rehydrate dispatch does NOT acquire
        the concurrency slot — the slot is owned by the prior dispatch
        incarnation that crashed. Therefore the reducer MUST NOT release
        on the rehydrate path, otherwise a downstream DECR would push
        the counter below zero or steal another incarnation's slot.

        Simulated by ``quota_acquired=False`` (or missing) in state.
        """
        reducer = AsyncMock()
        reducer.reduce = AsyncMock(return_value=ReducerOutput(
            apply_plan=None,
            group_outcome=GroupOutcome.SUCCESS,
            step_result_candidate="ok",
            diagnostics=ReducerDiagnostics(),
        ))
        probe_quota = AsyncMock()
        probe_quota.release_coordinator_quotas = AsyncMock()
        # Rehydrate path: ``_rehydrate_dispatch`` did NOT write
        # quota_acquired into Command.update — TypedDict total=False
        # leaves it absent, defaulting to False at .get(..., False).
        state = {
            "coordinator_run_id": "r1",
            "user_id": "u1",
            "work_units": [_wu("wu1")],
            "worker_results": [],
            # NB: no "quota_acquired" key — simulates rehydrate path
        }
        config = {"configurable": {
            "patch_reducer_service": reducer,
            "probe_quota": probe_quota,
        }}

        cmd = await reducer_node(state, config)

        # Reducer still ran + returned its Command.
        assert cmd.update["group_outcome"] == GroupOutcome.SUCCESS
        # Release MUST NOT have been called — quota_acquired was False.
        probe_quota.release_coordinator_quotas.assert_not_awaited()

    async def test_reducer_explicit_quota_acquired_false_skips_release(
        self,
    ) -> None:
        """[codex R2 P1-4 invariant] Explicit ``quota_acquired=False`` in
        state (e.g., dispatch-body rollback path that bailed before
        Command(update=) fired) must also skip release. Pins the
        boolean flag semantics independent of dict-key absence."""
        reducer = AsyncMock()
        reducer.reduce = AsyncMock(return_value=ReducerOutput(
            apply_plan=None,
            group_outcome=GroupOutcome.SUCCESS,
            step_result_candidate="ok",
            diagnostics=ReducerDiagnostics(),
        ))
        probe_quota = AsyncMock()
        probe_quota.release_coordinator_quotas = AsyncMock()
        state = {
            "coordinator_run_id": "r1",
            "user_id": "u1",
            "work_units": [_wu("wu1")],
            "worker_results": [],
            "quota_acquired": False,
        }
        config = {"configurable": {
            "patch_reducer_service": reducer,
            "probe_quota": probe_quota,
        }}

        cmd = await reducer_node(state, config)

        assert cmd.update["group_outcome"] == GroupOutcome.SUCCESS
        probe_quota.release_coordinator_quotas.assert_not_awaited()
