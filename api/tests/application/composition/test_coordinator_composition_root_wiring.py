"""PR-9b-A INV-A1 + INV-A2 — composition-root populates the four deferred
SupervisorContext slots when ACTUS_C2_COORDINATOR_ENABLED is irrelevant
(wiring is always-live; flag gates ENTRY, not WIRING).

Task A4 specifically locks the threading of the two new
``build_supervisor_registry`` kwargs through the ``_factory(root_session_id)``
closure into ``SupervisorContext(...)``:

* ``coordinator_envelope_store`` (PR-7 §12.4) — terminal envelope persistence
* ``cost_rollup_service`` (PR-6 §14.4) — parent-cost rollup PROLOGUE

Both fields existed on ``SupervisorContext`` as ``Optional[...] = None``
before this PR; A4 makes them actually populated rather than dangling at
``None`` in production. Spec §704 + plan INV-A1/A2: composition-root MUST
populate these slots regardless of the C2 entry flag — wiring is always-live;
the runtime feature flag only gates ENTRY (whether coordinator behaviour
fires), not WIRING (whether the dependencies are threaded through).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.application.services.coordinator_terminal_transition import (
    CoordinatorTerminalCommand,
    ExpectedCoordinatorLineage,
)
from app.domain.models.session import Session, SessionStatus
from app.interfaces import service_dependencies
from app.interfaces.service_dependencies import build_supervisor_registry


def _make_registry():
    """Construct a registry with every required kwarg as a MagicMock.

    Patches ``get_postgres`` so the inner ``DbMailboxEnvelopeAuditRepository``
    + session-repo adapter construction doesn't try to touch the real DB
    at import-time (mirrors the PR-4.5 gate test's pattern).
    """
    redis_client = MagicMock()
    redis_client.client = MagicMock()

    fake_postgres = MagicMock()
    fake_postgres.session_factory = MagicMock()

    with patch(
        "app.interfaces.service_dependencies.get_postgres",
        return_value=fake_postgres,
    ):
        return build_supervisor_registry(
            redis_client=redis_client,
            publisher=MagicMock(),
            sandbox_lifecycle_service=MagicMock(),
            coordinator_envelope_store=MagicMock(),
            cost_rollup_service=MagicMock(),
        )


def test_registry_factory_fills_coordinator_envelope_store():
    """INV-A1 — composition-root populates ``ctx.coordinator_envelope_store``
    in the ``_factory(root_session_id)`` closure regardless of feature flag.
    """
    sentinel = MagicMock(name="coordinator_envelope_store")
    redis_client = MagicMock()
    redis_client.client = MagicMock()

    fake_postgres = MagicMock()
    fake_postgres.session_factory = MagicMock()

    captured: dict[str, object] = {}

    def _capture_context(ctx, **kwargs):  # noqa: ANN001
        captured["coordinator_envelope_store"] = ctx.coordinator_envelope_store
        return MagicMock()

    with patch(
        "app.application.services.mailbox_supervisor.MailboxSupervisor",
        side_effect=_capture_context,
    ), patch(
        "app.interfaces.service_dependencies.get_postgres",
        return_value=fake_postgres,
    ):
        registry = build_supervisor_registry(
            redis_client=redis_client,
            publisher=MagicMock(),
            sandbox_lifecycle_service=MagicMock(),
            coordinator_envelope_store=sentinel,
            cost_rollup_service=MagicMock(),
        )
        registry._factory("root-1")

    assert captured["coordinator_envelope_store"] is sentinel, (
        "build_supervisor_registry must thread coordinator_envelope_store "
        "into SupervisorContext(...) inside the _factory closure; "
        f"got {captured.get('coordinator_envelope_store')!r}"
    )


def test_registry_factory_fills_cost_rollup_service():
    """INV-A2 — composition-root populates ``ctx.cost_rollup_service`` in the
    ``_factory(root_session_id)`` closure regardless of feature flag.
    """
    sentinel = MagicMock(name="cost_rollup_service")
    redis_client = MagicMock()
    redis_client.client = MagicMock()

    fake_postgres = MagicMock()
    fake_postgres.session_factory = MagicMock()

    captured: dict[str, object] = {}

    def _capture_context(ctx, **kwargs):  # noqa: ANN001
        captured["cost_rollup_service"] = ctx.cost_rollup_service
        return MagicMock()

    with patch(
        "app.application.services.mailbox_supervisor.MailboxSupervisor",
        side_effect=_capture_context,
    ), patch(
        "app.interfaces.service_dependencies.get_postgres",
        return_value=fake_postgres,
    ):
        registry = build_supervisor_registry(
            redis_client=redis_client,
            publisher=MagicMock(),
            sandbox_lifecycle_service=MagicMock(),
            coordinator_envelope_store=MagicMock(),
            cost_rollup_service=sentinel,
        )
        registry._factory("root-2")

    assert captured["cost_rollup_service"] is sentinel, (
        "build_supervisor_registry must thread cost_rollup_service into "
        "SupervisorContext(...) inside the _factory closure; "
        f"got {captured.get('cost_rollup_service')!r}"
    )


def test_registry_construction_succeeds_with_new_kwargs():
    """Smoke — calling ``build_supervisor_registry`` with the two new kwargs
    must not raise ``TypeError`` (signature accepts them) and must return
    a callable ``supervisor_factory`` closure that constructs a supervisor
    instance with the populated ctx.
    """
    registry = _make_registry()
    assert callable(registry._factory)
    # Sanity: closure runs end-to-end with the patched MailboxSupervisor.
    with patch(
        "app.application.services.mailbox_supervisor.MailboxSupervisor",
    ) as mock_supervisor_cls:
        mock_supervisor_cls.return_value = MagicMock()
        result = registry._factory("root-smoke")
    assert result is not None
    # Confirm the module-level callback symbol survives — backwards-compat
    # with the PR-4.5 gate test.
    assert service_dependencies._pr4_5_agent_service_callback is not None


def test_registry_factory_reuses_exact_shared_liveness_service():
    sentinel = MagicMock(name="shared_liveness_service")
    redis_client = MagicMock()
    redis_client.client = MagicMock()
    fake_postgres = MagicMock()
    fake_postgres.session_factory = MagicMock()
    captured: dict[str, object] = {}

    def _capture_context(ctx, **_kwargs):  # noqa: ANN001
        captured["liveness_service"] = ctx.liveness_service
        return MagicMock()

    with patch(
        "app.application.services.mailbox_supervisor.MailboxSupervisor",
        side_effect=_capture_context,
    ), patch(
        "app.interfaces.service_dependencies.get_postgres",
        return_value=fake_postgres,
    ):
        registry = build_supervisor_registry(
            redis_client=redis_client,
            publisher=MagicMock(),
            sandbox_lifecycle_service=MagicMock(),
            coordinator_envelope_store=MagicMock(),
            cost_rollup_service=MagicMock(),
            coordinator_liveness_service=sentinel,
        )
        registry._factory("root-liveness")

    assert captured["liveness_service"] is sentinel


@pytest.mark.anyio
async def test_registry_factory_wires_real_ssm_terminalizer_port():
    """Production locks, validates, writes and commits through one UoW."""
    redis_client = MagicMock()
    redis_client.client = MagicMock()
    fake_postgres = MagicMock()
    reader_db_session = MagicMock(name="reader_db_session")

    class _ReaderSessionContext:
        async def __aenter__(self):
            return reader_db_session

        async def __aexit__(self, *_args):
            return None

    fake_postgres.session_factory = lambda: _ReaderSessionContext()

    session_repo = MagicMock(name="session_repo")

    class _UoW:
        def __init__(self) -> None:
            self.session = session_repo
            self.commit = AsyncMock()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    ssm = MagicMock()
    ssm.terminate = AsyncMock(return_value=True)
    authoritative_row = Session(
        id="child-1",
        parent_session_id="root-terminal",
        root_session_id="root-terminal",
        worker_type="subagent",
        subagent_control_plane="mailbox",
        tool_filter_preset="coordinator_step",
        coordinator_run_id="run-1",
        work_unit_id="wu-1",
    )
    session_repo.get_by_id_for_update = AsyncMock(return_value=authoritative_row)
    uow = _UoW()
    captured: dict[str, object] = {}

    def _capture_context(ctx, **_kwargs):  # noqa: ANN001
        captured["terminalizer"] = getattr(ctx, "terminalize_child", None)
        captured["session_reader"] = getattr(ctx, "session_repo", None)
        return MagicMock()

    with patch(
        "app.application.services.mailbox_supervisor.MailboxSupervisor",
        side_effect=_capture_context,
    ), patch(
        "app.interfaces.service_dependencies.get_postgres",
        return_value=fake_postgres,
    ), patch(
        "app.interfaces.service_dependencies.get_uow",
        side_effect=lambda: uow,
    ), patch(
        "app.application.composition.graph_assembly.build_session_state_machine",
        return_value=ssm,
    ):
        registry = build_supervisor_registry(
            redis_client=redis_client,
            publisher=MagicMock(),
            sandbox_lifecycle_service=MagicMock(),
            coordinator_envelope_store=MagicMock(),
            cost_rollup_service=MagicMock(),
        )
        registry._factory("root-terminal")
        terminalizer = captured["terminalizer"]
        assert callable(terminalizer)
        transitioned = await terminalizer(
            CoordinatorTerminalCommand(
                lineage=ExpectedCoordinatorLineage(
                    child_session_id="child-1",
                    parent_session_id="root-terminal",
                    root_session_id="root-terminal",
                    coordinator_run_id="run-1",
                ),
                status=SessionStatus.COMPLETED,
                reason="natural",
            )
        )

    session_repo.get_by_id_for_update.assert_awaited_once_with("child-1")
    assert transitioned is True
    ssm.terminate.assert_awaited_once_with(
        "child-1",
        SessionStatus.COMPLETED,
        "natural",
        session_repo=session_repo,
    )
    uow.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# Task A8 — INV-A1: lifespan composition smoke (no real I/O / no real lifespan)
# ---------------------------------------------------------------------------
#
# The full FastAPI lifespan does real DB / Redis / Docker work that we don't
# want to bring into a unit test. We exercise the composition-root *helper*
# directly instead, calling the new ``build_coordinator_runtime_deps`` factory
# that ``main.py`` lifespan delegates to. The two assertions still match the
# plan literal contract:
#
# 1. After the helper runs, the four key app.state slots are populated.
# 2. The ``_CoordinatorRuntimeDeps`` aggregator is the value threaded through
#    to the live ``AgentTaskRunner`` construction path.


class _OrphanClaimRedis:
    def __init__(
        self,
        *,
        publish_error: BaseException | None = None,
        publish_errors: list[BaseException | None] | None = None,
        cleanup_error: BaseException | None = None,
        replace_owner_on_publish_error: bool = True,
        quota_renew_result: int = 1,
    ) -> None:
        self.values: dict[str, str] = {}
        self.orphan_keys: list[str] = []
        self.deleted: list[str] = []
        self.eval_calls: list[tuple[object, ...]] = []
        self.publish_error = publish_error
        self.publish_errors = list(publish_errors or [])
        self.cleanup_error = cleanup_error
        self.replace_owner_on_publish_error = replace_owner_on_publish_error
        self.quota_renew_result = quota_renew_result
        self.xadd_calls = 0

    async def set(
        self,
        key: str,
        value: str,
        *,
        nx: bool,
        ex: int,
    ) -> bool:
        assert nx is True
        assert ex > 0
        if key in self.values:
            return False
        self.values[key] = str(value)
        if key.startswith("coordinator:orphan-reconcile:"):
            self.orphan_keys.append(key)
        return True

    async def xadd(self, *_args, **_kwargs) -> str:
        self.xadd_calls += 1
        error = (
            self.publish_errors.pop(0)
            if self.publish_errors
            else self.publish_error
        )
        if error is not None:
            if self.replace_owner_on_publish_error:
                self.values[self.orphan_keys[-1]] = "replacement-owner"
            raise error
        return "1-0"

    async def delete(self, key: str) -> int:
        self.deleted.append(key)
        return int(self.values.pop(key, None) is not None)

    async def eval(self, script: str, numkeys: int, *args: object) -> int:
        self.eval_calls.append((script, numkeys, *args))
        if "ZSCORE" in script and "ZADD" in script:
            assert numkeys == 1
            return self.quota_renew_result
        assert "coordinator-orphan-owner-compare-delete-v1" in script
        assert numkeys == 1
        if self.cleanup_error is not None:
            raise self.cleanup_error
        key = str(args[0])
        owner_token = str(args[1])
        if self.values.get(key) != owner_token:
            return 0
        self.values.pop(key, None)
        return 1


def _build_runtime_deps_with_raw_redis(
    raw_redis: object,
    *,
    app_state: SimpleNamespace | None = None,
):
    from app.interfaces.service_dependencies import (
        build_coordinator_runtime_deps,
    )

    fake_redis = MagicMock()
    fake_redis.client = raw_redis
    fake_postgres = MagicMock()
    fake_postgres.session_factory = MagicMock()

    with patch(
        "app.interfaces.service_dependencies.get_postgres",
        return_value=fake_postgres,
    ), patch(
        "app.infrastructure.storage.postgres.get_postgres",
        return_value=fake_postgres,
    ), patch(
        "app.interfaces.service_dependencies.get_minio",
        return_value=MagicMock(),
    ):
        return build_coordinator_runtime_deps(
            app_state=app_state or SimpleNamespace(),
            redis_client=fake_redis,
        )


@pytest.mark.anyio
async def test_parent_lease_renews_auto_degrade_through_shared_supervisor() -> None:
    shared_supervisor = MagicMock()
    shared_supervisor.renew_auto_degrade_expiry_if_running = AsyncMock(
        return_value=True
    )
    app_state = SimpleNamespace(supervisor=shared_supervisor)
    coord_deps = _build_runtime_deps_with_raw_redis(
        _OrphanClaimRedis(),
        app_state=app_state,
    )
    guard = coord_deps.coordinator_wait_guard_factory(watchdog=MagicMock())
    handle = guard._parent_lease_factory(
        root_session_id="root",
        parent_session_id="parent",
        coordinator_run_id="run",
        step_id="step",
        child_session_ids=("child-1",),
        owner_alive=lambda: True,
    )

    await handle._renew_once()

    shared_supervisor.renew_auto_degrade_expiry_if_running.assert_awaited_once_with(
        session_id="parent",
    )


@pytest.mark.anyio
async def test_parent_lease_renews_existing_quota_member_without_reacquire() -> None:
    raw_redis = _OrphanClaimRedis()
    coord_deps = _build_runtime_deps_with_raw_redis(raw_redis)
    guard = coord_deps.coordinator_wait_guard_factory(watchdog=MagicMock())
    handle = guard._parent_lease_factory(
        root_session_id="root",
        parent_session_id="parent",
        user_id="user",
        coordinator_run_id="run",
        step_id="step",
        child_session_ids=("child-1",),
        owner_alive=lambda: True,
    )

    await handle._renew_once()

    quota_calls = [
        call for call in raw_redis.eval_calls
        if "ZSCORE" in str(call[0]) and "ZADD" in str(call[0])
    ]
    assert len(quota_calls) == 1
    assert quota_calls[0][2:4] == (
        "actus:coord:concurrent:user", "run",
    )
    assert "ZREMRANGEBYSCORE" not in str(quota_calls[0][0])


@pytest.mark.anyio
@pytest.mark.parametrize("renew_result", [1, 0])
async def test_authorized_child_heartbeat_renews_same_run_without_reacquire(
    renew_result: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from app.application.services.coordinator_liveness_lease_service import (
        CoordinatorChildLease,
    )

    raw_redis = _OrphanClaimRedis(quota_renew_result=renew_result)
    coord_deps = _build_runtime_deps_with_raw_redis(raw_redis)
    liveness = coord_deps.coordinator_liveness_service
    lease = CoordinatorChildLease(
        root_session_id="root",
        parent_session_id="parent",
        child_session_id="child-1",
        coordinator_run_id="run",
        work_unit_id="wu-1",
        last_seen_epoch=1.0,
        phase="in_tool",
    )
    authoritative_row = SimpleNamespace(
        id="child-1",
        user_id="user",
        coordinator_run_id="run",
        status=SessionStatus.RUNNING,
    )

    with patch.object(
        type(liveness._sessions),
        "get_by_id",
        new=AsyncMock(return_value=authoritative_row),
    ):
        await liveness._callbacks[-1](lease)

    quota_calls = [
        call for call in raw_redis.eval_calls
        if "ZSCORE" in str(call[0]) and "ZADD" in str(call[0])
    ]
    assert len(quota_calls) == 1
    assert quota_calls[0][2:4] == (
        "actus:coord:concurrent:user", "run",
    )
    assert "ZREMRANGEBYSCORE" not in str(quota_calls[0][0])
    if renew_result == 0:
        assert "heartbeat quota lease lost" in caplog.text


@pytest.mark.anyio
async def test_child_heartbeat_does_not_renew_after_row_turns_terminal() -> None:
    from app.application.services.coordinator_liveness_lease_service import (
        CoordinatorChildLease,
    )

    raw_redis = _OrphanClaimRedis()
    coord_deps = _build_runtime_deps_with_raw_redis(raw_redis)
    liveness = coord_deps.coordinator_liveness_service
    lease = CoordinatorChildLease(
        root_session_id="root",
        parent_session_id="parent",
        child_session_id="child-1",
        coordinator_run_id="run",
        work_unit_id="wu-1",
        last_seen_epoch=1.0,
        phase="in_tool",
    )
    terminal_row = SimpleNamespace(
        id="child-1",
        user_id="user",
        coordinator_run_id="run",
        status=SessionStatus.COMPLETED,
    )

    with patch.object(
        type(liveness._sessions),
        "get_by_id",
        new=AsyncMock(return_value=terminal_row),
    ):
        await liveness._callbacks[-1](lease)

    assert not [
        call for call in raw_redis.eval_calls
        if "ZSCORE" in str(call[0]) and "ZADD" in str(call[0])
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "first_error",
    [RuntimeError("first child failed"), asyncio.CancelledError()],
)
async def test_parent_lease_orphan_adapter_retries_failed_child_and_keeps_siblings(
    first_error: BaseException,
) -> None:
    from app.application.services.coordinator_parent_execution_lease import (
        CoordinatorParentExecutionLease,
    )

    raw_redis = _OrphanClaimRedis(
        publish_errors=[first_error, None, None],
        replace_owner_on_publish_error=False,
    )
    coord_deps = _build_runtime_deps_with_raw_redis(raw_redis)
    watchdog = MagicMock()
    guard = coord_deps.coordinator_wait_guard_factory(watchdog=watchdog)
    handle = guard._parent_lease_factory(
        root_session_id="root",
        parent_session_id="parent",
        coordinator_run_id="run",
        step_id="step",
        child_session_ids=("child-1", "child-2"),
        owner_alive=lambda: True,
    )

    assert isinstance(handle, CoordinatorParentExecutionLease)
    assert handle._liveness is coord_deps.coordinator_liveness_service
    assert guard.watchdog is watchdog
    assert handle._on_all_children_stale is not None

    await handle._notify_all_children_stale_once()

    # First child failed and released its owner claim; the second sibling was
    # still reconciled successfully. A failed aggregate must remain retryable.
    assert handle._all_stale_notified is False
    assert raw_redis.xadd_calls == 2
    assert len([
        key for key in raw_redis.values
        if key.startswith("coordinator:orphan-reconcile:")
    ]) == 1

    await handle._notify_all_children_stale_once()

    # Retry publishes only the previously failed child. The successful child
    # is suppressed by the real NX claim path, then the run becomes notified.
    assert handle._all_stale_notified is True
    assert raw_redis.xadd_calls == 3
    assert len([
        key for key in raw_redis.values
        if key.startswith("coordinator:orphan-reconcile:")
    ]) == 2


@pytest.mark.anyio
async def test_orphan_reconcile_uses_unique_opaque_owner_tokens() -> None:
    raw_redis = _OrphanClaimRedis()
    coord_deps = _build_runtime_deps_with_raw_redis(raw_redis)
    reconciler = coord_deps.terminal_waiter._orphan_reconciler

    await reconciler(
        child_session_id="child-1",
        root_session_id="root-1",
        coordinator_run_id="run-1",
    )
    await reconciler(
        child_session_id="child-2",
        root_session_id="root-1",
        coordinator_run_id="run-1",
    )

    tokens = [raw_redis.values[key] for key in raw_redis.orphan_keys]
    assert len(tokens) == 2
    assert len(set(tokens)) == 2
    assert all(token != "1" and len(token) >= 16 for token in tokens)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "publish_error",
    [RuntimeError("publish failed"), asyncio.CancelledError()],
)
async def test_orphan_reconcile_old_owner_cannot_delete_replacement_claim(
    publish_error: BaseException,
) -> None:
    raw_redis = _OrphanClaimRedis(publish_error=publish_error)
    coord_deps = _build_runtime_deps_with_raw_redis(raw_redis)
    reconciler = coord_deps.terminal_waiter._orphan_reconciler

    with pytest.raises(type(publish_error)) as caught:
        await reconciler(
            child_session_id="child-1",
            root_session_id="root-1",
            coordinator_run_id="run-1",
        )

    assert caught.value is publish_error
    orphan_key = raw_redis.orphan_keys[0]
    assert raw_redis.values[orphan_key] == "replacement-owner"
    assert orphan_key not in raw_redis.deleted
    assert len(raw_redis.eval_calls) == 1


@pytest.mark.anyio
async def test_orphan_cleanup_error_does_not_mask_publish_error() -> None:
    publish_error = RuntimeError("publish failed")
    raw_redis = _OrphanClaimRedis(
        publish_error=publish_error,
        cleanup_error=RuntimeError("redis cleanup failed"),
    )
    coord_deps = _build_runtime_deps_with_raw_redis(raw_redis)

    with pytest.raises(RuntimeError) as caught:
        await coord_deps.terminal_waiter._orphan_reconciler(
            child_session_id="child-1",
            root_session_id="root-1",
            coordinator_run_id="run-1",
        )

    assert caught.value is publish_error
    assert len(raw_redis.eval_calls) == 1


def test_lifespan_constructs_coordinator_singletons_exactly_once():
    """INV-A1 — composition-root populates four lifespan-scoped singletons.

    Calls ``build_coordinator_runtime_deps`` (the helper main.py lifespan
    invokes) and asserts each slot is non-None on the supplied ``app.state``
    namespace. We use a ``SimpleNamespace`` here as a stand-in for
    ``Starlette.app.state`` — both expose attribute-style access so the
    helper code path is identical.
    """
    from types import SimpleNamespace

    from app.application.services.coordinator_runtime_deps import (
        _CoordinatorRuntimeDeps,
    )
    from app.interfaces.service_dependencies import (
        build_coordinator_runtime_deps,
    )

    fake_state = SimpleNamespace()
    fake_redis = MagicMock()
    fake_redis.client = MagicMock()
    fake_session_factory = MagicMock()

    fake_postgres = MagicMock()
    fake_postgres.session_factory = fake_session_factory

    fake_minio = MagicMock()

    with patch(
        "app.interfaces.service_dependencies.get_postgres",
        return_value=fake_postgres,
    ), patch(
        "app.infrastructure.storage.postgres.get_postgres",
        return_value=fake_postgres,
    ), patch(
        "app.interfaces.service_dependencies.get_minio",
        return_value=fake_minio,
    ):
        coord_deps = build_coordinator_runtime_deps(
            app_state=fake_state,
            redis_client=fake_redis,
        )

    # INV-A1 — four lifespan-scoped singletons must be populated on app.state.
    assert getattr(fake_state, "cost_rollup_service", None) is not None, (
        "app.state.cost_rollup_service not populated by composition root"
    )
    assert getattr(fake_state, "coordinator_envelope_store", None) is not None, (
        "app.state.coordinator_envelope_store not populated by composition root"
    )
    assert getattr(fake_state, "patch_applier_deps", None) is not None, (
        "app.state.patch_applier_deps not populated by composition root"
    )
    assert getattr(fake_state, "coord_deps", None) is not None, (
        "app.state.coord_deps not populated by composition root"
    )
    assert getattr(fake_state, "coordinator_liveness_service", None) is not None
    assert (
        coord_deps.coordinator_liveness_service
        is fake_state.coordinator_liveness_service
    )
    assert (
        coord_deps.terminal_waiter._liveness
        is coord_deps.coordinator_liveness_service
    )
    # The returned coord_deps must be the value object aggregator.
    assert isinstance(coord_deps, _CoordinatorRuntimeDeps)
    # And the same instance must be stored on app.state so downstream wiring
    # (AgentService / AgentTaskRunner) sees a single source of truth.
    assert fake_state.coord_deps is coord_deps

    # [C2b budget D10] CoordinatorMetrics constructed at the composition root
    # (OtelMeter() is no-op-safe pre-setup_observability) and threaded BOTH
    # into coord_deps AND into the starter ctor.
    from app.infrastructure.observability.coordinator_telemetry import (
        CoordinatorMetrics,
    )

    assert isinstance(coord_deps.coordinator_metrics, CoordinatorMetrics)
    starter = fake_state.coordinator_child_runner_starter
    assert starter._coordinator_metrics is coord_deps.coordinator_metrics

    # [C2b rollout WS1b] CoordinatorMetricsRecorder built at the composition
    # root (wrapping the metrics bundle + the user_id_hash salt) and threaded
    # BOTH into coord_deps (→ _build_config cfg → reducer run-level metrics)
    # AND into the starter ctor (→ adapter tool_calls). Single instance.
    from app.application.services.coordinator_metrics_recorder import (
        CoordinatorMetricsRecorder,
    )

    assert isinstance(coord_deps.coordinator_metrics_recorder, CoordinatorMetricsRecorder)
    assert starter._coordinator_metrics_recorder is coord_deps.coordinator_metrics_recorder


def test_lifespan_threads_coord_deps_through_agent_service():
    """INV-A1 — the helper returns a real ``_CoordinatorRuntimeDeps``
    aggregator whose 17 fields are all populated (not the legacy null deps).

    Together with the wiring in ``_build_agent_service`` (which forwards
    ``app.state.coord_deps`` into ``AgentService`` and from there into
    ``AgentTaskRunner.__init__``), this is sufficient to lock in the
    end-to-end threading contract without spinning up a full TestClient
    lifespan against real Postgres / Redis.
    """
    from types import SimpleNamespace

    from app.application.services.coordinator_runtime_deps import (
        _CoordinatorRuntimeDeps,
        _NullCoordinatorRuntimeDeps,
    )
    from app.interfaces.service_dependencies import (
        build_coordinator_runtime_deps,
    )

    fake_state = SimpleNamespace()
    fake_redis = MagicMock()
    fake_redis.client = MagicMock()

    fake_postgres = MagicMock()
    fake_postgres.session_factory = MagicMock()

    fake_minio = MagicMock()

    with patch(
        "app.interfaces.service_dependencies.get_postgres",
        return_value=fake_postgres,
    ), patch(
        "app.infrastructure.storage.postgres.get_postgres",
        return_value=fake_postgres,
    ), patch(
        "app.interfaces.service_dependencies.get_minio",
        return_value=fake_minio,
    ):
        coord_deps = build_coordinator_runtime_deps(
            app_state=fake_state,
            redis_client=fake_redis,
        )

    # The composition root MUST return the real aggregator, not the legacy
    # NullCoordinatorRuntimeDeps sentinel — otherwise the planner's
    # _build_config() would silently SKIP all 18 coordinator cfg keys.
    assert isinstance(coord_deps, _CoordinatorRuntimeDeps)
    assert not isinstance(coord_deps, _NullCoordinatorRuntimeDeps)

    # Every aggregator field must be non-None — the null-object sentinel
    # has each as a property returning None, the real deps must be populated.
    field_names = (
        "parallel_execution_subgraph",
        "session_service",
        "rehydrate_service",
        "child_runner_starter",
        "mailbox_publisher",
        "mailbox_subscriber",
        "envelope_factory",
        "orchestrator_factory",
        "terminal_waiter",
        "probe_quota",
        "coordinator_limits",
        "session_repository",
        "patch_reducer_service",
        "patch_applier_deps",
        "artifact_storage",
        "cost_rollup_service",
        "coordinator_envelope_store",
        "team_repository",      # [S4 §5] dormant DI — constructed by comp root
        "skill_repository",     # [S4 §5/R7-1] dormant DI — constructed by comp root
    )
    for name in field_names:
        assert getattr(coord_deps, name) is not None, (
            f"_CoordinatorRuntimeDeps.{name} must be populated by composition root"
        )


def test_orchestrator_factory_exposes_build_method():
    """PR-9b-A audit round-1 P1 (Fix 2) — the composition root's
    ``orchestrator_factory`` MUST expose a ``.build(...)`` method.

    The consumer at ``api/app/domain/services/graphs/parallel_execution_subgraph.py``
    (``_first_time_dispatch`` Step 9) invokes ``orchestrator_factory.build(
    coordinator_run_id=..., root_session_id=..., parent_session_id=...,
    emit_event=...)``. If the composition root regresses back to a plain
    callable / function (no ``.build`` attribute), the first flag-on
    dispatch crashes with ``AttributeError`` BEFORE any orchestrator task
    starts. This guard fails fast at composition time.
    """
    from types import SimpleNamespace

    from app.interfaces.service_dependencies import (
        build_coordinator_runtime_deps,
    )

    fake_state = SimpleNamespace()
    fake_redis = MagicMock()
    fake_redis.client = MagicMock()

    fake_postgres = MagicMock()
    fake_postgres.session_factory = MagicMock()

    fake_minio = MagicMock()

    with patch(
        "app.interfaces.service_dependencies.get_postgres",
        return_value=fake_postgres,
    ), patch(
        "app.infrastructure.storage.postgres.get_postgres",
        return_value=fake_postgres,
    ), patch(
        "app.interfaces.service_dependencies.get_minio",
        return_value=fake_minio,
    ):
        coord_deps = build_coordinator_runtime_deps(
            app_state=fake_state,
            redis_client=fake_redis,
        )

    factory = coord_deps.orchestrator_factory
    assert factory is not None
    assert hasattr(factory, "build"), (
        "coord_deps.orchestrator_factory must expose .build(...) — "
        "consumer at parallel_execution_subgraph._first_time_dispatch "
        "calls orchestrator_factory.build(coordinator_run_id=..., "
        "root_session_id=..., parent_session_id=..., emit_event=...) "
        f"and would AttributeError otherwise. Got: {type(factory).__name__}"
    )
    # The app_state mirror must also expose .build for any downstream
    # caller resolving the factory via app.state instead of coord_deps.
    state_factory = getattr(fake_state, "coordinator_orchestrator_factory", None)
    assert state_factory is not None
    assert hasattr(state_factory, "build"), (
        "app_state.coordinator_orchestrator_factory must expose .build(...)"
    )


def test_orchestrator_factory_build_accepts_consumer_kwargs():
    """PR-9b-A audit round-1 P1 (Fix 2) — ``build(...)`` MUST accept the
    full kwarg set the consumer passes: ``parent_session_id``,
    ``coordinator_run_id``, ``root_session_id``, ``emit_event``.

    Pins the consumer-call-shape parity that codex round 1 flagged when
    the previous closure-style factory did not accept ``root_session_id``.
    """
    from types import SimpleNamespace

    from app.interfaces.service_dependencies import (
        build_coordinator_runtime_deps,
    )

    fake_state = SimpleNamespace()
    fake_redis = MagicMock()
    fake_redis.client = MagicMock()

    fake_postgres = MagicMock()
    fake_postgres.session_factory = MagicMock()

    fake_minio = MagicMock()

    with patch(
        "app.interfaces.service_dependencies.get_postgres",
        return_value=fake_postgres,
    ), patch(
        "app.infrastructure.storage.postgres.get_postgres",
        return_value=fake_postgres,
    ), patch(
        "app.interfaces.service_dependencies.get_minio",
        return_value=fake_minio,
    ):
        coord_deps = build_coordinator_runtime_deps(
            app_state=fake_state,
            redis_client=fake_redis,
        )

    factory = coord_deps.orchestrator_factory

    # Patch the orchestrator ctor to a capturing fake — we only care that
    # build(...) accepts the consumer kwarg set, NOT that the orchestrator
    # itself initializes against the fake publisher / subscriber.
    captured = {}

    class _FakeOrchestrator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with patch(
        "app.application.services.coordinator_run_orchestrator."
        "CoordinatorRunOrchestrator",
        _FakeOrchestrator,
    ):
        orch = factory.build(
            parent_session_id="p1",
            coordinator_run_id="r1",
            root_session_id="root1",
            emit_event=None,
        )

    assert orch is not None
    # ctor saw the per-run identity kwargs (parity with PR-7/§11.3 spec).
    assert captured["parent_session_id"] == "p1"
    assert captured["coordinator_run_id"] == "r1"
    assert captured["coordinator_limits"] is coord_deps.coordinator_limits
    assert captured["cost_rollup_service"] is coord_deps.cost_rollup_service
    # ``root_session_id`` is intentionally NOT forwarded into the ctor —
    # the orchestrator receives it via ``run(root_session_id=...)`` at
    # invocation time (coordinator_run_orchestrator.py:230).
    assert "root_session_id" not in captured


def test_real_parent_sandbox_adapter_factory_resolves_and_wraps():
    """[finish-core §5.2 G2 / INV-F2.1 regression] The PRODUCTION factory from
    ``build_coordinator_runtime_deps`` must import + construct
    ``ParentSandboxAdapter`` without ``NameError`` and return an object
    satisfying ``ParentSandboxPort``.

    The construct-only wiring tests above MISS this: a closure free name
    (``ParentSandboxAdapter`` inside ``_parent_sandbox_adapter_factory``)
    resolves at CALL time, not construction time. Those tests only build
    coord_deps (store the un-called closure); the planner ``_build_config``
    tests substitute a Mock factory. So no test invokes the REAL factory —
    a missing module-level import would fire ``NameError`` only on the first
    real coordinator dispatch in production. This test INVOKES the real
    factory so the missing import is caught at test time.
    """
    from types import SimpleNamespace

    from app.infrastructure.external.sandbox.parent_sandbox_adapter import (
        ParentSandboxAdapter,
    )
    from app.interfaces.service_dependencies import (
        build_coordinator_runtime_deps,
    )

    fake_state = SimpleNamespace()
    fake_redis = MagicMock()
    fake_redis.client = MagicMock()

    fake_postgres = MagicMock()
    fake_postgres.session_factory = MagicMock()

    fake_minio = MagicMock()

    with patch(
        "app.interfaces.service_dependencies.get_postgres",
        return_value=fake_postgres,
    ), patch(
        "app.infrastructure.storage.postgres.get_postgres",
        return_value=fake_postgres,
    ), patch(
        "app.interfaces.service_dependencies.get_minio",
        return_value=fake_minio,
    ):
        coord_deps = build_coordinator_runtime_deps(
            app_state=fake_state,
            redis_client=fake_redis,
        )

    # INVOKE the real factory (not a mock) so the closure's free name
    # ``ParentSandboxAdapter`` is actually resolved + constructed. A missing
    # module-level import surfaces here as NameError (proven RED).
    wrapped = coord_deps.parent_sandbox_adapter_factory(
        MagicMock(name="sandbox_handle")
    )

    # The factory must yield the concrete adapter. We assert the concrete type
    # rather than ``isinstance(wrapped, ParentSandboxPort)`` because the Port is
    # a plain ``typing.Protocol`` (NOT ``@runtime_checkable``) — both
    # ``isinstance`` and ``issubclass`` against it raise
    # ``TypeError: Instance and class checks can only be used with
    # @runtime_checkable protocols``. ``ParentSandboxAdapter`` declares
    # ``ParentSandboxPort`` as its base (parent_sandbox_adapter.py:54), so a
    # concrete-type check is the stable proxy for "satisfies the Port".
    assert type(wrapped) is ParentSandboxAdapter, (
        "production parent_sandbox_adapter_factory must return a "
        f"ParentSandboxAdapter; got {type(wrapped).__name__}"
    )
    # Structural sanity: the returned object exposes the full narrow Port
    # surface the domain consumer (dispatch_node / PatchApplier) calls.
    for method_name in (
        "compute_digest",
        "exists",
        "read_file",
        "atomic_write_file",
        "delete_file",
    ):
        assert callable(getattr(wrapped, method_name, None)), (
            f"adapter is missing ParentSandboxPort method: {method_name}"
        )
