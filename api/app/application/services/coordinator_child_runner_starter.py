"""[PR-9b-A] CoordinatorChildRunnerStarter Protocol + concrete adapter.

Bridge between parallel_execution_subgraph.dispatch_node and
ChildAgentTaskRunnerFactory.build(...) -> CoordinatorChildRunner.run_work_unit(...).

start(...) is FIRE-THEN-TRACK: returns control to dispatch as soon as the task
is created. _active_tasks holds references for the orchestrator to cancel/await.
The done-callback ALWAYS pops first, then guards task.cancelled() BEFORE
task.exception() (Task.exception() raises CancelledError on cancelled tasks).
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Protocol

from app.application.services.coordinator_child_runner import CoordinatorChildRunner
from app.application.services.cost_callback_factory import (
    build_supervisor_aware_callback_handler,
)
from app.domain.models.tool_filter_presets import COORDINATOR_STEP_PRESET
from app.infrastructure.external.sandbox.parent_sandbox_adapter import (
    ParentSandboxAdapter,
)
from app.domain.models.work_unit import PathLease  # PathLease lives on work_unit, NOT child_permission_context
from app.domain.services.permission.child_permission_context import (
    ChildBudget,
    ChildPermissionContext,
    ChildRuntimeCap,
    SpawnManifest,
)

if TYPE_CHECKING:
    from app.application.services.child_agent_runner_factory import (
        ChildAgentTaskRunnerFactory,
    )
    from app.application.services.coordinator_envelope_factory import (
        CoordinatorEnvelopeFactory,
    )
    from app.application.services.cost_rollup_service import CostRollupService
    from app.domain.external.artifact_storage import ArtifactStoragePort
    from app.domain.external.mailbox_publisher import MailboxPublisher
    from app.domain.external.mailbox_subscriber import MailboxSubscriber
    from app.domain.repositories.coordinator_result_envelope_store_repository import (
        CoordinatorResultEnvelopeStoreRepository,
    )
    from app.domain.repositories.session_repository import SessionRepository
    from app.domain.services.coordinator_limits import CoordinatorLimits

logger = logging.getLogger(__name__)


class CoordinatorChildRunnerStarter(Protocol):
    """Fire-then-track spawn of a coordinator-child agent task."""

    async def start(
        self,
        *,
        coordinator_run_id: str,
        work_unit: Any,
        child_session_id: str,
        spawn_manifest_ref: str,
        cancel_event: asyncio.Event,
        root_session_id: str,
        parent_session_id: str,
        parent_sandbox: Any,  # per-run; comes from cfg["parent_sandbox"]
        user_id: str,
    ) -> None: ...


def _decode_path_lease(pl: dict[str, Any]) -> PathLease:
    """[PR-9b-A INV-A11] PathLease wire-payload decoder.

    The canonical wire format produced by
    ``parallel_execution_subgraph._serialize_spawn_manifest`` (api/app/domain/
    services/graphs/parallel_execution_subgraph.py:203) emits
    ``lease.model_dump()`` which always contains the four PathLease fields:
    ``{path, op, base_digest, seed_content_ref}``. ``op`` is required by the
    PathLease pydantic model (api/app/domain/models/work_unit.py:34) so the
    producer can never legitimately omit it.

    We deliberately do NOT default a missing ``op`` — silently coercing a
    malformed manifest to e.g. ``"modify"`` (the most permissive value) would
    let a future schema drift escape the gate that ``PathLease``'s
    ``_add_op_invariants`` validator is supposed to enforce. Missing ``op``
    raises ``KeyError`` here so the dispatch path fails loudly with a clear
    pointer to the wire-schema bug.

    Only the four known PathLease fields are forwarded; any extra keys on the
    payload (e.g. legacy fixtures that included a non-existent ``expires_at``)
    are silently dropped. ``lease_expiry`` belongs on
    ``ChildPermissionContext``, not on individual leases, so dropping the
    extras is the correct schema interpretation.
    """
    return PathLease(
        path=pl["path"],
        op=pl["op"],
        base_digest=pl.get("base_digest"),
        seed_content_ref=pl.get("seed_content_ref"),
    )


class DefaultCoordinatorChildRunnerStarter:
    """Concrete starter — see module docstring for ownership semantics."""

    def __init__(
        self,
        *,
        runner_factory: "ChildAgentTaskRunnerFactory",
        mailbox_publisher: "MailboxPublisher",
        mailbox_subscriber: "MailboxSubscriber",
        envelope_factory: "CoordinatorEnvelopeFactory",
        session_repository: "SessionRepository",
        coordinator_envelope_store: "CoordinatorResultEnvelopeStoreRepository",
        cost_rollup_service: "CostRollupService",
        artifact_storage: "ArtifactStoragePort",
        coordinator_limits: "CoordinatorLimits",
        sandbox_lifecycle_service: Any,  # _DeferredLifecycle proxy or SandboxLifecycleService
        resolve_child_runner_deps: Any = None,  # () -> ChildRunnerSharedDeps; lazy (supervisor+uow for cost handler). Default None is INTENTIONAL: F1.7 build_coordinator_runtime_deps constructs this starter with None for the 2-kwarg comp-root test callers that never dispatch a child (so .start()/_resolve_child_runner_deps() is never reached). A real dispatch path is always wired by F1.7 -- do NOT add a hard __init__ guard (it would break those callers).
    ) -> None:
        self._runner_factory = runner_factory
        self._mailbox_publisher = mailbox_publisher
        self._mailbox_subscriber = mailbox_subscriber
        self._envelope_factory = envelope_factory
        self._session_repository = session_repository
        self._coordinator_envelope_store = coordinator_envelope_store
        self._cost_rollup_service = cost_rollup_service
        self._artifact_storage = artifact_storage
        self._coordinator_limits = coordinator_limits
        self._sandbox_lifecycle_service = sandbox_lifecycle_service
        self._resolve_child_runner_deps = resolve_child_runner_deps
        self._active_tasks: dict[str, asyncio.Task] = {}

    async def start(
        self,
        *,
        coordinator_run_id: str,
        work_unit: Any,
        child_session_id: str,
        spawn_manifest_ref: str,
        cancel_event: asyncio.Event,
        root_session_id: str,
        parent_session_id: str,
        parent_sandbox: Any,  # per-run; provided by dispatch_node from cfg
        user_id: str,
    ) -> None:
        # 1. Fetch + decode SpawnManifest.
        raw = await self._artifact_storage.get_bytes(spawn_manifest_ref)
        data = json.loads(raw)
        spawn_manifest = SpawnManifest(
            allowed_tools=frozenset(data["allowed_tools"]),
            path_leases=tuple(_decode_path_lease(pl) for pl in data["write_lease"]),
            runtime_caps=frozenset(
                ChildRuntimeCap(c) for c in data.get("runtime_caps", [])
            ),
        )
        # 2. session_mode_revision.
        session_mode_revision = await self._session_repository.read_mode_revision(
            child_session_id
        )
        # 3. ChildBudget from coordinator limits.
        limits = self._coordinator_limits
        budget = ChildBudget(
            max_tool_calls=limits.max_tool_calls_per_child,
            max_token_cost_usd=limits.max_token_cost_usd_per_child,
            max_wallclock_seconds=limits.max_wallclock_seconds_per_child,
        )
        # 4. ChildPermissionContext (7 fields + lease_expiry).
        child_permission_context = ChildPermissionContext(
            parent_session_id=parent_session_id,
            child_session_id=child_session_id,
            coordinator_run_id=coordinator_run_id,
            work_unit_id=work_unit.work_unit_id,
            spawn_manifest=spawn_manifest,
            session_mode_revision=session_mode_revision,
            budget=budget,
            lease_expiry=None,
        )
        # 5-11. Provision + spawn under an M1-safe leak-guard. Steps 1-4
        # (manifest decode / revision / budget / ChildPermissionContext)
        # stay OUTSIDE: a malformed manifest or revision-read failure must
        # propagate to dispatch loudly — there is no sandbox to reap yet.
        try:
            # 5. Provision the per-child sandbox (A1) + browser + cost handler.
            child_handle = await self._sandbox_lifecycle_service.bind_new(
                child_session_id, user_id=user_id
            )
            child_browser = await child_handle.get_browser()
            child_sandbox_port = ParentSandboxAdapter(child_handle)
            _shared = self._resolve_child_runner_deps()  # lazy resolve (post-lifespan)
            cost_callback_handler = build_supervisor_aware_callback_handler(
                _shared.execution_supervisor,
                child_session_id,
                user_id,
                _shared.uow_factory,
            )
            # 6. Build the adapter-wrapped inner runner with per-child deps.
            built = await self._runner_factory.build(
                child_session_id=child_session_id,
                child_permission_context=child_permission_context,
                tool_filter_preset=COORDINATOR_STEP_PRESET,
                cancel_event=cancel_event,
                sandbox=child_handle,
                browser=child_browser,
                user_id=user_id,
                cost_callback_handler=cost_callback_handler,
            )
            # 7. CoordinatorChildRunner — child_sandbox Port for seed/extraction.
            child_runner = CoordinatorChildRunner(
                cancel_event=cancel_event,
                inner_runner=built.runner,
                publisher=self._mailbox_publisher,
                parent_sandbox=parent_sandbox,
                child_sandbox=child_sandbox_port,
                artifact_storage=self._artifact_storage,
                envelope_factory=self._envelope_factory,
                parent_session_id=parent_session_id,
                coordinator_run_id=coordinator_run_id,
                mailbox_subscriber=self._mailbox_subscriber,
            )
            # 8. Spawn background task (fire-then-track).
            task = asyncio.create_task(child_runner.run_work_unit(
                coordinator_run_id=coordinator_run_id,
                work_unit=work_unit,
                child_session_id=child_session_id,
                spawn_manifest=spawn_manifest,
                cancel_event=cancel_event,
                root_session_id=root_session_id,
            ))
            # 9-11. Name, register, done-callback.
            task.set_name(f"coord-child-{child_session_id}")
            self._active_tasks[child_session_id] = task
            task.add_done_callback(self._on_task_done)
        except BaseException as exc:  # noqa: BLE001
            # [finish-core §5.1.3 INV-F1.4a] M1-safe: never destroy the child
            # sandbox here. Publish a FAILED terminal envelope so the
            # MailboxSupervisor cascade reaps the orphan; swallow so dispatch
            # keeps starting siblings and the orchestrator observes the terminal.
            logger.warning(
                "coordinator child start failed for %s (wu=%s); publishing "
                "FAILED envelope, NO self-destroy (M1)",
                child_session_id,
                getattr(work_unit, "work_unit_id", "?"),
                exc_info=exc,
            )
            try:
                failure_publisher = CoordinatorChildRunner(
                    cancel_event=cancel_event,
                    publisher=self._mailbox_publisher,
                    envelope_factory=self._envelope_factory,
                    parent_session_id=parent_session_id,
                    coordinator_run_id=coordinator_run_id,
                    mailbox_subscriber=self._mailbox_subscriber,
                )
                await failure_publisher._finalize_failed(
                    coordinator_run_id, work_unit, child_session_id, exc,
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "leak-guard FAILED-publish ALSO failed for child=%s (wu=%s); "
                    "orphan sandbox will be reaped by supervisor timeout "
                    "(original start failure logged above)",
                    child_session_id,
                    getattr(work_unit, "work_unit_id", "?"),
                )
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                # [R3 hardening] FAILED envelope is published above so the
                # supervisor reaps the bound sandbox; never swallow a true
                # cancellation/interrupt — propagate it.
                raise
            return

    def _on_task_done(self, task: asyncio.Task) -> None:
        name = task.get_name()
        if name.startswith("coord-child-"):
            child_session_id = name.removeprefix("coord-child-")
            self._active_tasks.pop(child_session_id, None)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "Coordinator child task crashed unhandled: name=%s",
                name,
                exc_info=exc,
            )
