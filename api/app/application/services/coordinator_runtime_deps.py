"""[PR-9b-A] Lifespan-scoped coordinator runtime deps container.

Aggregates **17 fields** (singletons). Of those, ``coordinator_envelope_store``
is consumed only by ``SupervisorContext`` at ``_factory`` time (NOT by the
coordinator graph), so ``PlannerReActFlow._build_config()`` only copies the
**other 16 deps** into ``configurable``. It then ADDS 2 per-run keys
(``cancel_event`` + ``parent_sandbox``) — final total: **16 + 2 = 18
coordinator cfg keys**.

``_NullCoordinatorRuntimeDeps`` is the legacy-test null object that yields
``None`` for every field; ``_build_config()`` detects it as a sentinel and
SKIPS the 18 coordinator cfg keys entirely.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _CoordinatorRuntimeDeps:
    """Process-scoped singletons required by the coordinator dispatch path."""

    parallel_execution_subgraph: object
    session_service: object
    rehydrate_service: object
    child_runner_starter: object
    mailbox_publisher: object
    mailbox_subscriber: object
    envelope_factory: object
    orchestrator_factory: object
    terminal_waiter: object
    probe_quota: object
    coordinator_limits: object
    session_repository: object
    patch_reducer_service: object
    patch_applier_deps: object  # PatchApplierDeps — see Task A6
    artifact_storage: object
    cost_rollup_service: object
    coordinator_envelope_store: object


@dataclass(frozen=True)
class _NullCoordinatorRuntimeDeps:
    """Sentinel null object for legacy non-coordinator tests."""

    @property
    def parallel_execution_subgraph(self) -> None: return None
    @property
    def session_service(self) -> None: return None
    @property
    def rehydrate_service(self) -> None: return None
    @property
    def child_runner_starter(self) -> None: return None
    @property
    def mailbox_publisher(self) -> None: return None
    @property
    def mailbox_subscriber(self) -> None: return None
    @property
    def envelope_factory(self) -> None: return None
    @property
    def orchestrator_factory(self) -> None: return None
    @property
    def terminal_waiter(self) -> None: return None
    @property
    def probe_quota(self) -> None: return None
    @property
    def coordinator_limits(self) -> None: return None
    @property
    def session_repository(self) -> None: return None
    @property
    def patch_reducer_service(self) -> None: return None
    @property
    def patch_applier_deps(self) -> None: return None
    @property
    def artifact_storage(self) -> None: return None
    @property
    def cost_rollup_service(self) -> None: return None
    @property
    def coordinator_envelope_store(self) -> None: return None
