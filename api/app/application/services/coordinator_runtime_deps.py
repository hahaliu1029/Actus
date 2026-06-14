"""[PR-9b-A] Lifespan-scoped coordinator runtime deps container.

Aggregates **19 fields** (singletons). ``PlannerReActFlow._build_config()``
copies the **first 16 deps** into ``configurable``. The 17th
(``coordinator_envelope_store``), 18th (``parent_sandbox_adapter_factory``)
and 19th (``coordinator_metrics`` — C2b budget D10, consumed by the starter
ctor at composition time) are NOT copied as their own cfg keys:

- ``coordinator_envelope_store`` is consumed only by ``SupervisorContext`` at
  ``_factory`` time (NOT by the coordinator graph).
- ``parent_sandbox_adapter_factory`` is CONSUMED at ``_build_config`` time to
  WRAP the per-run raw ``SandboxHandle`` into a ``ParentSandboxPort`` — it is
  the transform applied to the ``parent_sandbox`` value, not a key of its own.
  (finish-core §5.2 G2: keeps the domain flow infra-free — the flow injects a
  factory instead of importing ``ParentSandboxAdapter`` from infrastructure.)

``_build_config()`` then ADDS 2 per-run keys (``cancel_event`` +
``parent_sandbox``) — final total: **16 + 2 = 18 coordinator cfg keys**
(unchanged by the 18th dep; the factory is consumed, not emitted).

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
    parent_sandbox_adapter_factory: object  # Callable[[SandboxHandle], ParentSandboxPort]
    # [C2b budget D10] CoordinatorMetrics instrument bundle. The ONLY
    # defaulted field (None) so every pre-existing 18-field construction —
    # tests and composition alike — stays source-compatible. Consumed at the
    # composition root to thread into the starter ctor; NOT projected as a
    # cfg key by _build_config (the graph never reads it).
    coordinator_metrics: object = None


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
    @property
    def parent_sandbox_adapter_factory(self) -> None: return None
    @property
    def coordinator_metrics(self) -> None: return None
