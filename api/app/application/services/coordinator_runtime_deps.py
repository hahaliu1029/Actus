"""[PR-9b-A] Lifespan-scoped coordinator runtime deps container.

Aggregates **22 fields** (singletons). ``PlannerReActFlow._build_config()``
projects **21 coordinator cfg keys**: the first 16 deps + 2 per-run keys
(``cancel_event`` + the wrapped ``parent_sandbox``) + ``coordinator_metrics_recorder``
(the 20th field, [C2b rollout WS1b] — the reducer reads it duck-typed for the
run-level run_cost_usd / duration_seconds metrics) + ``team_repository`` +
``skill_repository`` ([S4 §5] dormant DI plumbing — projected for the team
expander (3.3) + planner_node (PR-4); no consumer in 3.1). NOT projected as cfg keys:

- ``coordinator_envelope_store`` — consumed only by ``SupervisorContext`` at
  ``_factory`` time.
- ``parent_sandbox_adapter_factory`` — CONSUMED at ``_build_config`` time to wrap
  the raw ``SandboxHandle`` into a ``ParentSandboxPort`` (transform, not a key).
- ``coordinator_metrics`` — consumed at composition time by the starter ctor
  (budget D10 exhaustion counter); the graph never reads it.

``_NullCoordinatorRuntimeDeps`` is the legacy-test null object (every field →
None); ``_build_config()`` detects it as a sentinel and SKIPS the coordinator
cfg keys entirely.
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
    # [C2b rollout WS1b] CoordinatorMetricsRecorder | None. Tail-defaulted so
    # every pre-existing construction stays source-compatible. Threaded into
    # cfg by _build_config as the 19th coordinator cfg key (the reducer reads
    # it duck-typed) AND independently into the starter ctor at composition
    # time (for the adapter's tool_calls path).
    coordinator_metrics_recorder: object = None
    # [S4 §5] AgentTeamRepository | None. Tail-defaulted so every pre-existing
    # construction stays source-compatible. Read from cfg by _run_parallel_backend
    # (expander) AND planner_node (teaching load). The graph reads it from cfg —
    # it NEVER `new`s a FileTeamRepository (Clean Architecture).
    team_repository: object = None
    # [S4 §5/R7-1] SkillRepository | None. Needed by the expander to resolve
    # member skill slugs → Skill manifests for generated-name resolution (§13).
    # Ordered LAST.
    skill_repository: object = None


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
    @property
    def coordinator_metrics_recorder(self) -> None: return None
    @property
    def team_repository(self) -> None: return None
    @property
    def skill_repository(self) -> None: return None
