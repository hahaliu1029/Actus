"""[PR-9b-A Task A6] Lifespan-scoped deps for per-run PatchApplier construction.

Spec §5.1.4 originally specified a singleton ``patch_applier`` cfg key. The
production reality (codex R3 audit) requires PER-RUN construction so the
``emit_event`` callable can close over the per-stream ``event_queue``.

This dataclass bundles the THREE lifespan-scoped ports that PatchApplier
needs at ctor time (``snapshot_store`` / ``audit_repo`` / ``redis``); the
fourth port (``emit_event``) is bound per-run inside
``main_graph._run_parallel_backend`` to the active ``cfg["event_queue"]``
via ``put_nowait`` (synchronous, cancellation-safe — see INV-A3 / INV-A4).

The composition root constructs one of these in lifespan, and the planner
graph injects it into ``configurable["patch_applier_deps"]``. PatchApplier
itself is NOT a singleton — it's built per-coordinator-run.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PatchApplierDeps:
    """Lifespan-scoped deps required by ``PatchApplier.__init__``.

    Field types are intentionally ``object`` rather than concrete classes so
    this module doesn't pull the real ports' import graph (pydantic / redis
    / SQLAlchemy) into the planner's hot path. Production callers wire the
    real ``RollbackSnapshotStore`` / ``CoordinatorApplyAuditRepository`` /
    ``redis.asyncio.Redis`` instances.
    """

    snapshot_store: object  # RollbackSnapshotStore
    audit_repo: object      # CoordinatorApplyAuditRepository
    redis: object           # redis.asyncio.Redis
