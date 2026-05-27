"""DB-backed CoordinatorApplyAuditRepository (C2 PR-5 spec §12.4).

Short-lived sessions per call — the PatchApplier runs as the
coordinator orchestrator's background task and has no enclosing UoW
span. Each insert/update commits in its own session; failures bubble
to the applier's finalizer where they're logged.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.repositories.coordinator_apply_audit_repository import (
    CoordinatorApplyAuditRepository,
)
from app.infrastructure.models.coordinator_apply_audit import (
    CoordinatorApplyAudit,
)


logger = logging.getLogger(__name__)


class DbCoordinatorApplyAuditRepository(CoordinatorApplyAuditRepository):
    """DB impl. Mirrors the short-session pattern used by
    ``DbMailboxEnvelopeAuditRepository``."""

    def __init__(
        self, session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._sf = session_factory

    async def insert_in_progress(
        self,
        *,
        coordinator_run_id: str,
        parent_session_id: str,
        plan_files_preview: list[str],
        plan_hash: Optional[str] = None,
    ) -> int:
        async with self._sf() as s:
            row = CoordinatorApplyAudit(
                coordinator_run_id=coordinator_run_id,
                parent_session_id=parent_session_id,
                plan_hash=plan_hash,
                status="in_progress",
                started_at=datetime.now(timezone.utc),
                plan_files_preview={"paths": plan_files_preview},
            )
            s.add(row)
            await s.commit()
            await s.refresh(row)
            return row.id

    async def update_terminal(
        self,
        audit_id: int,
        *,
        status: str,
        file_count: int = 0,
        total_bytes: int = 0,
        failed_at_path: Optional[str] = None,
        failed_reason: Optional[str] = None,
        rollback_status: Optional[str] = None,
        rollback_failed_paths: Optional[list[str]] = None,
        applied_files: Optional[list[dict[str, Any]]] = None,
        diagnostics: Optional[dict[str, Any]] = None,
        duration_ms: int = 0,
    ) -> None:
        async with self._sf() as s:
            row = await s.get(CoordinatorApplyAudit, audit_id)
            if row is None:
                # Audit row vanished — programmer error or someone
                # truncated the table mid-run. We no-op rather than
                # raise (the applier's actual apply outcome is already
                # returned to the orchestrator; raising here would mask
                # the real result and could leave the orchestrator
                # observing a "rolled back" sandbox but no audit trail
                # on the result). [codex R3 P2#1] WARN so the missing
                # row is observable instead of an entirely silent loss.
                logger.warning(
                    "update_terminal: audit row %d not found; "
                    "skipping terminal update (status=%s, "
                    "rollback_status=%s). Apply outcome already "
                    "returned to caller — investigate the missing row.",
                    audit_id, status, rollback_status,
                )
                return
            row.status = status
            row.file_count = file_count
            row.total_bytes = total_bytes
            row.failed_at_path = failed_at_path
            row.failed_reason = failed_reason
            row.rollback_status = rollback_status
            # JSONB columns: wrap lists/dicts at the boundary so the
            # ORM sees the dict-typed Mapped declaration.
            row.rollback_failed_paths = (
                {"paths": list(rollback_failed_paths)}
                if rollback_failed_paths
                else None
            )
            row.applied_files = (
                {"files": applied_files} if applied_files is not None else None
            )
            row.diagnostics = diagnostics
            row.duration_ms = duration_ms
            row.finished_at = datetime.now(timezone.utc)
            await s.commit()

    async def find_latest_for_run(
        self, coordinator_run_id: str,
    ) -> Optional[CoordinatorApplyAudit]:
        async with self._sf() as s:
            stmt = (
                select(CoordinatorApplyAudit)
                .where(
                    CoordinatorApplyAudit.coordinator_run_id
                    == coordinator_run_id
                )
                .order_by(CoordinatorApplyAudit.started_at.desc())
                .limit(1)
            )
            return (await s.execute(stmt)).scalar_one_or_none()
