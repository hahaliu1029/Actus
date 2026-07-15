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

from sqlalchemy import select, update
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
    ) -> bool:
        async with self._sf() as s:
            values = {
                "status": status,
                "file_count": file_count,
                "total_bytes": total_bytes,
                "failed_at_path": failed_at_path,
                "failed_reason": failed_reason,
                "rollback_status": rollback_status,
                "rollback_failed_paths": (
                    {"paths": list(rollback_failed_paths)}
                    if rollback_failed_paths
                    else None
                ),
                "applied_files": (
                    {"files": applied_files}
                    if applied_files is not None
                    else None
                ),
                "diagnostics": diagnostics,
                "duration_ms": duration_ms,
                "finished_at": datetime.now(timezone.utc),
            }
            stmt = (
                update(CoordinatorApplyAudit)
                .where(
                    CoordinatorApplyAudit.id == audit_id,
                    CoordinatorApplyAudit.status == "in_progress",
                )
                .values(**values)
            )
            result = await s.execute(stmt)
            updated = result.rowcount == 1
            if not updated:
                logger.warning(
                    "update_terminal: audit row %d missing or already "
                    "terminal; CAS skipped (status=%s, rollback_status=%s)",
                    audit_id, status, rollback_status,
                )
            await s.commit()
            return updated

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
