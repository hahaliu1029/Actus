"""ORM model for ``coordinator_apply_audit`` (C2 PR-5 spec §12.4).

Schema mirrors ``api/alembic/versions/c2pr5_add_coordinator_apply_audit``.
See the migration docstring for column semantics + partial unique
``status='success'`` index rationale.

The model is intentionally write-mostly — the application layer
(``PatchApplier``) inserts an ``in_progress`` row on start and updates
to a terminal status on finish. Read paths (PR-7 rehydrate, PR-8 SSE
replay) are simple ``find_latest_for_run`` queries.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import TIMESTAMP, BigInteger, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.models.base import Base


class CoordinatorApplyAudit(Base):
    """One row per PatchApplier invocation per coordinator_run_id.

    See migration docstring for column semantics. JSON-typed columns
    (``rollback_failed_paths`` / ``applied_files`` / ``plan_files_preview``
    / ``diagnostics``) are dict-typed at the ORM layer so the repo
    converts at the boundary.
    """

    __tablename__ = "coordinator_apply_audit"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True,
    )
    coordinator_run_id: Mapped[str] = mapped_column(String(320))
    parent_session_id: Mapped[str] = mapped_column(String(255))
    # [codex R4 P1] SHA-256 hex of canonical PatchApplyPlan JSON; PR-7
    # rehydrate idempotency key (nullable for PR-5 backward compat).
    plan_hash: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True,
    )
    status: Mapped[str] = mapped_column(String(32))
    file_count: Mapped[int] = mapped_column(Integer, default=0)
    total_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    failed_at_path: Mapped[Optional[str]] = mapped_column(
        String(2048), nullable=True,
    )
    failed_reason: Mapped[Optional[str]] = mapped_column(
        String(256), nullable=True,
    )
    rollback_status: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True,
    )
    rollback_failed_paths: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSONB, nullable=True,
    )
    started_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True,
    )
    duration_ms: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
    )
    applied_files: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSONB, nullable=True,
    )
    plan_files_preview: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSONB, nullable=True,
    )
    diagnostics: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSONB, nullable=True,
    )
