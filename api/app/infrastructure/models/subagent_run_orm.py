"""subagent_runs 表 ORM 模型（C4.1a spec §4）。

C4.1a 观测面：每条 = 一次 subagent run（coordinator child / research child /
未来 REMOTE）的投影结果。列扁平化自 SubagentRunResult 以支持 SQL 查询；JSONB
artifacts 存 ArtifactRef 列表。无 FK（observation 表 + REMOTE-ready：
child_session_id 可能是无本地 session 行的远端 id）。幂等键 =
UNIQUE(child_session_id)（全局唯一；见 spec §4 为何非 source_ref/work_unit_id）。
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class SubagentRunModel(Base):
    """subagent_runs 行 ORM。"""

    __tablename__ = "subagent_runs"
    __table_args__ = (
        UniqueConstraint("child_session_id", name="uq_subagent_runs_child_session_id"),
        Index("ix_subagent_runs_parent_session_id", "parent_session_id"),
    )

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    runtime: Mapped[str] = mapped_column(String(16), nullable=False)
    lifecycle_state: Mapped[str] = mapped_column(String(32), nullable=False)
    terminal_outcome: Mapped[str | None] = mapped_column(String(32), nullable=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''"))
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    parent_session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    child_session_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    cost_authoritative: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false"),
    )
    cost_total_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_total_output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_total_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_tool_call_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_source: Mapped[str] = mapped_column(String(32), nullable=False)
    artifacts: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()"),
    )
