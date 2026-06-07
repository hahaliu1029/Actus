import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    PrimaryKeyConstraint,
    SmallInteger,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.schema import conv

from ...domain.models.session import (
    DestroyReason,
    SandboxBinding,
    SandboxBindingState,
    Session,
)
from .base import Base


class SessionModel(Base):
    """会话ORM模型"""

    __tablename__ = "sessions"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_sessions_id"),
        # C3 PR-1 (codex round 11 P2): mirror the migration-level CHECK so
        # ``Base.metadata.create_all`` (used by tests/integration conftest)
        # also gets the constraint. NULL ≡ 'legacy' per R1 P2.3 contract.
        #
        # C3 PR-1 (codex round 12 P2): wrap the name with ``conv()`` to mark it
        # as already-conventionalized. Base.metadata.naming_convention applies
        # ``ck_%(table_name)s_%(constraint_name)s``, so a bare
        # ``name="ck_sessions_subagent_control_plane_valid"`` would double-
        # prefix to ``ck_sessions_ck_sessions_subagent_control_plane_valid`` —
        # mismatching the migration's raw SQL name and creating spurious
        # alembic autogen diffs (drop/recreate). ``conv()`` opts out.
        CheckConstraint(
            "subagent_control_plane IS NULL "
            "OR subagent_control_plane IN ('legacy', 'mailbox')",
            name=conv("ck_sessions_subagent_control_plane_valid"),
        ),
    )

    id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )  # 会话id
    parent_session_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        ForeignKey("sessions.id", ondelete="RESTRICT"),
        nullable=True,
    )  # C1a canonical column — sole lineage field after PR-4 contract drop.
    worker_type: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        server_default=text("'root'::character varying"),
    )  # C1a identity axis root/subagent; CHECK constraints live in c1a migration.
    # C3 PR-1 — control plane discriminator (spec §11.2). NULLABLE with NO
    # server_default; NULL ≡ 'legacy' (pre-C3 rows). Consumers must apply
    # ``coalesce(value, 'legacy')`` semantics; see migration
    # c3_add_mailbox_envelope_audit.py for the canonical contract.
    subagent_control_plane: Mapped[Optional[str]] = mapped_column(
        String(16),
        nullable=True,
        default=None,
    )
    tool_filter_preset: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )  # T12: 持久化 tool_filter 预设名，让 _create_task 重建路径还原 allowlist；CHECK 约束见 migration
    # ── C2 PR-1 coordinator columns (migration c2pr1_coordinator_columns) ──
    # Persist coordinator (run_id, work_unit_id, per-step attempts) so a pod
    # restart can rehydrate in-flight coordinator children without losing the
    # idempotent-retry guard (partial unique index ux_sessions_coordinator_wu)
    # or the attempts counter. NULL on rows that pre-date C2 and on non-
    # coordinator children. See alembic migration for index + CHECK details.
    coordinator_run_id: Mapped[Optional[str]] = mapped_column(
        String(320), nullable=True,
        comment="C2 v1: f'{session_id}:{step_id_hash16}:a{attempt_ix}'",
    )
    work_unit_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True,
        comment="C2 v1: f'{step_id_hash16}.a{attempt_ix}.{i}'",
    )
    coordinator_attempts: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        default=dict,
        comment="C2 v1 per-step attempt counter map",
    )
    sandbox_id: Mapped[str] = mapped_column(String(255), nullable=True)  # 沙箱id
    # ── Sandbox binding columns (lifecycle state machine, I8) ──
    sandbox_state: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default=text("'unbound'::character varying"),
    )
    sandbox_generation: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )
    sandbox_created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    sandbox_destroyed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    sandbox_destroy_reason: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )
    task_id: Mapped[str] = mapped_column(String(255), nullable=True)  # 任务id
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        server_default=text("''::character varying"),
    )  # 会话标题
    unread_message_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )  # 未读消息数
    latest_message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("''::text"),
    )  # 最后一条消息
    latest_message_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=True,
    )  # 最后一条消息时间
    events: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
    )  # 事件列表
    files: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
    )
    memories: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )  # 会话两个Agent的记忆
    status: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        server_default=text("''::character varying"),
    )  # 会话状态
    mode_revision: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=text("0"),
    )  # PE-0: strict-monotonic CAS counter for SessionStateMachine transitions
    user_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )  # 会话所属用户ID
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True,
    )  # 完成时间
    execution_mode: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="foreground",
        server_default="foreground",
    )
    background_reason: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True, default=None
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    last_activity_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    execution_phase: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="running",
        server_default="running",
    )
    retry_budget_remaining: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        default=3,
        server_default="3",
    )
    terminal_reason: Mapped[Optional[str]] = mapped_column(
        String(40), nullable=True, default=None
    )
    suspended_reason: Mapped[Optional[str]] = mapped_column(
        String(40), nullable=True, default=None
    )
    was_background: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        onupdate=datetime.now,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )  # 更新时间
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )  # 创建时间

    # ── Sandbox binding reconstruction for to_domain() ──

    @property
    def sandbox_binding(self) -> SandboxBinding:
        """Reconstruct domain SandboxBinding from flat ORM columns."""
        return SandboxBinding(
            id=self.sandbox_id,
            state=SandboxBindingState(self.sandbox_state),
            generation=self.sandbox_generation,
            created_at=self.sandbox_created_at,
            destroyed_at=self.sandbox_destroyed_at,
            destroy_reason=(
                DestroyReason(self.sandbox_destroy_reason)
                if self.sandbox_destroy_reason
                else None
            ),
        )

    def _apply_sandbox_binding(self, binding: SandboxBinding) -> None:
        """Flatten SandboxBinding into ORM columns."""
        self.sandbox_id = binding.id
        self.sandbox_state = binding.state.value
        self.sandbox_generation = binding.generation
        self.sandbox_created_at = binding.created_at
        self.sandbox_destroyed_at = binding.destroyed_at
        self.sandbox_destroy_reason = (
            binding.destroy_reason.value if binding.destroy_reason else None
        )

    @classmethod
    def from_domain(cls, session: Session) -> "SessionModel":
        """从会话领域模型构建ORM模型"""
        # Exclude sandbox_binding (nested) — we flatten it into ORM columns
        model = cls(
            # 1.基础字段: 使用BaseModel提供的python字典转换格式
            **session.model_dump(
                mode="python",
                exclude={
                    "memories",
                    "files",
                    "events",
                    "updated_at",
                    "created_at",
                    "sandbox_binding",
                },
            ),
            # 2.复杂字段: 使用BaseModel提供的json字典转换格式
            **session.model_dump(
                mode="json",
                include={"memories", "files", "events"},
            ),
        )
        # PR-4 contract: parent_session_id is the sole lineage source. Derive
        # worker_type locally to keep ck_sessions_worker_type_parent_invariant
        # satisfied even if a caller passes the default 'root' on a non-root
        # domain Session.
        model.parent_session_id = session.parent_session_id
        model.worker_type = "subagent" if model.parent_session_id is not None else "root"
        model._apply_sandbox_binding(session.sandbox_binding)
        return model

    def to_domain(self) -> Session:
        """将会话ORM模型转换成领域模型"""
        session = Session.model_validate(self, from_attributes=True)
        return session.model_copy(
            update={
                "parent_session_id": self.parent_session_id,
                "worker_type": self.worker_type or "root",
            }
        )

    def update_from_domain(self, session: Session) -> None:
        """从传递的领域模型更新ORM数据。

        注意：memories 列通过 save_memory / save_skill_graph_state 等
        专用 JSONB patch 方法更新，此处不覆写，避免丢失
        _skill_graph / _summary 等非 Memory 类型的子键。
        """
        # 1.基础字段: Python模式
        base_data = session.model_dump(
            mode="python",
            exclude={
                "memories",
                "files",
                "events",
                "updated_at",
                "created_at",
                "sandbox_binding",
                "status",  # A4-1 §4: status transitions go only through SSM-owned repo mutators
            },
        )

        # 2.复杂字段: JSON模式（排除 memories，由专用方法管理）
        json_data = session.model_dump(
            mode="json",
            include={"files", "events"},
        )

        # 3.合并更新
        for field, value in {**base_data, **json_data}.items():
            setattr(self, field, value)

        # 4. PR-4 contract: parent_session_id is the sole lineage source;
        # derive worker_type to stay ck_sessions_worker_type_parent_invariant-consistent.
        self.parent_session_id = session.parent_session_id
        self.worker_type = "subagent" if self.parent_session_id is not None else "root"

        # 5. Sandbox binding: flatten into ORM columns
        self._apply_sandbox_binding(session.sandbox_binding)
