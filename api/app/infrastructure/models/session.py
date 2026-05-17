import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
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
    __table_args__ = (PrimaryKeyConstraint("id", name="pk_sessions_id"),)

    id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )  # 会话id
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
        model._apply_sandbox_binding(session.sandbox_binding)
        return model

    def to_domain(self) -> Session:
        """将会话ORM模型转换成领域模型"""
        return Session.model_validate(self, from_attributes=True)

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

        # 4. Sandbox binding: flatten into ORM columns
        self._apply_sandbox_binding(session.sandbox_binding)
