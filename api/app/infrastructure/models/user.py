"""用户 ORM 模型"""

import uuid
from datetime import datetime

from app.domain.models.user import User, UserRole, UserStatus
from sqlalchemy import DateTime, Enum, PrimaryKeyConstraint, String, text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class UserModel(Base):
    """用户数据 ORM 模型"""

    __tablename__ = "users"
    __table_args__ = (PrimaryKeyConstraint("id", name="pk_users_id"),)

    id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    username: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        unique=True,
    )
    email: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        unique=True,
    )
    phone: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        unique=True,
    )
    password_hash: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    nickname: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    avatar: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
    )
    role: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default=text("'user'"),
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default=text("'active'"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        onupdate=datetime.now,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )

    @classmethod
    def from_domain(cls, user: User) -> "UserModel":
        """从领域模型创建 ORM 模型"""
        return cls(
            id=user.id,
            username=user.username,
            email=user.email,
            phone=user.phone,
            password_hash=user.password_hash,
            nickname=user.nickname,
            avatar=user.avatar,
            role=user.role.value,
            status=user.status.value,
            created_at=user.created_at,
            updated_at=user.updated_at,
        )

    def to_domain(self) -> User:
        """将 ORM 模型转换为领域模型"""
        return User(
            id=self.id,
            username=self.username,
            email=self.email,
            phone=self.phone,
            password_hash=self.password_hash,
            nickname=self.nickname,
            avatar=self.avatar,
            role=UserRole(self.role),
            status=UserStatus(self.status),
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    def update_from_domain(self, user: User) -> None:
        """从领域模型更新数据"""
        self.username = user.username
        self.email = user.email
        self.phone = user.phone
        self.password_hash = user.password_hash
        self.nickname = user.nickname
        self.avatar = user.avatar
        self.role = user.role.value
        self.status = user.status.value
        self.updated_at = datetime.now()
