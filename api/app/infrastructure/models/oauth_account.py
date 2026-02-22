"""OAuth 账户 ORM 模型"""

import uuid
from datetime import datetime
from typing import Any

from app.domain.models.oauth_account import OAuthAccount, OAuthProvider
from sqlalchemy import DateTime, ForeignKey, PrimaryKeyConstraint, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class OAuthAccountModel(Base):
    """OAuth 账户数据 ORM 模型"""

    __tablename__ = "oauth_accounts"
    __table_args__ = (PrimaryKeyConstraint("id", name="pk_oauth_accounts_id"),)

    id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    provider: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    provider_user_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    unionid: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    access_token: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    refresh_token: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )
    raw_data: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
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
    def from_domain(cls, oauth: OAuthAccount) -> "OAuthAccountModel":
        """从领域模型创建 ORM 模型"""
        return cls(
            id=oauth.id,
            user_id=oauth.user_id,
            provider=oauth.provider.value,
            provider_user_id=oauth.provider_user_id,
            unionid=oauth.unionid,
            access_token=oauth.access_token,
            refresh_token=oauth.refresh_token,
            expires_at=oauth.expires_at,
            raw_data=oauth.raw_data,
            created_at=oauth.created_at,
            updated_at=oauth.updated_at,
        )

    def to_domain(self) -> OAuthAccount:
        """将 ORM 模型转换为领域模型"""
        return OAuthAccount(
            id=self.id,
            user_id=self.user_id,
            provider=OAuthProvider(self.provider),
            provider_user_id=self.provider_user_id,
            unionid=self.unionid,
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            expires_at=self.expires_at,
            raw_data=self.raw_data,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    def update_from_domain(self, oauth: OAuthAccount) -> None:
        """从领域模型更新数据"""
        self.access_token = oauth.access_token
        self.refresh_token = oauth.refresh_token
        self.expires_at = oauth.expires_at
        self.raw_data = oauth.raw_data
        self.updated_at = datetime.now()
