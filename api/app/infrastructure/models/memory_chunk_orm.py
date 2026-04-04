"""memory_chunks 表 ORM 模型。"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, PrimaryKeyConstraint, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base

# 向量维度常量——必须与以下三处保持一致：
#   1. MemoryConfig.embedding_dim（app_config.py，运行时配置）
#   2. Alembic migration f1a2b3c4d5e6（DDL 建表）
#   3. DI 注入时传给 OpenAIEmbeddingProvider 的 dimensions 参数
# C4 阶段将在启动时校验 config vs DB 列维度一致性，不匹配则 fail-fast。
# 变更维度需要新 migration 重建列，不能只改配置。
MEMORY_EMBEDDING_DIM = 512


class MemoryChunkModel(Base):
    """记忆分块 ORM 模型。"""

    __tablename__ = "memory_chunks"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_memory_chunks"),
        UniqueConstraint("user_id", "content_hash", name="uq_memory_user_hash"),
    )

    id: Mapped[str] = mapped_column(
        String(255),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    session_id: Mapped[str | None] = mapped_column(
        String(255), ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding = mapped_column(Vector(MEMORY_EMBEDDING_DIM), nullable=True)
    source: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="session_flush",
    )
    # DB column = "metadata" (JSONB); Python attribute = "metadata_"
    # DeclarativeBase.metadata is SQLAlchemy MetaData object (Python-side conflict).
    # mapped_column("metadata", ...) sets DB column name via first positional arg;
    # Python attribute name comes from the class variable name (metadata_).
    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb"),
    )

    def __init__(self, **kwargs):
        # Ensure metadata_ has a Python-side default so callers always see dict,
        # not None, even before flush. server_default only applies at DB INSERT.
        kwargs.setdefault("metadata_", {})
        super().__init__(**kwargs)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()"),
        onupdate=datetime.now,
    )
