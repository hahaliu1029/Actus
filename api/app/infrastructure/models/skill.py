"""Skill ORM 模型"""

import uuid
from datetime import datetime
from typing import Any

from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType
from sqlalchemy import Boolean, DateTime, ForeignKey, PrimaryKeyConstraint, String, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class SkillModel(Base):
    """Skill 数据 ORM 模型"""

    __tablename__ = "skills"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_skills_id"),
        UniqueConstraint("slug", name="uq_skills_slug"),
    )

    id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    slug: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(String(1024), nullable=False, server_default=text("''::character varying"))
    version: Mapped[str] = mapped_column(String(64), nullable=False, server_default=text("'0.1.0'"))
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source_ref: Mapped[str] = mapped_column(String(512), nullable=False)
    runtime_type: Mapped[str] = mapped_column(String(32), nullable=False)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    installed_by: Mapped[str | None] = mapped_column(
        String(255),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
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
    def from_domain(cls, skill: Skill) -> "SkillModel":
        """从领域模型创建 ORM 模型"""
        return cls(
            id=skill.id,
            slug=skill.slug,
            name=skill.name,
            description=skill.description,
            version=skill.version,
            source_type=skill.source_type.value,
            source_ref=skill.source_ref,
            runtime_type=skill.runtime_type.value,
            manifest=skill.manifest,
            enabled=skill.enabled,
            installed_by=skill.installed_by,
            created_at=skill.created_at,
            updated_at=skill.updated_at,
        )

    def to_domain(self) -> Skill:
        """将 ORM 模型转换为领域模型"""
        return Skill(
            id=self.id,
            slug=self.slug,
            name=self.name,
            description=self.description,
            version=self.version,
            source_type=SkillSourceType(self.source_type),
            source_ref=self.source_ref,
            runtime_type=SkillRuntimeType(self.runtime_type),
            manifest=self.manifest,
            enabled=self.enabled,
            installed_by=self.installed_by,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    def update_from_domain(self, skill: Skill) -> None:
        """从领域模型更新数据"""
        self.slug = skill.slug
        self.name = skill.name
        self.description = skill.description
        self.version = skill.version
        self.source_type = skill.source_type.value
        self.source_ref = skill.source_ref
        self.runtime_type = skill.runtime_type.value
        self.manifest = skill.manifest
        self.enabled = skill.enabled
        self.installed_by = skill.installed_by
        self.updated_at = datetime.now()
