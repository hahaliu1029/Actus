"""create user tables

Revision ID: 3a5b7c9d1e2f
Revises: 22abf18600b4
Create Date: 2026-01-26 10:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "3a5b7c9d1e2f"
down_revision: Union[str, Sequence[str], None] = "22abf18600b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 创建用户表
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=True),
        sa.Column("email", sa.String(length=255), nullable=True),
        sa.Column("phone", sa.String(length=20), nullable=True),
        sa.Column("password_hash", sa.String(length=255), nullable=True),
        sa.Column("nickname", sa.String(length=64), nullable=True),
        sa.Column("avatar", sa.String(length=512), nullable=True),
        sa.Column(
            "role",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'user'"),
        ),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'active'"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(0)"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(0)"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_users_id"),
        sa.UniqueConstraint("username", name="uq_users_username"),
        sa.UniqueConstraint("email", name="uq_users_email"),
        sa.UniqueConstraint("phone", name="uq_users_phone"),
    )

    # 创建 OAuth 账户表
    op.create_table(
        "oauth_accounts",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("provider_user_id", sa.String(length=255), nullable=False),
        sa.Column("unionid", sa.String(length=255), nullable=True),
        sa.Column("access_token", sa.Text(), nullable=True),
        sa.Column("refresh_token", sa.Text(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("raw_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(0)"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(0)"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_oauth_accounts_id"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_oauth_accounts_user_id_users",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_oauth_accounts_user_id", "oauth_accounts", ["user_id"])
    op.create_index(
        "ix_oauth_accounts_provider_user_id",
        "oauth_accounts",
        ["provider", "provider_user_id"],
        unique=True,
    )

    # 创建用户工具偏好表
    op.create_table(
        "user_tool_preferences",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("tool_type", sa.String(length=32), nullable=False),
        sa.Column("tool_id", sa.String(length=255), nullable=False),
        sa.Column(
            "enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(0)"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(0)"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_user_tool_preferences_id"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_user_tool_preferences_user_id_users",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "user_id", "tool_type", "tool_id", name="uq_user_tool_preferences_user_tool"
        ),
    )
    op.create_index(
        "ix_user_tool_preferences_user_id", "user_tool_preferences", ["user_id"]
    )

    # 为 files 表添加 user_id 列
    op.add_column("files", sa.Column("user_id", sa.String(length=255), nullable=True))
    op.create_index("ix_files_user_id", "files", ["user_id"])
    op.create_foreign_key(
        "fk_files_user_id_users",
        "files",
        "users",
        ["user_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # 为 sessions 表添加 user_id 列
    op.add_column(
        "sessions", sa.Column("user_id", sa.String(length=255), nullable=True)
    )
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])
    op.create_foreign_key(
        "fk_sessions_user_id_users",
        "sessions",
        "users",
        ["user_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    """Downgrade schema."""
    # 移除 sessions 表的 user_id 列
    op.drop_constraint("fk_sessions_user_id_users", "sessions", type_="foreignkey")
    op.drop_index("ix_sessions_user_id", table_name="sessions")
    op.drop_column("sessions", "user_id")

    # 移除 files 表的 user_id 列
    op.drop_constraint("fk_files_user_id_users", "files", type_="foreignkey")
    op.drop_index("ix_files_user_id", table_name="files")
    op.drop_column("files", "user_id")

    # 删除用户工具偏好表
    op.drop_index(
        "ix_user_tool_preferences_user_id", table_name="user_tool_preferences"
    )
    op.drop_table("user_tool_preferences")

    # 删除 OAuth 账户表
    op.drop_index("ix_oauth_accounts_provider_user_id", table_name="oauth_accounts")
    op.drop_index("ix_oauth_accounts_user_id", table_name="oauth_accounts")
    op.drop_table("oauth_accounts")

    # 删除用户表
    op.drop_table("users")
