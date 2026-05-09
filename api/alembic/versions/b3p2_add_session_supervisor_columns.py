"""Add B3 supervisor fields to sessions.

Revision ID: b3p2_supervisor_columns
Revises: b6_add_conversation_compactions
Create Date: 2026-05-08
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "b3p2_supervisor_columns"
down_revision = "b6_add_conversation_compactions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "sessions",
        sa.Column(
            "execution_mode",
            sa.String(length=20),
            nullable=False,
            server_default="foreground",
        ),
    )
    op.add_column(
        "sessions",
        sa.Column("background_reason", sa.String(length=20), nullable=True),
    )
    op.add_column(
        "sessions",
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "sessions",
        sa.Column("last_activity_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "sessions",
        sa.Column(
            "execution_phase",
            sa.String(length=20),
            nullable=False,
            server_default="running",
        ),
    )
    op.add_column(
        "sessions",
        sa.Column(
            "retry_budget_remaining",
            sa.SmallInteger(),
            nullable=False,
            server_default="3",
        ),
    )
    op.add_column(
        "sessions",
        sa.Column("terminal_reason", sa.String(length=40), nullable=True),
    )
    op.add_column(
        "sessions",
        sa.Column("suspended_reason", sa.String(length=40), nullable=True),
    )
    op.add_column(
        "sessions",
        sa.Column(
            "was_background",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )

    op.create_check_constraint(
        "execution_mode",
        "sessions",
        "execution_mode IN ('foreground', 'background')",
    )
    op.create_check_constraint(
        "background_reason",
        "sessions",
        "background_reason IS NULL OR background_reason IN ('explicit', 'auto_degrade')",
    )
    op.create_check_constraint(
        "execution_phase",
        "sessions",
        "execution_phase IN ('running', 'recovering', 'idle', 'suspended', 'terminating', 'terminated')",
    )
    op.create_check_constraint(
        "retry_budget_range",
        "sessions",
        "retry_budget_remaining >= 0 AND retry_budget_remaining <= 3",
    )
    op.create_check_constraint(
        "terminal_reason",
        "sessions",
        "terminal_reason IS NULL OR terminal_reason IN "
        "('natural', 'user_cancel', 'server_restart', 'resume_state_lost', "
        "'watchdog_timeout')",
    )
    op.create_check_constraint(
        "suspended_reason",
        "sessions",
        "suspended_reason IS NULL OR suspended_reason IN "
        "('bg_idle_timeout', 'server_restart')",
    )
    op.create_check_constraint(
        "bg_requires_expires_at",
        "sessions",
        "(execution_mode != 'background') OR (expires_at IS NOT NULL)",
    )
    op.create_check_constraint(
        "bg_requires_reason",
        "sessions",
        "(execution_mode != 'background') OR (background_reason IS NOT NULL)",
    )
    op.create_check_constraint(
        "fg_no_bg_fields",
        "sessions",
        "(execution_mode != 'foreground') OR (expires_at IS NULL AND background_reason IS NULL)",
    )
    op.create_check_constraint(
        "suspended_requires_reason",
        "sessions",
        "(execution_phase != 'suspended') OR (suspended_reason IS NOT NULL)",
    )
    op.create_check_constraint(
        "terminal_requires_reason",
        "sessions",
        "(execution_phase NOT IN ('terminating', 'terminated')) OR (terminal_reason IS NOT NULL)",
    )

    op.create_index(
        "idx_sessions_user_bg_recent",
        "sessions",
        ["user_id", sa.text("last_activity_at DESC")],
        postgresql_where=sa.text("execution_mode = 'background'"),
    )
    op.create_index(
        "idx_sessions_expires_at",
        "sessions",
        ["expires_at"],
        postgresql_where=sa.text("expires_at IS NOT NULL"),
    )
    op.create_index(
        "idx_sessions_phase_running",
        "sessions",
        ["execution_phase", "status"],
        postgresql_where=sa.text("execution_phase IN ('running', 'recovering')"),
    )


def downgrade() -> None:
    op.drop_index("idx_sessions_phase_running", table_name="sessions")
    op.drop_index("idx_sessions_expires_at", table_name="sessions")
    op.drop_index("idx_sessions_user_bg_recent", table_name="sessions")
    op.drop_constraint(op.f("ck_sessions_terminal_requires_reason"), "sessions", type_="check")
    op.drop_constraint(op.f("ck_sessions_suspended_requires_reason"), "sessions", type_="check")
    op.drop_constraint(op.f("ck_sessions_fg_no_bg_fields"), "sessions", type_="check")
    op.drop_constraint(op.f("ck_sessions_bg_requires_reason"), "sessions", type_="check")
    op.drop_constraint(op.f("ck_sessions_bg_requires_expires_at"), "sessions", type_="check")
    op.drop_constraint(op.f("ck_sessions_suspended_reason"), "sessions", type_="check")
    op.drop_constraint(op.f("ck_sessions_terminal_reason"), "sessions", type_="check")
    op.drop_constraint(op.f("ck_sessions_retry_budget_range"), "sessions", type_="check")
    op.drop_constraint(op.f("ck_sessions_execution_phase"), "sessions", type_="check")
    op.drop_constraint(op.f("ck_sessions_background_reason"), "sessions", type_="check")
    op.drop_constraint(op.f("ck_sessions_execution_mode"), "sessions", type_="check")
    op.drop_column("sessions", "was_background")
    op.drop_column("sessions", "suspended_reason")
    op.drop_column("sessions", "terminal_reason")
    op.drop_column("sessions", "retry_budget_remaining")
    op.drop_column("sessions", "execution_phase")
    op.drop_column("sessions", "last_activity_at")
    op.drop_column("sessions", "expires_at")
    op.drop_column("sessions", "background_reason")
    op.drop_column("sessions", "execution_mode")
