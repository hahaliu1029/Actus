"""b6 add conversation_compactions table

Revision ID: b6_add_conversation_compactions
Revises: b4m1_cost_records_value_checks
Create Date: 2026-05-03

Spec: docs/superpowers/specs/2026-05-02-b6-compaction-metadata-persistence-design.md
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "b6_add_conversation_compactions"
down_revision = "b4m1_cost_records_value_checks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversation_compactions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("compaction_id", sa.String(16), nullable=False, unique=True),
        sa.Column(
            "session_id",
            sa.String(255),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("summary_tokens", sa.Integer(), nullable=False),
        sa.Column("first_visible_event_id", sa.String(64), nullable=True),
        sa.Column("last_visible_event_id", sa.String(64), nullable=True),
        sa.Column("pre_compact_checkpoint_id", sa.String(255), nullable=True),
        sa.Column("operations", postgresql.JSONB(), nullable=False),
        sa.Column("parent_compaction_id", sa.String(16), nullable=True),
        sa.Column("tokens_before_total", sa.Integer(), nullable=False),
        sa.Column("tokens_after_total", sa.Integer(), nullable=False),
        sa.Column("messages_removed_total", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "tokens_after_total <= tokens_before_total",
            name="ck_compaction_tokens_monotonic",
        ),
        sa.CheckConstraint(
            "jsonb_array_length(operations) >= 1",
            name="ck_compaction_operations_nonempty",
        ),
    )
    op.create_index(
        "ix_compaction_session_created",
        "conversation_compactions",
        ["session_id", sa.text("created_at DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_compaction_session_created", table_name="conversation_compactions")
    op.drop_table("conversation_compactions")
