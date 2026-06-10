"""C2b child-row reaper: additive partial index for find_running_mailbox_children.

Keeps the once-per-boot startup sweep off a seq-scan as ``sessions`` grows.
Covers 4 of the query's 5 clauses (spec §4.3 predicate verbatim);
``parent_session_id IS NOT NULL`` is deliberately omitted — it is redundant
under ``ck_sessions_worker_type_parent_invariant`` (subagent ⇒ parent NOT NULL),
and the query's stricter WHERE still implies this index predicate, so the
planner can use it. Additive + droppable — no column/enum/CHECK change.

Revision ID: c2b1_mailbox_running_child_idx
Revises: pe4d2_drop_tool_approval_rules
Create Date: 2026-06-09
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "c2b1_mailbox_running_child_idx"
down_revision = "pe4d2_drop_tool_approval_rules"
branch_labels = None
depends_on = None

_INDEX_NAME = "ix_sessions_mailbox_running"


def upgrade() -> None:
    op.create_index(
        _INDEX_NAME,
        "sessions",
        ["id"],
        unique=False,
        postgresql_where=sa.text(
            "worker_type = 'subagent' "
            "AND subagent_control_plane = 'mailbox' "
            "AND status = 'running' "
            "AND execution_mode = 'foreground'"
        ),
    )


def downgrade() -> None:
    op.drop_index(_INDEX_NAME, table_name="sessions")
