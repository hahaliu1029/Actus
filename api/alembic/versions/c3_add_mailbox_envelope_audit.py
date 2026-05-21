"""C3 PR-1 — mailbox envelope audit table + subagent_control_plane column.

Revision ID: c3_add_mailbox_envelope_audit
Revises: c1d_drop_sample_session_id
Create Date: 2026-05-21

Spec: docs/superpowers/specs/2026-05-21-c3-mailbox-control-protocol-design.md
      §5.8 (audit schema) + §11.2 (control_plane column).

Notes:
- DestroyReason is a domain Python enum + DB String(64) (api/app/infrastructure/
  models/session.py:73-74) — no PG enum alter needed for the 4 new values.
- **subagent_control_plane is NULLABLE with NO DB default** (R1 P2.3 — spec §14
  risk-register mentioned ``DEFAULT 'legacy'`` but plan intentionally chose
  nullable + canonical "NULL ≡ legacy" semantics): pre-PR-5 rows stay NULL;
  post-PR-5 SessionService picks 'mailbox' or 'legacy' from feature flag.
  Every consumer (helper, supervisor stop check, AST gate) must apply
  ``coalesce(value, 'legacy')`` semantics. This is enforced via:
    1. mailbox_skip_helper._should_skip_mailbox_lifecycle treats None as not-skip
    2. supervisor rollback-stop uses ``(c.subagent_control_plane or 'legacy')``
       (R1 P2.2 fix)
    3. Integration test
       ``test_rollback_with_null_legacy_subagent_children_stops_supervisor``
       (PR-5 Step 7) locks the invariant.
- PRIMARY KEY (parent_session_id, envelope_id) provides consumer-side
  idempotency per spec §5.8 Layer 2.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "c3_add_mailbox_envelope_audit"
down_revision = "c1d_drop_sample_session_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "mailbox_envelope_audit",
        # C3 PR-1 (codex round 15 P2): widen ``parent_session_id`` and
        # ``child_session_id`` to ``String(255)`` to match ``sessions.id``
        # (api/app/infrastructure/models/session.py:55-60). A narrower 64-char
        # cap would raise ``DataError: value too long`` on custom-assigned
        # session ids (e.g. external lineage hooks, fixture-style ids exceeding
        # 64 chars). ``envelope_id`` stays at 64 — ULIDs are 26 chars and 64
        # is generous headroom.
        sa.Column("parent_session_id", sa.String(255), nullable=False),
        sa.Column("envelope_id", sa.String(64), nullable=False),
        sa.Column("child_session_id", sa.String(255), nullable=False),
        sa.Column("type", sa.String(32), nullable=False),
        sa.Column("producer_role", sa.String(32), nullable=False),
        sa.Column("correlation_id", sa.String(64), nullable=True),
        sa.Column(
            "received_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("processing_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("audit_payload", JSONB, nullable=True),
        sa.Column(
            "reclaim_count",
            sa.Integer,
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("last_error", sa.String(2048), nullable=True),
        sa.PrimaryKeyConstraint(
            "parent_session_id", "envelope_id", name="pk_mailbox_envelope_audit"
        ),
    )

    op.create_index(
        "ix_mailbox_envelope_audit_child",
        "mailbox_envelope_audit",
        ["child_session_id"],
    )
    op.create_index(
        "ix_mailbox_envelope_audit_unprocessed",
        "mailbox_envelope_audit",
        ["parent_session_id", "received_at"],
        postgresql_where=sa.text("processed_at IS NULL"),
    )

    op.add_column(
        "sessions",
        sa.Column("subagent_control_plane", sa.String(16), nullable=True),
    )
    # C3 PR-1 (codex round 11 P2): constrain the value set so typos / external
    # writers can't pollute the column. NULL stays the canonical legacy marker
    # per the R1 P2.3 "NULL ≡ legacy" contract — any consumer applies
    # ``coalesce(value, 'legacy')`` semantics; the CHECK keeps the non-NULL
    # domain to {'legacy', 'mailbox'}.
    #
    # Raw SQL (not ``op.create_check_constraint``) on purpose:
    # ``Base.metadata.naming_convention`` (api/app/infrastructure/models/base.py
    # :5-11) would route the ``ck`` template ``ck_%(table_name)s_%(constraint_name)s``
    # and produce ``ck_sessions_ck_sessions_subagent_control_plane_valid``
    # (double prefix). Same pattern as t12_add_tool_filter_preset.py and
    # b4m1_add_cost_records_value_checks.py.
    op.execute(
        "ALTER TABLE sessions ADD CONSTRAINT ck_sessions_subagent_control_plane_valid "
        "CHECK (subagent_control_plane IS NULL "
        "OR subagent_control_plane IN ('legacy', 'mailbox'))"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE sessions DROP CONSTRAINT ck_sessions_subagent_control_plane_valid"
    )
    op.drop_column("sessions", "subagent_control_plane")
    op.drop_index(
        "ix_mailbox_envelope_audit_unprocessed",
        table_name="mailbox_envelope_audit",
    )
    op.drop_index(
        "ix_mailbox_envelope_audit_child",
        table_name="mailbox_envelope_audit",
    )
    op.drop_table("mailbox_envelope_audit")
