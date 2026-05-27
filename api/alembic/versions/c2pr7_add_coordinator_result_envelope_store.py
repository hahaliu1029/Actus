"""C2 PR-7: add coordinator_result_envelope_store table.

Revision ID: c2pr7_envelope_store
Revises: c2pr5_apply_audit
Create Date: 2026-05-25

Spec ref: §13.x (PR-7 crash rehydrate) — persists terminal RESULT_READY /
CANCEL_ACK envelopes so that a coordinator orchestrator restarting after a
pod crash can rebuild the in-flight work-unit map without depending on
volatile in-memory state or replaying every audit/mailbox row from scratch.

**Table purpose.** When a child agent reaches a terminal state the
mailbox supervisor synthesises a single RESULT_READY (success / failure)
or CANCEL_ACK (cooperative cancel) envelope per (coordinator_run_id,
work_unit_id). The supervisor persists that envelope here AT-MOST-ONCE
(via the partial unique index below); the rehydrate path on next boot
reads back the rows for the run and reconstructs the
``CoordinatorRunState.completed_work_units`` map directly — no need to
re-derive terminal outcomes from raw progress / cost / tool events.

**Why ``ux_result_store_run_wu_terminal`` UNIQUE on (run_id, wu_id)?**
Each work-unit has exactly one terminal outcome — RESULT_READY OR
CANCEL_ACK, never both, never duplicated. The UNIQUE index turns the
"persist terminal envelope" call into a self-enforcing idempotency
operation: a buggy supervisor that retries the persist (e.g. after a
mid-write crash) will hit IntegrityError on the second insert, which
the caller catches and logs as "already persisted, no-op". This is
cheaper and stricter than a SELECT-then-INSERT guard.

**Why a separate non-unique ``ix_result_store_run_wu``?** The unique
index covers (run_id, wu_id) for equality lookups already. We keep a
second non-unique index with the same columns because PR-7 rehydrate
queries by run_id alone (``WHERE coordinator_run_id = ...``) — the
planner prefers a regular btree over a partial / unique index when both
columns are predicate-selectable. The non-unique index also stays valid
during rare windows where rows are being moved around for backfill
maintenance.

**Payload is JSONB minimum rehydrate fields only.** The repo layer
filters incoming dicts to ``{outcome, patch_manifest, cost_summary,
needs_authorization_details, final_state}`` before insert — free-text
fields (assistant message, raw tool transcripts) are stripped to keep
the row narrow (target ≤ 64KB) and to avoid persisting model-generated
text into a long-term recovery table. PII regex guards (email / phone)
further degrade the payload to ``{outcome, _pii_redacted: true}`` if a
match slips through.

**Downgrade contract (DESTRUCTIVE):** ``downgrade()`` drops the table
unconditionally. Any persisted terminal envelopes are lost on rollback
— acceptable because the table only matters for crash recovery on the
NEXT boot; live coordinator runs already have their state in memory
plus the audit / mailbox tables.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "c2pr7_envelope_store"
down_revision = "c2pr5_apply_audit"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "coordinator_result_envelope_store",
        sa.Column(
            "id", sa.BigInteger,
            primary_key=True, autoincrement=True,
        ),
        sa.Column(
            "coordinator_run_id", sa.String(length=320),
            nullable=False,
        ),
        sa.Column(
            "work_unit_id", sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "child_session_id", sa.String(length=255),
            nullable=False,
        ),
        # 32 chars covers the two allowed values ('RESULT_READY' /
        # 'CANCEL_ACK', longest at 12 chars). No CHECK constraint —
        # callers validate enum membership; widening is cheap if a
        # third terminal envelope kind is ever added.
        sa.Column(
            "envelope_type", sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "received_at", sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    # Non-unique lookup index for rehydrate's run-scoped scan. Columns
    # mirror the unique index; planner picks whichever it prefers for
    # the workload (typically the non-unique one when the query has
    # no UNIQUE-friendly equality on wu_id).
    op.create_index(
        "ix_result_store_run_wu", "coordinator_result_envelope_store",
        ["coordinator_run_id", "work_unit_id"],
    )
    # At-most-once idempotency guard — a buggy persist retry hits
    # IntegrityError instead of double-inserting.
    op.create_index(
        "ux_result_store_run_wu_terminal",
        "coordinator_result_envelope_store",
        ["coordinator_run_id", "work_unit_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "ux_result_store_run_wu_terminal",
        table_name="coordinator_result_envelope_store",
    )
    op.drop_index(
        "ix_result_store_run_wu",
        table_name="coordinator_result_envelope_store",
    )
    op.drop_table("coordinator_result_envelope_store")
