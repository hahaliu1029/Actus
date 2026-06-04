"""PE-4d2 — drop the legacy tool_approval_rules table (guarded, forward-only)

Revision ID: pe4d2_drop_tool_approval_rules
Revises: c2pr7_envelope_store
Create Date: 2026-06-03

Per the PE-4 design (docs/superpowers/specs/2026-06-03-pe-4-legacy-flag-
retirement-design.md §6), the legacy ``tool_approval_rules`` table is the last
piece of the pre-PermissionEngine approval stack. PE-4d1 already stopped reading
it (ApprovalStateReader Priority-5 + SessionLegacyRuleQuery + the
``legacy_rule_fallback`` flag are gone), so the table is now write-only dead
weight. This migration drops it.

Defensive count-and-abort guard:
  Actus auto-runs ``alembic upgrade head`` on startup (api/app/main.py). The
  ``backfill_approval_grants`` CLI is a MANUAL step that is NOT in the alembic
  chain. To avoid silently destroying rows that were never migrated to the new
  ``tool_approval_grants`` table, this migration first counts legacy rules that
  have no corresponding grant (same rule→grant matching key as the backfill
  CLI's ``_SELECT_LEGACY_SQL`` —
  api/app/cli/backfill_approval_grants.py:110-118 — namely user_id / tool_name /
  primary_arg=command_pattern / dir_arg=COALESCE(NULLIF(dir_pattern,''),'') /
  scope='always'). If any un-migrated rules exist it ABORTS with instructions to
  run the backfill CLI; otherwise it drops the table. In this solo-dev repo the
  count is expected to be 0 and the table drops cleanly.

The backfill CLI (and its ORM/repo/model) are intentionally KEPT at this
revision so the abort message above points to a CLI that still exists; they are
deleted in PE-4d3, only after this table is gone.

Forward-only: the dropped table's contents are unrecoverable, so ``downgrade()``
raises. To roll back, restore from a pre-migration DB snapshot.

Rationale for bypassing a formal operational metrics gate: this repository is
solo open-source development with no live production deployment to observe; CI +
the cross-PR codex xhigh review loop are the substitute correctness gates
(mirrors c3pr6_retire_legacy_ctrl_plane.py).
"""

from __future__ import annotations

from alembic import op
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision = "pe4d2_drop_tool_approval_rules"
down_revision = "c2pr7_envelope_store"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    # Count legacy rules with no corresponding grant. The join key MIRRORS the
    # backfill CLI's _SELECT_LEGACY_SQL (backfill_approval_grants.py:110-118):
    #   g.user_id    = r.user_id
    #   g.tool_name  = r.tool_name
    #   g.primary_arg = r.command_pattern
    #   g.dir_arg    = COALESCE(NULLIF(r.dir_pattern, ''), '')
    #   g.scope      = 'always'
    unmigrated = conn.execute(
        text(
            "SELECT COUNT(*) FROM tool_approval_rules r "
            "WHERE NOT EXISTS ("
            "SELECT 1 FROM tool_approval_grants g "
            "WHERE g.user_id = r.user_id "
            "AND g.tool_name = r.tool_name "
            "AND g.primary_arg = r.command_pattern "
            "AND g.dir_arg = COALESCE(NULLIF(r.dir_pattern, ''), '') "
            "AND g.scope = 'always'"
            ")"
        )
    ).scalar_one()
    if unmigrated:
        raise RuntimeError(
            f"PE-4d2 abort: {unmigrated} un-migrated tool_approval_rules row(s) "
            "have no corresponding tool_approval_grants entry. Run "
            "`uv run python -m app.cli.backfill_approval_grants` first, then "
            "re-run the migration. (This migration is forward-only and would "
            "otherwise permanently lose those rows.)"
        )
    op.drop_table("tool_approval_rules")


def downgrade() -> None:
    raise NotImplementedError(
        "PE-4d2 (pe4d2_drop_tool_approval_rules) is forward-only — the dropped "
        "tool_approval_rules table contents are unrecoverable. To roll back, "
        "restore from a pre-migration DB snapshot."
    )
