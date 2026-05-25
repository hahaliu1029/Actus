"""C3 PR-6 — retire legacy subagent control plane

Revision ID: c3pr6_retire_legacy_control_plane
Revises: c3_add_mailbox_envelope_audit
Create Date: 2026-05-24

Per C3 spec §11.7 — the legacy subagent suspend path is retired. This
migration upgrades any existing subagent rows whose
``subagent_control_plane`` is either ``'legacy'`` OR ``NULL`` to
``'mailbox'`` so the M1 single-writer invariant (MailboxSupervisor is
the sole writer of subagent terminal sandbox transitions) holds for
every live subagent row.

Why both ``'legacy'`` AND ``NULL``: the C3 PR-1 baseline migration
(``c3_add_mailbox_envelope_audit``) declared the column as NULLABLE
with no default, and the supervisor's rollback-stop check explicitly
coalesces ``(subagent_control_plane or 'legacy') == 'legacy'`` (see
``mailbox_supervisor.py`` orphan + rollback paths). NULL is therefore a
canonical synonym for legacy in operational semantics; any pre-C3
historic row that was never backfilled lives as NULL. PR-6 collapses
both into ``'mailbox'`` so the post-PR-6 agent_service ``elif
session.worker_type == 'root'`` guard does NOT orphan their sandboxes.

Scope:
  - Touches ``worker_type='subagent'`` rows only.
  - Root rows (``worker_type='root'``) always carry NULL plane in
    practice (``SessionService.create_session()`` never sets it), and
    the actual CHECK constraint ``ck_sessions_subagent_control_plane_valid``
    permits ``NULL | 'legacy' | 'mailbox'`` for all rows so no DB-level
    invariant is violated by leaving root NULLs untouched.
  - Subagent rows already at ``'mailbox'`` are no-ops.

Forward-only: the original ``'legacy'`` vs ``'mailbox'`` vs ``NULL``
distinction is lost after the upgrade, so ``downgrade()`` raises. To
roll back, restore from a pre-migration DB snapshot.

Rationale for bypassing the §11.7 4-week operational metrics gate: this
repository is solo open-source development with no live production
deployment to observe; CI + the cross-PR codex xhigh review loop are
the substitute correctness gates.
"""

from __future__ import annotations

from alembic import op


# revision identifiers, used by Alembic.
revision = "c3pr6_retire_legacy_control_plane"
down_revision = "c3_add_mailbox_envelope_audit"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE sessions
           SET subagent_control_plane = 'mailbox'
         WHERE worker_type = 'subagent'
           AND (subagent_control_plane = 'legacy'
                OR subagent_control_plane IS NULL);
        """
    )


def downgrade() -> None:
    raise NotImplementedError(
        "C3 PR-6 legacy retirement (c3pr6_retire_legacy_control_plane) "
        "is forward-only — the original 'legacy' / NULL / 'mailbox' "
        "distinction is unrecoverable. To roll back, restore from a "
        "pre-migration DB snapshot."
    )
