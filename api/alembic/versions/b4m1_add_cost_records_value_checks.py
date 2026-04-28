"""B4 M1: lock cost_records token + total_usd to non-negative.

Revision ID: b4m1_cost_records_value_checks
Revises: b4m0_cost_records
Create Date: 2026-04-28 10:00:00.000000

Adds 6 CHECK constraints (one per token column + total_usd) so the ledger
rejects negative inputs at INSERT time. Without these, a buggy provider
adapter or hand-written SQL fixture could pollute aggregation: a single
``input_tokens=-1`` row would skew GET /cost into nonsense (or worse,
silently negate other rows).

Statement shape: ``ADD CONSTRAINT ... NOT VALID`` (6×) then
``VALIDATE CONSTRAINT`` (6×) — 12 separate ``op.execute`` calls.

Lock note (be precise — the original docstring overclaimed and codex
caught it): alembic wraps each migration in a single transaction
(env.py:80-83 ``with context.begin_transaction(): context.run_migrations()``),
and PG holds locks acquired via DDL until end-of-transaction. So inside
this revision the ``ACCESS EXCLUSIVE`` taken by each ``ADD CONSTRAINT``
is held throughout — including across the subsequent VALIDATE pass.
Splitting NOT VALID and VALIDATE within the same txn does NOT yield the
"online migration" lock-minimization benefit you'd get if the two ran
in separate transactions; it only buys a clean all-or-nothing revision
boundary for atomicity / clean rollback.

For the small B4 ledger (just shipped, low row count, single API
replica) that's fine — atomicity is the actual property worth keeping,
and a multi-second writer pause during VALIDATE is acceptable. If a
future enlargement of this table (or a similar table) needs true online
migration semantics, lift NOT VALID and VALIDATE into separate
revisions so each runs in its own transaction.

Raw SQL (not ``op.create_check_constraint``) on purpose:
- ``op.create_check_constraint("input_tokens_nonneg", ...)`` would route
  through SQLAlchemy's naming_convention (base.py:5-11 — ``ck`` template
  is ``ck_%(table_name)s_%(constraint_name)s``), producing
  ``ck_cost_records_ck_cost_records_input_tokens_nonneg`` (double prefix)
  — which is what happened to ``cost_status_allowed`` in b4m0.
- Raw SQL with the literal name skips the naming_convention pass entirely,
  giving us the clean canonical name we want to assert in the integration
  test.
- VALIDATE CONSTRAINT has no first-class alembic helper anyway, so any
  approach mixing ``create_check_constraint`` + raw VALIDATE would be
  inconsistent on its own.

Recovery: if a deploy ships into a database that already contains
negative-value rows, VALIDATE raises and the entire migration
transaction rolls back — alembic_version does not advance and no
constraint additions persist. Recovery is a single step: clean the
offending rows, then re-run ``alembic upgrade head``. (PG has no
``ADD CONSTRAINT IF NOT EXISTS`` syntax, so re-running only succeeds
because the prior failed run left zero residue, not because the ADDs
are individually idempotent.)
"""

from typing import Sequence, Union

from alembic import op

revision: str = "b4m1_cost_records_value_checks"
down_revision: Union[str, Sequence[str], None] = "b4m0_cost_records"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# (column_name, constraint_name) — explicit names so the integration test
# can pin them via pg_constraint reflection.
_NONNEG_COLUMNS: tuple[tuple[str, str], ...] = (
    ("input_tokens", "ck_cost_records_input_tokens_nonneg"),
    ("output_tokens", "ck_cost_records_output_tokens_nonneg"),
    ("cache_read_tokens", "ck_cost_records_cache_read_tokens_nonneg"),
    ("cache_write_tokens", "ck_cost_records_cache_write_tokens_nonneg"),
    ("reasoning_tokens", "ck_cost_records_reasoning_tokens_nonneg"),
    ("total_usd", "ck_cost_records_total_usd_nonneg"),
)


def upgrade() -> None:
    for column, name in _NONNEG_COLUMNS:
        op.execute(
            f"ALTER TABLE cost_records "
            f"ADD CONSTRAINT {name} CHECK ({column} >= 0) NOT VALID"
        )
    for _, name in _NONNEG_COLUMNS:
        op.execute(f"ALTER TABLE cost_records VALIDATE CONSTRAINT {name}")


def downgrade() -> None:
    for _, name in _NONNEG_COLUMNS:
        op.execute(f"ALTER TABLE cost_records DROP CONSTRAINT {name}")
