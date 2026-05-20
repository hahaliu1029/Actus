"""c1a_session_tree_expand

Revision ID: c1a_session_tree_expand
Revises: t12_tool_filter_preset
Create Date: 2026-05-19

C1a expand-contract step 1 (PR-1). Adds:
  - sessions.parent_session_id (nullable VARCHAR(255), FK -> sessions.id ON DELETE RESTRICT)
  - sessions.worker_type (NOT NULL VARCHAR(16) DEFAULT 'root', CHECK (root, subagent))
  - mirror trigger `trg_sessions_mirror_sample` keeps `sample_session_id` and
    `parent_session_id` in sync during the dual-write window.
  - DEFERRABLE CONSTRAINT TRIGGER `trg_sessions_parent_user_match` enforces
    parent.user_id == child.user_id at COMMIT time.

The legacy column `sample_session_id` is RETAINED here for backward compatibility
with old pods during rolling deploy. PR-4 (c1d_drop_sample_session_id) drops it.

Raw SQL CHECK / FK names because `Base.metadata.naming_convention` (base.py:5-11)
would double-prefix `op.create_check_constraint("ck_sessions_X", ...)` to
`ck_sessions_ck_sessions_X` (per T12 / b4m1 precedent).
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "c1a_session_tree_expand"
down_revision = "t12_tool_filter_preset"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---- Step 1: ADD parent_session_id ----
    op.add_column(
        "sessions",
        sa.Column("parent_session_id", sa.String(length=255), nullable=True),
    )

    # ---- Step 2: backfill from existing sample_session_id ----
    op.execute(
        "UPDATE sessions "
        "   SET parent_session_id = sample_session_id "
        " WHERE sample_session_id IS NOT NULL"
    )

    # ---- Step 3: FK + partial index on the new column (old FK/index kept) ----
    op.create_foreign_key(
        "fk_sessions_parent_session_id_sessions",
        "sessions",
        "sessions",
        ["parent_session_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_index(
        "ix_sessions_parent_session_id",
        "sessions",
        ["parent_session_id"],
        postgresql_where=sa.text("parent_session_id IS NOT NULL"),
    )

    # ---- Step 3.5: mirror_sample_to_parent() function definition ----
    # Old pod INSERT only writes sample_session_id (no worker_type); mirror trigger:
    #   1) propagates value between sample_session_id and parent_session_id (NULL-aware)
    #   2) auto-derives worker_type from parent_session_id non-NULL
    #   3) raises on explicit two-column conflict
    # NOTE: trigger is created LAST (after worker_type column exists). See task 1.5 step 3.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION mirror_sample_to_parent() RETURNS trigger AS $$
        BEGIN
          IF NEW.sample_session_id IS NOT NULL
             AND NEW.parent_session_id IS NOT NULL
             AND NEW.sample_session_id <> NEW.parent_session_id THEN
            RAISE EXCEPTION 'sample_session_id and parent_session_id must be equal during expand-contract window'
              USING ERRCODE = '23514';
          END IF;
          IF TG_OP = 'INSERT' THEN
            IF NEW.sample_session_id IS NULL AND NEW.parent_session_id IS NOT NULL THEN
              NEW.sample_session_id := NEW.parent_session_id;
            ELSIF NEW.parent_session_id IS NULL AND NEW.sample_session_id IS NOT NULL THEN
              NEW.parent_session_id := NEW.sample_session_id;
            END IF;
          ELSIF TG_OP = 'UPDATE' THEN
            IF NEW.sample_session_id IS NULL AND OLD.sample_session_id IS NOT NULL
               AND NEW.parent_session_id IS NOT DISTINCT FROM OLD.parent_session_id THEN
              NEW.parent_session_id := NULL;
              NEW.worker_type := 'root';
            ELSIF NEW.parent_session_id IS NULL AND OLD.parent_session_id IS NOT NULL
               AND NEW.sample_session_id IS NOT DISTINCT FROM OLD.sample_session_id THEN
              NEW.sample_session_id := NULL;
              NEW.worker_type := 'root';
            ELSIF NEW.sample_session_id IS NULL AND NEW.parent_session_id IS NOT NULL THEN
              NEW.sample_session_id := NEW.parent_session_id;
            ELSIF NEW.parent_session_id IS NULL AND NEW.sample_session_id IS NOT NULL THEN
              NEW.parent_session_id := NEW.sample_session_id;
            END IF;
          END IF;
          IF NEW.parent_session_id IS NOT NULL AND NEW.worker_type = 'root' THEN
            NEW.worker_type := 'subagent';
          END IF;
          RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )

    # ---- Step 4: ADD worker_type with DEFAULT 'root' ----
    op.add_column(
        "sessions",
        sa.Column(
            "worker_type",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'root'::character varying"),
        ),
    )

    # ---- Step 5: backfill subagent rows (existing children with non-NULL sample_session_id) ----
    op.execute(
        "UPDATE sessions "
        "   SET worker_type = 'subagent' "
        " WHERE parent_session_id IS NOT NULL"
    )

    # ---- Step 6: raw-SQL CHECK constraints (avoid Base naming_convention double-prefix) ----
    op.execute(
        "ALTER TABLE sessions ADD CONSTRAINT ck_sessions_worker_type "
        "CHECK (worker_type IN ('root', 'subagent'))"
    )
    op.execute(
        "ALTER TABLE sessions ADD CONSTRAINT ck_sessions_worker_type_parent_invariant "
        "CHECK ("
        "  (parent_session_id IS NULL AND worker_type = 'root') OR "
        "  (parent_session_id IS NOT NULL AND worker_type = 'subagent')"
        ")"
    )
    op.execute(
        "ALTER TABLE sessions ADD CONSTRAINT ck_sessions_no_self_parent "
        "CHECK (parent_session_id IS NULL OR parent_session_id <> id)"
    )

    # ---- Step 7: cross-row parent.user_id == child.user_id invariant ----
    # Uses CONSTRAINT TRIGGER DEFERRABLE INITIALLY DEFERRED so cascade SET NULL
    # (sessions.user_id ON DELETE SET NULL - see 3a5b7c9d1e2f migration) does not
    # raise mid-statement during admin user delete.
    # parent_user typed VARCHAR(255) to match sessions.user_id String(255).
    op.execute(
        """
        CREATE OR REPLACE FUNCTION ensure_parent_same_user() RETURNS trigger AS $$
        DECLARE
          effective_parent_id TEXT;
          parent_user VARCHAR(255);
        BEGIN
          effective_parent_id := COALESCE(NEW.parent_session_id, NEW.sample_session_id);
          IF effective_parent_id IS NOT NULL THEN
            SELECT user_id INTO parent_user FROM sessions WHERE id = effective_parent_id;
            IF NEW.user_id IS NOT NULL THEN
              IF parent_user IS DISTINCT FROM NEW.user_id THEN
                RAISE EXCEPTION 'sessions parent must belong to same user (child=%, parent=%, parent_user=%)',
                  NEW.id, effective_parent_id, parent_user
                  USING ERRCODE = '23514';
              END IF;
            ELSE
              IF parent_user IS NOT NULL THEN
                RAISE EXCEPTION 'orphaned child cannot reference still-owned parent (child=%, parent=%)',
                  NEW.id, effective_parent_id
                  USING ERRCODE = '23514';
              END IF;
            END IF;
          END IF;
          RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE CONSTRAINT TRIGGER trg_sessions_parent_user_match
          AFTER INSERT OR UPDATE OF parent_session_id, sample_session_id, user_id ON sessions
          DEFERRABLE INITIALLY DEFERRED
          FOR EACH ROW EXECUTE FUNCTION ensure_parent_same_user();
        """
    )

    # ---- Step 8: mirror trigger (fires LAST so column + CHECKs are in place) ----
    op.execute(
        """
        CREATE TRIGGER trg_sessions_mirror_sample
          BEFORE INSERT OR UPDATE OF sample_session_id, parent_session_id ON sessions
          FOR EACH ROW EXECUTE FUNCTION mirror_sample_to_parent();
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_sessions_parent_user_match ON sessions")
    op.execute("DROP FUNCTION IF EXISTS ensure_parent_same_user()")
    op.execute("DROP TRIGGER IF EXISTS trg_sessions_mirror_sample ON sessions")
    op.execute("DROP FUNCTION IF EXISTS mirror_sample_to_parent()")
    op.execute("ALTER TABLE sessions DROP CONSTRAINT IF EXISTS ck_sessions_no_self_parent")
    op.execute("ALTER TABLE sessions DROP CONSTRAINT IF EXISTS ck_sessions_worker_type_parent_invariant")
    op.execute("ALTER TABLE sessions DROP CONSTRAINT IF EXISTS ck_sessions_worker_type")
    op.drop_column("sessions", "worker_type")
    op.drop_index("ix_sessions_parent_session_id", table_name="sessions")
    op.drop_constraint(
        "fk_sessions_parent_session_id_sessions",
        "sessions",
        type_="foreignkey",
    )
    op.drop_column("sessions", "parent_session_id")
