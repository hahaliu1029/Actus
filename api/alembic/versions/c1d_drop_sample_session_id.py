"""c1d_drop_sample_session_id

Revision ID: c1d_drop_sample_session_id
Revises: c1a_session_tree_expand
Create Date: 2026-05-19

C1a expand-contract step 2 (PR-4). Removes the legacy `sample_session_id`
surface now that all pods read `parent_session_id` exclusively.

Drops, in order:
  - mirror trigger + mirror function (no longer needed)
  - the old t12 CHECK `ck_sessions_child_must_have_preset` referencing sample_session_id
  - the partial index `ix_sessions_sample_session_id`
  - the FK `fk_sessions_sample_session_id_sessions`
  - the column `sample_session_id`

Recreates CHECK `ck_sessions_child_must_have_preset` referencing `parent_session_id`
(same semantics: child sessions must have a tool_filter_preset).
"""
from __future__ import annotations

from alembic import op


revision = "c1d_drop_sample_session_id"
down_revision = "c1a_session_tree_expand"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_sessions_mirror_sample ON sessions")
    op.execute("DROP FUNCTION IF EXISTS mirror_sample_to_parent()")
    op.execute(
        "ALTER TABLE sessions DROP CONSTRAINT IF EXISTS ck_sessions_child_must_have_preset"
    )
    op.execute(
        "ALTER TABLE sessions ADD CONSTRAINT ck_sessions_child_must_have_preset "
        "CHECK (parent_session_id IS NULL OR tool_filter_preset IS NOT NULL)"
    )
    # The c1a parent-user invariant function + constraint trigger reference
    # NEW.sample_session_id (function body) and UPDATE OF sample_session_id
    # (trigger column list). Swap both to parent-only form BEFORE dropping
    # the column so the trigger body doesn't blow up at runtime and so the
    # column-drop isn't blocked by the trigger's column dependency.
    op.execute("DROP TRIGGER IF EXISTS trg_sessions_parent_user_match ON sessions")
    op.execute("DROP FUNCTION IF EXISTS ensure_parent_same_user()")
    op.execute(
        """
        CREATE OR REPLACE FUNCTION ensure_parent_same_user() RETURNS trigger AS $$
        DECLARE
          parent_user VARCHAR(255);
        BEGIN
          IF NEW.parent_session_id IS NOT NULL THEN
            SELECT user_id INTO parent_user FROM sessions WHERE id = NEW.parent_session_id;
            IF NEW.user_id IS NOT NULL THEN
              IF parent_user IS DISTINCT FROM NEW.user_id THEN
                RAISE EXCEPTION 'sessions parent must belong to same user (child=%, parent=%, parent_user=%)',
                  NEW.id, NEW.parent_session_id, parent_user
                  USING ERRCODE = '23514';
              END IF;
            ELSE
              IF parent_user IS NOT NULL THEN
                RAISE EXCEPTION 'orphaned child cannot reference still-owned parent (child=%, parent=%)',
                  NEW.id, NEW.parent_session_id
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
          AFTER INSERT OR UPDATE OF parent_session_id, user_id ON sessions
          DEFERRABLE INITIALLY DEFERRED
          FOR EACH ROW EXECUTE FUNCTION ensure_parent_same_user();
        """
    )
    op.execute("DROP INDEX IF EXISTS ix_sessions_sample_session_id")
    op.drop_constraint(
        "fk_sessions_sample_session_id_sessions",
        "sessions",
        type_="foreignkey",
    )
    op.drop_column("sessions", "sample_session_id")


def downgrade() -> None:
    import sqlalchemy as sa
    op.add_column(
        "sessions",
        sa.Column("sample_session_id", sa.String(length=255), nullable=True),
    )
    op.execute(
        "UPDATE sessions SET sample_session_id = parent_session_id "
        "WHERE parent_session_id IS NOT NULL"
    )
    op.create_foreign_key(
        "fk_sessions_sample_session_id_sessions",
        "sessions",
        "sessions",
        ["sample_session_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_index(
        "ix_sessions_sample_session_id",
        "sessions",
        ["sample_session_id"],
        postgresql_where=sa.text("sample_session_id IS NOT NULL"),
    )
    op.execute(
        "ALTER TABLE sessions DROP CONSTRAINT IF EXISTS ck_sessions_child_must_have_preset"
    )
    op.execute(
        "ALTER TABLE sessions ADD CONSTRAINT ck_sessions_child_must_have_preset "
        "CHECK (sample_session_id IS NULL OR tool_filter_preset IS NOT NULL)"
    )
    # Restore the c1a dual-column parent-user function + constraint trigger so
    # a downgrade lands at byte-identical c1a head state. SQL bodies mirror
    # api/alembic/versions/c1a_session_tree_expand.py:151-185.
    op.execute("DROP TRIGGER IF EXISTS trg_sessions_parent_user_match ON sessions")
    op.execute("DROP FUNCTION IF EXISTS ensure_parent_same_user()")
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
    # Restore the mirror function + trigger so a downgrade to c1a head leaves
    # the schema in the same shape PR-1's expand migration produced. SQL bodies
    # mirror api/alembic/versions/c1a_session_tree_expand.py:71-105 and 190-193.
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
    op.execute(
        """
        CREATE TRIGGER trg_sessions_mirror_sample
          BEFORE INSERT OR UPDATE OF sample_session_id, parent_session_id ON sessions
          FOR EACH ROW EXECUTE FUNCTION mirror_sample_to_parent();
        """
    )
