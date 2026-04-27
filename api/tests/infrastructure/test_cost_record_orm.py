"""B4 M0 Phase C: cost_records ORM table + CHECK constraint (Issue 2C).

Unit tests that introspect SQLAlchemy metadata — no DB connection required.
The integration migration test (postgres-only) lives separately and verifies
that an invalid ``cost_status`` is rejected at INSERT time.
"""

from __future__ import annotations

from sqlalchemy import CheckConstraint

from app.infrastructure.models import Base
from app.infrastructure.models.cost_record_orm import CostRecordModel


class TestCostRecordORM:
    def test_table_name(self) -> None:
        assert CostRecordModel.__tablename__ == "cost_records"

    def test_cost_status_check_constraint_present_with_four_states(self) -> None:
        """Issue 2C: CHECK rejects invalid cost_status strings at INSERT time."""
        cost_status_checks = [
            c
            for c in CostRecordModel.__table__.constraints
            if isinstance(c, CheckConstraint)
            and c.name
            and "cost_status" in c.name
        ]
        assert cost_status_checks, (
            "cost_records must declare a CHECK constraint on cost_status "
            "(design Issue 2C). No such constraint found in __table_args__."
        )
        sqltext = str(cost_status_checks[0].sqltext).lower()
        for state in ("actual", "estimated", "partial", "unknown"):
            assert state in sqltext, (
                f"CHECK sqltext must whitelist {state!r}; got: {sqltext!r}"
            )

    def test_required_columns_present(self) -> None:
        cols = {c.name for c in CostRecordModel.__table__.columns}
        required = {
            "id",
            "session_id",
            "user_id",
            "run_id",
            "node_name",
            "step_ix",
            "attempt_ix",
            "model",
            "provider",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "total_usd",
            "pricing_version",
            "cost_status",
            "created_at",
        }
        missing = required - cols
        assert not missing, f"cost_records missing columns: {missing}"

    def test_session_id_has_cascade_delete_fk(self) -> None:
        col = CostRecordModel.__table__.columns["session_id"]
        fks = list(col.foreign_keys)
        assert fks, "session_id must be a foreign key"
        assert fks[0].ondelete == "CASCADE"

    def test_user_id_has_cascade_delete_fk(self) -> None:
        col = CostRecordModel.__table__.columns["user_id"]
        fks = list(col.foreign_keys)
        assert fks, "user_id must be a foreign key"
        assert fks[0].ondelete == "CASCADE"

    def test_run_id_has_unique_index_for_idempotency(self) -> None:
        """Handler dedup is defense-in-depth; DB-level unique on run_id is authoritative."""
        indexes = CostRecordModel.__table__.indexes
        run_id_indexes = [
            ix for ix in indexes if "run_id" in {c.name for c in ix.columns}
        ]
        assert any(ix.unique for ix in run_id_indexes), (
            "run_id must be covered by a UNIQUE index so the callback's "
            "retry path cannot double-bill a single LLM call."
        )

    def test_created_at_is_tz_aware_column(self) -> None:
        col = CostRecordModel.__table__.columns["created_at"]
        assert col.type.timezone is True, (
            "cost_records.created_at must be TIMESTAMPTZ (Issue OV-3). "
            "Naive timestamps break tuple-sort aggregation across DB restarts."
        )

    def test_metadata_fk_targets_resolve(self) -> None:
        """``Base.metadata.sorted_tables`` must not raise NoReferencedTableError.

        cost_records has FKs to both ``sessions`` and ``users``. If any FK target
        table isn't imported at ``app.infrastructure.models`` level, alembic
        autogenerate / ``Base.metadata.create_all`` crash with a cryptic error.
        """
        tables = [t.name for t in Base.metadata.sorted_tables]
        assert "cost_records" in tables
        assert "users" in tables, (
            "users table must be registered in Base.metadata (import UserModel "
            "in app/infrastructure/models/__init__.py) — otherwise cost_records' "
            "user_id FK cannot resolve."
        )
        assert "sessions" in tables
