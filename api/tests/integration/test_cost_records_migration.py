"""B4 M0 Phase J: cost_records migration integration tests.

Locks the migration's DB-level schema contract + repository / aggregator
consumer paths against a real migrated Postgres. ORM-level CHECK / unique /
FK behavior is unit-tested at the SQLAlchemy layer; this file exists because
those don't exercise alembic upgrade head, FK CASCADE semantics under real
PG, ON CONFLICT DO NOTHING idempotency, or the degraded-marker aggregator
override path.

Codex 4-round adversarial review converged 2026-04-28 (LOCKABLE).

Run: cd api && uv run pytest tests/integration/test_cost_records_migration.py -v
Requires: pgvector-enabled Postgres reachable via SQLALCHEMY_DATABASE_URL.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.application.services.cost_aggregation_service import (
    CostAggregationService,
)
from app.domain.models.cost_record import CostRecord, CostStatus
from app.infrastructure.repositories.db_cost_record_repository import (
    DbCostRecordRepository,
)

pytestmark = pytest.mark.anyio


async def _ensure_user(db_session, user_id: str) -> None:
    await db_session.execute(
        text("INSERT INTO users (id) VALUES (:uid) ON CONFLICT DO NOTHING"),
        {"uid": user_id},
    )


async def _ensure_session(db_session, session_id: str) -> None:
    await db_session.execute(
        text("INSERT INTO sessions (id) VALUES (:sid) ON CONFLICT DO NOTHING"),
        {"sid": session_id},
    )


_INSERT_COLUMNS = (
    "id, session_id, user_id, run_id, node_name, step_ix, attempt_ix, "
    "model, provider, input_tokens, output_tokens, cache_read_tokens, "
    "cache_write_tokens, reasoning_tokens, total_usd, pricing_version, "
    "cost_status, created_at"
)


async def _raw_insert_full(db_session, **overrides) -> dict:
    """INSERT a fully-specified cost_records row via raw SQL; return values dict."""
    values = {
        "id": str(uuid.uuid4()),
        "session_id": str(uuid.uuid4()),
        "user_id": str(uuid.uuid4()),
        "run_id": str(uuid.uuid4()),
        "node_name": "test_node",
        "step_ix": 0,
        "attempt_ix": 0,
        "model": "test-model",
        "provider": "test-provider",
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "total_usd": Decimal("0"),
        "pricing_version": "test_v1",
        "cost_status": "actual",
        "created_at": datetime.now(timezone.utc),
    }
    values.update(overrides)
    placeholders = ", ".join(f":{k}" for k in values)
    await db_session.execute(
        text(
            f"INSERT INTO cost_records ({_INSERT_COLUMNS}) VALUES ({placeholders})"
        ),
        values,
    )
    return values


class TestSchemaReflection:
    async def test_table_exists(self, db_session):
        result = await db_session.execute(
            text(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_name = 'cost_records'"
            )
        )
        assert result.scalar() == 1

    async def test_columns_match_orm_shape(self, db_session):
        result = await db_session.execute(
            text(
                "SELECT column_name, data_type, is_nullable, "
                "  character_maximum_length, numeric_precision, numeric_scale, "
                "  column_default "
                "FROM information_schema.columns "
                "WHERE table_name = 'cost_records'"
            )
        )
        cols = {r.column_name: r for r in result.all()}

        expected = {
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
        assert set(cols.keys()) == expected

        varchar_lengths = {
            "id": 64,
            "session_id": 255,
            "user_id": 255,
            "run_id": 64,
            "node_name": 128,
            "model": 128,
            "provider": 64,
            "pricing_version": 32,
            "cost_status": 16,
        }
        for col_name, expected_len in varchar_lengths.items():
            col = cols[col_name]
            assert col.data_type == "character varying", (
                f"{col_name}: expected character varying, got {col.data_type}"
            )
            assert col.character_maximum_length == expected_len, (
                f"{col_name}: expected length {expected_len}, "
                f"got {col.character_maximum_length}"
            )

        total_usd = cols["total_usd"]
        assert total_usd.data_type == "numeric"
        assert total_usd.numeric_precision == 28
        assert total_usd.numeric_scale == 10

        for int_col in (
            "step_ix",
            "attempt_ix",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
        ):
            assert cols[int_col].data_type == "integer", (
                f"{int_col}: expected integer, got {cols[int_col].data_type}"
            )

        for col_name in expected:
            assert cols[col_name].is_nullable == "NO", (
                f"{col_name}: expected NOT NULL, got {cols[col_name].is_nullable}"
            )

        cols_with_default = {
            "step_ix",
            "attempt_ix",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "total_usd",
            "created_at",
        }
        for col_name in cols_with_default:
            assert cols[col_name].column_default is not None, (
                f"{col_name}: expected server_default, got None"
            )
        for col_name in expected - cols_with_default:
            assert cols[col_name].column_default is None, (
                f"{col_name}: expected no server_default, "
                f"got {cols[col_name].column_default}"
            )

    async def test_primary_key_columns_are_id_only(self, db_session):
        result = await db_session.execute(
            text(
                "SELECT a.attname "
                "FROM pg_constraint c "
                "JOIN pg_class t ON t.oid = c.conrelid "
                "JOIN pg_attribute a ON a.attrelid = t.oid "
                "  AND a.attnum = ANY(c.conkey) "
                "WHERE t.relname = 'cost_records' AND c.contype = 'p' "
                "ORDER BY a.attnum"
            )
        )
        assert [r.attname for r in result.all()] == ["id"]

    async def test_indexes_exist(self, db_session):
        # Names alone don't lock the schema shape — assert each index covers
        # the right column AND the unique index is actually UNIQUE.
        result = await db_session.execute(
            text(
                "SELECT indexname, indexdef FROM pg_indexes "
                "WHERE tablename = 'cost_records'"
            )
        )
        defs_by_name = {r.indexname: r.indexdef for r in result.all()}

        ix_session = defs_by_name.get("ix_cost_records_session_id")
        assert ix_session is not None, "missing ix_cost_records_session_id"
        assert "(session_id)" in ix_session, (
            f"ix_session_id covers wrong column: {ix_session!r}"
        )
        assert not ix_session.upper().startswith("CREATE UNIQUE"), (
            f"ix_session_id should NOT be unique: {ix_session!r}"
        )

        ix_user = defs_by_name.get("ix_cost_records_user_id")
        assert ix_user is not None, "missing ix_cost_records_user_id"
        assert "(user_id)" in ix_user, (
            f"ix_user_id covers wrong column: {ix_user!r}"
        )
        assert not ix_user.upper().startswith("CREATE UNIQUE"), (
            f"ix_user_id should NOT be unique: {ix_user!r}"
        )

        uq_run = defs_by_name.get("uq_cost_records_run_id")
        assert uq_run is not None, "missing uq_cost_records_run_id"
        assert "(run_id)" in uq_run, (
            f"uq_run_id covers wrong column: {uq_run!r}"
        )
        assert uq_run.upper().startswith("CREATE UNIQUE INDEX"), (
            f"uq_run_id must be UNIQUE: {uq_run!r}"
        )

    async def test_check_constraint_exists(self, db_session):
        # Alembic naming_convention auto-prefixes 'ck_<table>_' so the actual
        # name is 'ck_cost_records_ck_cost_records_cost_status_allowed'. Lock
        # by extracting literal strings from the CHECK definition and
        # asserting set equality — this rejects drift in BOTH directions
        # (missing valid status OR sneaky extra status added later).
        result = await db_session.execute(
            text(
                "SELECT pg_get_constraintdef(oid) AS definition "
                "FROM pg_constraint "
                "WHERE conrelid = 'cost_records'::regclass "
                "AND contype = 'c'"
            )
        )
        defs = [row.definition for row in result.all()]
        cost_status_defs = [d for d in defs if "cost_status" in d]
        assert cost_status_defs, (
            f"no CHECK constraint references cost_status; got: {defs!r}"
        )
        # Extract single-quoted literals from the constraint def(s).
        literals: set[str] = set()
        for d in cost_status_defs:
            literals.update(re.findall(r"'([^']*)'", d))
        assert literals == {"actual", "estimated", "partial", "unknown"}, (
            f"expected exactly {{actual, estimated, partial, unknown}}, "
            f"got {literals!r} in defs {cost_status_defs!r}"
        )

    async def test_created_at_is_timestamptz(self, db_session):
        result = await db_session.execute(
            text(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_name = 'cost_records' AND column_name = 'created_at'"
            )
        )
        assert result.scalar() == "timestamp with time zone"


class TestConstraintEnforcement:
    async def test_check_constraint_rejects_invalid_status(self, db_session):
        user_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        await _ensure_user(db_session, user_id)
        await _ensure_session(db_session, session_id)
        with pytest.raises(IntegrityError):
            await _raw_insert_full(
                db_session,
                user_id=user_id,
                session_id=session_id,
                cost_status="bogus",
            )

    async def test_unique_run_id_blocks_raw_duplicate(self, db_session):
        user_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        await _ensure_user(db_session, user_id)
        await _ensure_session(db_session, session_id)
        await _raw_insert_full(
            db_session, user_id=user_id, session_id=session_id, run_id=run_id
        )
        with pytest.raises(IntegrityError):
            await _raw_insert_full(
                db_session,
                user_id=user_id,
                session_id=session_id,
                run_id=run_id,
            )

    async def test_orphan_session_id_rejected(self, db_session):
        user_id = str(uuid.uuid4())
        await _ensure_user(db_session, user_id)
        with pytest.raises(IntegrityError):
            await _raw_insert_full(
                db_session,
                user_id=user_id,
                session_id="00000000-0000-0000-0000-000000000000",
            )

    async def test_orphan_user_id_rejected(self, db_session):
        session_id = str(uuid.uuid4())
        await _ensure_session(db_session, session_id)
        with pytest.raises(IntegrityError):
            await _raw_insert_full(
                db_session,
                user_id="00000000-0000-0000-0000-000000000000",
                session_id=session_id,
            )


class TestHappyPath:
    @pytest.mark.parametrize(
        "status", ["actual", "estimated", "partial", "unknown"]
    )
    async def test_check_constraint_accepts_all_4_valid_statuses(
        self, db_session, status
    ):
        user_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        await _ensure_user(db_session, user_id)
        await _ensure_session(db_session, session_id)
        await _raw_insert_full(
            db_session,
            user_id=user_id,
            session_id=session_id,
            cost_status=status,
        )

    async def test_session_cascade_deletes_cost_records(self, db_session):
        user_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        await _ensure_user(db_session, user_id)
        await _ensure_session(db_session, session_id)
        values = await _raw_insert_full(
            db_session, user_id=user_id, session_id=session_id
        )
        await db_session.execute(
            text("DELETE FROM sessions WHERE id = :sid"), {"sid": session_id}
        )
        result = await db_session.execute(
            text("SELECT COUNT(*) FROM cost_records WHERE id = :id"),
            {"id": values["id"]},
        )
        assert result.scalar() == 0

    async def test_user_cascade_deletes_cost_records(self, db_session):
        user_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        await _ensure_user(db_session, user_id)
        await _ensure_session(db_session, session_id)
        values = await _raw_insert_full(
            db_session, user_id=user_id, session_id=session_id
        )
        await db_session.execute(
            text("DELETE FROM users WHERE id = :uid"), {"uid": user_id}
        )
        result = await db_session.execute(
            text("SELECT COUNT(*) FROM cost_records WHERE id = :id"),
            {"id": values["id"]},
        )
        assert result.scalar() == 0

    async def test_numeric_28_10_round_trip(self, db_session):
        user_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        await _ensure_user(db_session, user_id)
        await _ensure_session(db_session, session_id)
        precise = Decimal("0.0000001234")
        values = await _raw_insert_full(
            db_session,
            user_id=user_id,
            session_id=session_id,
            total_usd=precise,
        )
        result = await db_session.execute(
            text("SELECT total_usd FROM cost_records WHERE id = :id"),
            {"id": values["id"]},
        )
        round_tripped = result.scalar()
        assert round_tripped == precise, (
            f"expected {precise}, got {round_tripped}"
        )

    async def test_timestamptz_aware_round_trip(self, db_session):
        user_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        await _ensure_user(db_session, user_id)
        await _ensure_session(db_session, session_id)
        aware = datetime.now(timezone.utc)
        values = await _raw_insert_full(
            db_session,
            user_id=user_id,
            session_id=session_id,
            created_at=aware,
        )
        result = await db_session.execute(
            text("SELECT created_at FROM cost_records WHERE id = :id"),
            {"id": values["id"]},
        )
        round_tripped = result.scalar()
        assert round_tripped.tzinfo is not None, (
            f"expected tz-aware, got {round_tripped!r}"
        )

    async def test_server_defaults_apply_when_columns_omitted(self, db_session):
        user_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        await _ensure_user(db_session, user_id)
        await _ensure_session(db_session, session_id)
        row_id = str(uuid.uuid4())
        await db_session.execute(
            text(
                "INSERT INTO cost_records "
                "(id, session_id, user_id, run_id, node_name, model, "
                " provider, pricing_version, cost_status) "
                "VALUES (:id, :sid, :uid, :rid, :nn, :m, :p, :pv, :cs)"
            ),
            {
                "id": row_id,
                "sid": session_id,
                "uid": user_id,
                "rid": str(uuid.uuid4()),
                "nn": "test_node",
                "m": "test-model",
                "p": "test-provider",
                "pv": "test_v1",
                "cs": "actual",
            },
        )
        result = await db_session.execute(
            text(
                "SELECT step_ix, attempt_ix, input_tokens, output_tokens, "
                "  cache_read_tokens, cache_write_tokens, reasoning_tokens, "
                "  total_usd, created_at "
                "FROM cost_records WHERE id = :id"
            ),
            {"id": row_id},
        )
        row = result.one()
        assert row.step_ix == 0
        assert row.attempt_ix == 0
        assert row.input_tokens == 0
        assert row.output_tokens == 0
        assert row.cache_read_tokens == 0
        assert row.cache_write_tokens == 0
        assert row.reasoning_tokens == 0
        assert row.total_usd == Decimal("0")
        assert row.created_at is not None


class TestConsumerContracts:
    async def test_repository_insert_is_idempotent_on_run_id(self, db_session):
        """Two inserts, same run_id, different payload → first row wins."""
        user_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        await _ensure_user(db_session, user_id)
        await _ensure_session(db_session, session_id)

        repo = DbCostRecordRepository(db_session)
        first_total = Decimal("1.2345678900")
        second_total = Decimal("9.9999999900")
        first_id = str(uuid.uuid4())
        second_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        first = CostRecord(
            id=first_id,
            session_id=session_id,
            user_id=user_id,
            run_id=run_id,
            node_name="first_node",
            step_ix=0,
            attempt_ix=0,
            model="first_model",
            provider="first_provider",
            input_tokens=10,
            output_tokens=20,
            cache_read_tokens=0,
            cache_write_tokens=0,
            reasoning_tokens=0,
            total_usd=first_total,
            pricing_version="v1",
            cost_status=CostStatus.ACTUAL,
            created_at=now,
        )
        second = CostRecord(
            id=second_id,
            session_id=session_id,
            user_id=user_id,
            run_id=run_id,
            node_name="second_node",
            step_ix=99,
            attempt_ix=99,
            model="second_model",
            provider="second_provider",
            input_tokens=999,
            output_tokens=999,
            cache_read_tokens=0,
            cache_write_tokens=0,
            reasoning_tokens=0,
            total_usd=second_total,
            pricing_version="v2",
            cost_status=CostStatus.UNKNOWN,
            created_at=now,
        )
        await repo.insert(first)
        await repo.insert(second)

        result = await db_session.execute(
            text(
                "SELECT id, model, provider, total_usd, pricing_version, "
                "       cost_status "
                "FROM cost_records WHERE run_id = :rid"
            ),
            {"rid": run_id},
        )
        rows = result.all()
        assert len(rows) == 1
        row = rows[0]
        assert row.id == first_id
        assert row.model == "first_model"
        assert row.provider == "first_provider"
        assert row.total_usd == first_total
        assert row.pricing_version == "v1"
        assert row.cost_status == "actual"

    async def test_aggregation_marks_partial_on_sentinel_only(self, db_session):
        """Sentinel-only session → degraded-marker override → PARTIAL."""
        user_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        await _ensure_user(db_session, user_id)
        await _ensure_session(db_session, session_id)
        await _raw_insert_full(
            db_session,
            user_id=user_id,
            session_id=session_id,
            node_name="persist_degraded",
            cost_status="unknown",
            model="session_degraded_marker",
            provider="internal",
        )
        repo = DbCostRecordRepository(db_session)
        agg_service = CostAggregationService(repo)
        agg = await agg_service.get_aggregate(session_id)

        assert agg.cost_status == CostStatus.PARTIAL
        assert "session_degraded_marker" in agg.by_model
        assert "internal" in agg.by_provider
        for bucket in (agg.by_model, agg.by_provider, agg.by_node):
            for key in bucket.keys():
                assert key, f"empty key found in bucket: {bucket!r}"
        assert agg.has_partial_records is True
