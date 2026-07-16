"""SPM PR-1c Task 15 — ``DBSessionRepository.add_file_if_absent`` JSONB predicate.

CI-only: needs a real (pgvector) PostgreSQL — the JSONB ``@>`` contains predicate
+ rowcount-0 disambiguation cannot be exercised against SQLite/mocks. Locally this
is skipped unless ``SQLALCHEMY_DATABASE_URL`` points at a reachable test DB (see the
"Local Test Infrastructure" section of ``CLAUDE.md`` — never point it at the dev
``manus`` DB). CI (``ci.yml``) provides ``manus_test``.

Verifies:
* same ``file.id`` written twice → ``files`` length stays 1 (idempotent no-op);
* two distinct ids → both retained (length 2);
* legacy ``add_file`` still appends UNCONDITIONALLY (same id twice → length 2,
  INV-SPM-2 — always-path behavior unchanged);
* a missing session still raises ``ValueError`` (contract preserved).
"""
from __future__ import annotations

import pytest
from sqlalchemy import select

from app.domain.models.file import File
from app.infrastructure.models.session import SessionModel
from app.infrastructure.repositories.db_session_repository import DBSessionRepository

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


async def _files(db_session, session_id: str) -> list:
    db_session.expire_all()  # drop identity-map cache so the Core UPDATE is visible
    row = (
        await db_session.execute(
            select(SessionModel).where(SessionModel.id == session_id)
        )
    ).scalar_one()
    return list(row.files or [])


async def test_add_file_if_absent_same_id_idempotent(db_session, seed_session):
    repo = DBSessionRepository(db_session=db_session)
    sid = seed_session.id
    f = File(id="f1", filename="f1.pdf", filepath="/home/ubuntu/upload/f1.pdf")

    await repo.add_file_if_absent(sid, f)
    await repo.add_file_if_absent(sid, f)  # second write = idempotent no-op

    files = await _files(db_session, sid)
    assert [x["id"] for x in files] == ["f1"]


async def test_add_file_if_absent_distinct_ids_both_retained(db_session, seed_session):
    repo = DBSessionRepository(db_session=db_session)
    sid = seed_session.id

    await repo.add_file_if_absent(sid, File(id="f1", filename="f1.pdf"))
    await repo.add_file_if_absent(sid, File(id="f2", filename="f2.pdf"))

    files = await _files(db_session, sid)
    assert sorted(x["id"] for x in files) == ["f1", "f2"]


async def test_legacy_add_file_still_unconditional(db_session, seed_session):
    """INV-SPM-2: legacy ``add_file`` is byte-untouched — it still appends the same
    file id twice (two rows), unlike the new id-idempotent method."""
    repo = DBSessionRepository(db_session=db_session)
    sid = seed_session.id
    f = File(id="dup", filename="dup.pdf")

    await repo.add_file(sid, f)
    await repo.add_file(sid, f)

    files = await _files(db_session, sid)
    assert [x["id"] for x in files] == ["dup", "dup"]


async def test_add_file_if_absent_missing_session_raises(db_session):
    repo = DBSessionRepository(db_session=db_session)
    with pytest.raises(ValueError):
        await repo.add_file_if_absent(
            "sess-does-not-exist-xyz", File(id="f1", filename="f1.pdf")
        )
