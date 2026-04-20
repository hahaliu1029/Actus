"""R5 CS4 ``tool_approval_grants`` schema 集成测试（需要 Postgres）。

覆盖 test plan §1 的 DB-level 断言：
- ck_scope / ck_effect / ck_tool_source / ck_session_scope_has_session_id /
  ck_expires_at_only_for_session 五个 CHECK 约束
- confirmation_id UNIQUE（主 UNIQUE）
- SmartApprove partial UNIQUE (confirmation_id IS NULL) 去重

运行：

    cd api && SQLALCHEMY_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/manus_test \\
        uv run pytest tests/integration/test_r5_grants_schema.py -v
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from app.infrastructure.models.tool_approval_grant import ToolApprovalGrantModel

pytestmark = pytest.mark.anyio


def _make_grant(**overrides) -> ToolApprovalGrantModel:
    """构造一条合法的 session scope approve grant；测试里按需 override 字段。"""
    defaults = dict(
        decision_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        session_id=str(uuid.uuid4()),
        tool_name="shell_execute",
        tool_source="native",
        arg_digest="d1",
        primary_arg="ls *",
        dir_arg="",
        scope="session",
        effect="approve",
        source_type="user_click",
        confirmation_id=str(uuid.uuid4()),
        expires_at=(datetime.now(timezone.utc) + timedelta(hours=24)).replace(tzinfo=None),
    )
    defaults.update(overrides)
    return ToolApprovalGrantModel(**defaults)


async def _ensure_user(db_session, user_id: str) -> None:
    await db_session.execute(
        text("INSERT INTO users (id) VALUES (:uid) ON CONFLICT DO NOTHING"),
        {"uid": user_id},
    )


# ---------------- CHECK constraints ----------------


async def test_ck_scope_rejects_invalid_value(db_session) -> None:
    user_id = str(uuid.uuid4())
    await _ensure_user(db_session, user_id)
    grant = _make_grant(user_id=user_id, scope="forever")
    db_session.add(grant)
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            await db_session.flush()


async def test_ck_effect_rejects_invalid_value(db_session) -> None:
    user_id = str(uuid.uuid4())
    await _ensure_user(db_session, user_id)
    grant = _make_grant(user_id=user_id, effect="maybe")
    db_session.add(grant)
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            await db_session.flush()


async def test_ck_tool_source_rejects_invalid_value(db_session) -> None:
    user_id = str(uuid.uuid4())
    await _ensure_user(db_session, user_id)
    grant = _make_grant(user_id=user_id, tool_source="unknown")
    db_session.add(grant)
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            await db_session.flush()


async def test_ck_session_scope_requires_session_id(db_session) -> None:
    """scope='session' + session_id IS NULL → 违反 CHECK。"""
    user_id = str(uuid.uuid4())
    await _ensure_user(db_session, user_id)
    grant = _make_grant(user_id=user_id, session_id=None)
    db_session.add(grant)
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            await db_session.flush()


async def test_ck_always_scope_forbids_session_id(db_session) -> None:
    """scope='always' + session_id NOT NULL → 违反 CHECK。"""
    user_id = str(uuid.uuid4())
    await _ensure_user(db_session, user_id)
    grant = _make_grant(
        user_id=user_id,
        scope="always",
        session_id=str(uuid.uuid4()),  # 违例：always scope 不能带 session_id
        expires_at=None,
    )
    db_session.add(grant)
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            await db_session.flush()


async def test_ck_expires_at_required_for_session_scope(db_session) -> None:
    """scope='session' + expires_at IS NULL → 违反 CHECK。"""
    user_id = str(uuid.uuid4())
    await _ensure_user(db_session, user_id)
    grant = _make_grant(user_id=user_id, expires_at=None)
    db_session.add(grant)
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            await db_session.flush()


async def test_ck_expires_at_forbidden_for_always_scope(db_session) -> None:
    """scope='always' + expires_at NOT NULL → 违反 CHECK。"""
    user_id = str(uuid.uuid4())
    await _ensure_user(db_session, user_id)
    grant = _make_grant(
        user_id=user_id,
        scope="always",
        session_id=None,
        expires_at=(datetime.now(timezone.utc) + timedelta(hours=24)).replace(tzinfo=None),
    )
    db_session.add(grant)
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            await db_session.flush()


# ---------------- UNIQUE constraints ----------------


async def test_confirmation_id_unique(db_session) -> None:
    """同 confirmation_id 两次 INSERT → 第二次 IntegrityError。"""
    user_id = str(uuid.uuid4())
    await _ensure_user(db_session, user_id)
    conf_id = str(uuid.uuid4())

    g1 = _make_grant(user_id=user_id, confirmation_id=conf_id)
    db_session.add(g1)
    await db_session.flush()

    g2 = _make_grant(user_id=user_id, confirmation_id=conf_id)  # 相同 confirmation_id
    db_session.add(g2)
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            await db_session.flush()


async def test_smartapprove_partial_unique_dedup(db_session) -> None:
    """同 (user, session, tool, arg_digest, effect) + confirmation_id=NULL
    两次 INSERT → 第二次 IntegrityError（partial UNIQUE 去重）。"""
    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    await _ensure_user(db_session, user_id)

    g1 = _make_grant(
        user_id=user_id,
        session_id=session_id,
        tool_name="shell_execute",
        arg_digest="digest-shared",
        effect="approve",
        confirmation_id=None,
    )
    db_session.add(g1)
    await db_session.flush()

    g2 = _make_grant(
        user_id=user_id,
        session_id=session_id,
        tool_name="shell_execute",
        arg_digest="digest-shared",
        effect="approve",
        confirmation_id=None,
    )
    db_session.add(g2)
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            await db_session.flush()


async def test_smartapprove_partial_unique_does_not_affect_rows_with_confirmation_id(
    db_session,
) -> None:
    """``confirmation_id`` 非 NULL 的行不参与 partial UNIQUE；同
    ``(user, session, tool, arg_digest, effect)`` 的两行只要 ``confirmation_id``
    不同就都能 INSERT。"""
    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    await _ensure_user(db_session, user_id)

    g1 = _make_grant(
        user_id=user_id,
        session_id=session_id,
        tool_name="shell_execute",
        arg_digest="d-same",
        effect="approve",
        confirmation_id=str(uuid.uuid4()),
    )
    g2 = _make_grant(
        user_id=user_id,
        session_id=session_id,
        tool_name="shell_execute",
        arg_digest="d-same",
        effect="approve",
        confirmation_id=str(uuid.uuid4()),
    )
    db_session.add_all([g1, g2])
    await db_session.flush()  # 不应抛

    stmt = select(ToolApprovalGrantModel).where(
        ToolApprovalGrantModel.user_id == user_id,
        ToolApprovalGrantModel.arg_digest == "d-same",
    )
    rows = (await db_session.execute(stmt)).scalars().all()
    assert len(rows) == 2


# ---------------- Happy-path smoke ----------------


async def test_session_grant_insert_and_read(db_session) -> None:
    user_id = str(uuid.uuid4())
    await _ensure_user(db_session, user_id)
    g = _make_grant(user_id=user_id)
    db_session.add(g)
    await db_session.flush()

    got = await db_session.get(ToolApprovalGrantModel, g.decision_id)
    assert got is not None
    assert got.scope == "session"
    assert got.effect == "approve"


async def test_always_grant_insert_and_read(db_session) -> None:
    user_id = str(uuid.uuid4())
    await _ensure_user(db_session, user_id)
    g = _make_grant(
        user_id=user_id,
        scope="always",
        session_id=None,
        expires_at=None,
    )
    db_session.add(g)
    await db_session.flush()

    got = await db_session.get(ToolApprovalGrantModel, g.decision_id)
    assert got is not None
    assert got.scope == "always"
    assert got.session_id is None
    assert got.expires_at is None
