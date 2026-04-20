"""R5 CS4 ApprovalStateWriter 真 DB 集成测试（Codex round-2 CRIT-1 / round-3 MED 回归守卫）。

本文件锁住：
1. aware UTC ``expires_at`` 下推到 ``TIMESTAMP WITHOUT TIME ZONE`` 列不应抛
   ``DataError``（``to_naive_utc`` 边界归一修复）
2. Writer happy path 在真 DB 的端到端可跑性
3. IntegrityError → find_by_confirmation_id 回读分支的幂等语义
4. (scope, effect) 冲突 raise ValueError
5. delete_grant 对称删 grant + audit log

测试夹具走**生产路径**：``DBUnitOfWork(session_factory=...)``，让 writer 的
``async with uow_factory()`` 正常 commit/rollback；不复用 conftest 的 rollback-on-
teardown ``db_session`` fixture（Codex round-3 MED：那种夹具会让 writer 的
IntegrityError 分支 ``await uow.rollback()`` 把外层 ``session.begin()`` 事务一并
关掉，后续 SELECT 报 InvalidRequestError）。

清盘策略：每 test 随机 user_id，teardown 用独立 session ``DELETE FROM users``
级联删除所有 grants / audit log / rules 行。FK ``ON DELETE CASCADE`` 兜底。

运行：

    cd api && SQLALCHEMY_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/manus_test \\
        uv run pytest tests/integration/test_approval_state_writer_db.py -v
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.application.services.approval_state_writer import ApprovalStateWriter
from app.domain.models.approval_grant import ApprovalDecision
from app.domain.services.approval_grant_policy import session_grant_expires_at
from app.infrastructure.repositories.db_uow import DBUnitOfWork

pytestmark = pytest.mark.anyio


@pytest.fixture
async def session_factory(async_engine):
    """独立 sessionmaker；每 UoW.__aenter__ 都开新 session，commit 走生产路径。"""
    return async_sessionmaker(bind=async_engine, expire_on_commit=False)


@pytest.fixture
async def writer(session_factory) -> ApprovalStateWriter:
    """生产路径 writer：真实 ``DBUnitOfWork`` + 独立 session factory。"""

    def _factory() -> DBUnitOfWork:
        return DBUnitOfWork(session_factory=session_factory)

    return ApprovalStateWriter(uow_factory=_factory)


@pytest.fixture
async def test_user_id(session_factory) -> str:
    """随机 user_id，teardown DELETE 级联清 grants + audit log + rules。"""
    uid = str(uuid.uuid4())
    async with session_factory() as session:
        await session.execute(
            text("INSERT INTO users (id) VALUES (:uid) ON CONFLICT DO NOTHING"),
            {"uid": uid},
        )
        await session.commit()
    yield uid
    async with session_factory() as session:
        await session.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": uid})
        await session.commit()


async def _select_grant(session_factory, decision_id: str):
    async with session_factory() as session:
        row = (
            await session.execute(
                text("SELECT * FROM tool_approval_grants WHERE decision_id = :did"),
                {"did": decision_id},
            )
        ).mappings().one_or_none()
        return row


async def _count_audit(session_factory, decision_id: str) -> int:
    async with session_factory() as session:
        return (
            await session.execute(
                text(
                    "SELECT COUNT(*) FROM tool_approval_log WHERE decision_id = :did"
                ),
                {"did": decision_id},
            )
        ).scalar() or 0


async def test_writer_happy_path_session_scope_with_aware_expires_at(
    writer: ApprovalStateWriter, session_factory, test_user_id: str
) -> None:
    """Codex CRIT-1 回归：aware UTC ``expires_at`` 下推到真 Postgres 应该成功。"""
    expires = session_grant_expires_at()  # aware UTC
    assert expires.tzinfo is not None  # policy 产出的是 aware

    decision = ApprovalDecision(
        user_id=test_user_id,
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
        expires_at=expires,
        risk_level="medium",
    )

    # 不应抛 DataError
    decision_id, newly_created = await writer.write(decision)
    assert newly_created is True
    assert decision_id

    row = await _select_grant(session_factory, decision_id)
    assert row is not None
    assert row["scope"] == "session"
    assert row["effect"] == "approve"
    assert row["expires_at"] is not None
    assert row["expires_at"].tzinfo is None  # DB 回读永远 naive
    delta = abs((row["expires_at"] - expires.replace(tzinfo=None)).total_seconds())
    assert delta < 5, f"expires_at 墙钟对不上：got={row['expires_at']} expected={expires}"


async def test_writer_happy_path_always_scope(
    writer: ApprovalStateWriter, session_factory, test_user_id: str
) -> None:
    """always scope：session_id=None / expires_at=None，不触发任何约束违例。"""
    decision = ApprovalDecision(
        user_id=test_user_id,
        session_id=None,
        tool_name="shell_execute",
        tool_source="native",
        arg_digest="d1",
        primary_arg="ls *",
        dir_arg="",
        scope="always",
        effect="approve",
        source_type="user_click",
        confirmation_id=str(uuid.uuid4()),
        expires_at=None,
        risk_level="medium",
    )
    decision_id, newly_created = await writer.write(decision)
    assert newly_created is True

    row = await _select_grant(session_factory, decision_id)
    assert row is not None
    assert row["scope"] == "always"
    assert row["session_id"] is None
    assert row["expires_at"] is None


async def test_writer_idempotent_same_confirmation_id(
    writer: ApprovalStateWriter, session_factory, test_user_id: str
) -> None:
    """并发重入：同 confirmation_id 第二次 write 返 ``(existing_id, False)``。"""
    conf_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    base_kwargs = dict(
        user_id=test_user_id,
        session_id=session_id,
        tool_name="shell_execute",
        tool_source="native",
        arg_digest="d1",
        primary_arg="ls *",
        dir_arg="",
        scope="session",
        effect="approve",
        source_type="user_click",
        confirmation_id=conf_id,
        expires_at=session_grant_expires_at(),
        risk_level="medium",
    )

    first_id, first_new = await writer.write(ApprovalDecision(**base_kwargs))
    assert first_new is True

    # 第二次同 confirmation_id 同 (scope, effect) → 幂等返 existing
    second_id, second_new = await writer.write(
        ApprovalDecision(**{**base_kwargs, "expires_at": session_grant_expires_at()})
    )
    assert second_new is False
    assert second_id == first_id

    # grants 表仍然只有 1 行带此 confirmation_id
    async with session_factory() as session:
        count = (
            await session.execute(
                text(
                    "SELECT COUNT(*) FROM tool_approval_grants WHERE confirmation_id = :cid"
                ),
                {"cid": conf_id},
            )
        ).scalar()
        assert count == 1


async def test_writer_conflict_different_effect_raises(
    writer: ApprovalStateWriter, session_factory, test_user_id: str
) -> None:
    """同 confirmation_id 不同 effect → raise ValueError。"""
    conf_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    # 先写 approve
    await writer.write(
        ApprovalDecision(
            user_id=test_user_id,
            session_id=session_id,
            tool_name="shell_execute",
            tool_source="native",
            arg_digest="d1",
            primary_arg="ls *",
            dir_arg="",
            scope="session",
            effect="approve",
            source_type="user_click",
            confirmation_id=conf_id,
            expires_at=session_grant_expires_at(),
            risk_level="medium",
        )
    )
    # 再写同 confirmation_id 但 effect=deny → 语义冲突
    with pytest.raises(ValueError, match="claim collision"):
        await writer.write(
            ApprovalDecision(
                user_id=test_user_id,
                session_id=session_id,
                tool_name="shell_execute",
                tool_source="native",
                arg_digest="d1",
                primary_arg="ls *",
                dir_arg="",
                scope="session",
                effect="deny",
                source_type="user_click",
                confirmation_id=conf_id,
                expires_at=session_grant_expires_at(),
                risk_level="medium",
            )
        )


async def test_writer_delete_grant_cascades_audit(
    writer: ApprovalStateWriter, session_factory, test_user_id: str
) -> None:
    """delete_grant 同事务删 grant + audit log 行（独立 session 验证可见性）。"""
    decision_id, _ = await writer.write(
        ApprovalDecision(
            user_id=test_user_id,
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
            expires_at=session_grant_expires_at(),
            risk_level="medium",
        )
    )

    # 写入后 grant + audit 各 1 行
    assert await _select_grant(session_factory, decision_id) is not None
    assert await _count_audit(session_factory, decision_id) == 1

    # delete_grant → 两表都空
    await writer.delete_grant(decision_id)

    assert await _select_grant(session_factory, decision_id) is None
    assert await _count_audit(session_factory, decision_id) == 0
