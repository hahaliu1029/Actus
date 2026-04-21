"""R5 CS4 concurrent claim 合同集成测试。

TODO2.md line 1195 成功标准：``test_concurrent_confirm_and_resume`` 通过
即可认定 CS4 "单一 writer + 并发 race 根除" 已交付。

覆盖 3 条并发 claim 路径：

1. **session scope 并发**（persistent / DB UNIQUE 原子 claim）
   两个同 ``confirmation_id`` 同时 ``preflight_resume_tool_confirmation``，
   结果必须是 1 winner（state.decision_id） + 1 HTTP 409；DB 中仅 1 个 grant +
   1 个 audit 行挂到该 decision_id。

2. **session scope kickoff 失败 → sequential retry**
   winner claim 成功后模拟 drive 阶段 ``task.resume`` 失败触发的
   ``_rollback_resume_claim``，必须把 grant + audit 干净删掉；之后同
   ``confirmation_id`` 的第二次 preflight 必须重新赢（``newly_created=True``）。

3. **once scope 并发**（ConfirmationManager.mark_processing_if_pending /
   Redis Lua CAS）
   once path 不建 grant，claim 由 Redis Lua 原子 CAS 把守。两个并发
   preflight 必须 1 winner + 1 HTTP 409；winner audit 经 ``write_audit_only``
   落 ``tool_approval_log``（scope='once'，decision_id IS NULL）。
   Stage 1 刚把 once audit 收进 writer，此 case 是 writer.write_audit_only 的
   integration guard。

运行（需真 pg + redis）::

    cd api && \\
        SQLALCHEMY_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/manus_test \\
        ACTUS_TEST_REDIS_URL=redis://localhost:6379/15 \\
        uv run pytest tests/integration/test_concurrent_confirm_and_resume.py -v

``ACTUS_TEST_REDIS_URL`` 未设且 fallback ``redis://localhost:6379/15`` 连不上时，
整个文件 skip（CI 里由 ci.yml 提供 redis 服务）。**不**使用通用 ``REDIS_URL``
以免误连开发或共享实例。
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator
from unittest.mock import MagicMock

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.application.errors.exceptions import ConflictError
from app.application.services.agent_service import AgentService
from app.domain.services.confirmation_manager import (
    ZSET_KEY,
    ConfirmationDetail,
    ConfirmationManager,
)
from app.infrastructure.repositories.db_uow import DBUnitOfWork

from tests.app.application.services.conftest import default_snapshot

pytestmark = pytest.mark.anyio


_REDIS_URL_ENV = "ACTUS_TEST_REDIS_URL"
_DEFAULT_REDIS_URL = "redis://localhost:6379/15"
_SESSION_PREFIX = "race_test_sess_"


# ---------------------------------------------------------------------------
# Fixtures — inline per design doc (user approval 2026-04-21)：不修改现有
# integration/conftest.py，等第 3 个 Redis-dep integration test 再提取。
# ---------------------------------------------------------------------------


async def _probe_redis(url: str):
    """ping，失败返 None。用于 fixture 在不可连时 skip。"""
    try:
        from redis.asyncio import Redis

        client = Redis.from_url(url, decode_responses=True)
        await client.ping()
        return client
    except Exception:  # noqa: BLE001 — broad catch 仅用于 skip 决策
        return None


@pytest.fixture
async def redis_client():
    """真 Redis 连接；连不上 skip 整个 case。

    teardown 只删本测试文件可能写入的 key（``confirmation_detail:<prefix>*``
    + ZSET ``confirmation_deadlines`` 中 ``<prefix>*`` 成员）——**不**
    ``flushdb()``，防止误删共享 redis 实例里其他数据。
    """
    url = os.environ.get(_REDIS_URL_ENV, _DEFAULT_REDIS_URL)
    client = await _probe_redis(url)
    if client is None:
        pytest.skip(
            f"Redis 不可连（{_REDIS_URL_ENV}={url}）；"
            "concurrent claim integration 测试需要真 redis 以驱动 Lua CAS。"
        )
    try:
        yield client
    finally:
        # 精确清理：hash keys + ZSET members
        hash_keys = await client.keys(f"confirmation_detail:{_SESSION_PREFIX}*")
        if hash_keys:
            await client.delete(*hash_keys)
        members = await client.zrangebyscore(ZSET_KEY, "-inf", "+inf")
        to_remove = [m for m in members if m.startswith(_SESSION_PREFIX)]
        if to_remove:
            await client.zrem(ZSET_KEY, *to_remove)
        await client.aclose()


@pytest.fixture
async def session_factory(async_engine):
    return async_sessionmaker(bind=async_engine, expire_on_commit=False)


@pytest.fixture
def uow_factory(session_factory):
    def _factory() -> DBUnitOfWork:
        return DBUnitOfWork(session_factory=session_factory)

    return _factory


@pytest.fixture
async def test_user_id(session_factory) -> AsyncIterator[str]:
    """随机 user_id + INSERT + teardown DELETE（FK CASCADE 级联删 sessions /
    grants / audit / rules）。"""
    uid = str(uuid.uuid4())
    async with session_factory() as s:
        await s.execute(
            text("INSERT INTO users (id) VALUES (:uid) ON CONFLICT DO NOTHING"),
            {"uid": uid},
        )
        await s.commit()
    yield uid
    async with session_factory() as s:
        await s.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": uid})
        await s.commit()


@pytest.fixture
async def test_session_id(session_factory, test_user_id) -> str:
    """插 sessions 行（FK 到 users），teardown 由 user cascade 负责。

    session_id 使用固定前缀 ``race_test_sess_`` 以便 redis teardown 精确清理。
    """
    sid = f"{_SESSION_PREFIX}{uuid.uuid4().hex[:12]}"
    async with session_factory() as s:
        # status 显式传 'running'：DB server_default 是 '' 但 Pydantic
        # ``SessionStatus`` 不含空串，``to_domain()`` 会 ValidationError。
        await s.execute(
            text(
                "INSERT INTO sessions (id, user_id, status) "
                "VALUES (:sid, :uid, 'running')"
            ),
            {"sid": sid, "uid": test_user_id},
        )
        await s.commit()
    return sid


@pytest.fixture
def confirmation_mgr(redis_client) -> ConfirmationManager:
    return ConfirmationManager(redis=redis_client, timeout_seconds=300)


@pytest.fixture
def service(uow_factory, confirmation_mgr) -> AgentService:
    """AgentService with real uow + real ConfirmationManager + patched task methods.

    J1 (user approval 2026-04-21)：task lifecycle 与 claim race 无关，统一 mock。
    """
    svc = AgentService(
        uow_factory=uow_factory,
        config_snapshot=default_snapshot(),
        sandbox_cls=object,
        task_cls=object,
        search_engine=object(),
        file_storage=object(),
    )
    svc._confirmation_manager = confirmation_mgr

    fake_task = MagicMock()
    fake_task.done = True

    async def _get_task(_session):
        return fake_task

    async def _create_task(_session):
        return fake_task

    async def _safe_update(_sid: str):
        return None

    svc._get_task = _get_task  # type: ignore[assignment]
    svc._create_task = _create_task  # type: ignore[assignment]
    svc._safe_update_unread_count = _safe_update  # type: ignore[assignment]
    return svc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detail(
    *, session_id: str, user_id: str, tool_call_id: str = "tc-race-1"
) -> ConfirmationDetail:
    return ConfirmationDetail(
        session_id=session_id,
        tool_call_id=tool_call_id,
        user_id=user_id,
        tool_name="shell_execute",
        tool_args={"command": "ls /"},
        risk_level="medium",
        arg_digest="digest-race-1",
        primary_arg="ls *",
        dir_arg="",
        matched_patterns=[],
        deadline_ts=(datetime.now(timezone.utc) + timedelta(hours=1)).timestamp(),
        status="pending",
    )


class _Confirmation:
    """Minimal ``tool_confirmation`` payload shape for preflight."""

    def __init__(self, *, action: str, scope: str, tool_call_id: str) -> None:
        self.action = action
        self.scope = scope
        self.tool_call_id = tool_call_id


async def _count_grants(session_factory, confirmation_id: str) -> int:
    async with session_factory() as s:
        return (
            await s.execute(
                text(
                    "SELECT COUNT(*) FROM tool_approval_grants "
                    "WHERE confirmation_id = :cid"
                ),
                {"cid": confirmation_id},
            )
        ).scalar() or 0


async def _count_audit(
    session_factory, *, session_id: str, scope: str
) -> int:
    """count audit rows for a given session_id + scope."""
    async with session_factory() as s:
        return (
            await s.execute(
                text(
                    "SELECT COUNT(*) FROM tool_approval_log "
                    "WHERE session_id = :sid AND scope = :scope"
                ),
                {"sid": session_id, "scope": scope},
            )
        ).scalar() or 0


async def _gather_with_results(coros):
    """asyncio.gather with return_exceptions=True; returns list."""
    return await asyncio.gather(*coros, return_exceptions=True)


def _split_winner_and_409(results: list) -> tuple[list, list[ConflictError]]:
    """Partition results into (non-exception values, ConflictError list)."""
    winners = []
    conflicts = []
    for r in results:
        if isinstance(r, ConflictError):
            conflicts.append(r)
        elif isinstance(r, BaseException):
            raise AssertionError(f"unexpected exception in concurrent preflight: {r!r}")
        else:
            winners.append(r)
    return winners, conflicts


# ---------------------------------------------------------------------------
# Case 1: session scope 并发（persistent / DB UNIQUE claim）
# ---------------------------------------------------------------------------


async def test_session_scope_concurrent_resume_has_one_winner(
    service: AgentService,
    confirmation_mgr: ConfirmationManager,
    session_factory,
    test_session_id: str,
    test_user_id: str,
) -> None:
    """persistent scope：2 个同 ``confirmation_id`` 并发 resume → 1 winner + 1 409。

    **Invariant**：DB 中仅 1 个 grant 挂到该 ``confirmation_id``，audit 行数 == 1
    且 ``decision_id`` 指向 winner。
    """
    tool_call_id = f"tc-race-session-{uuid.uuid4().hex[:8]}"
    detail = _make_detail(
        session_id=test_session_id, user_id=test_user_id, tool_call_id=tool_call_id
    )
    await confirmation_mgr.store(detail)

    conf = _Confirmation(action="approve", scope="session", tool_call_id=tool_call_id)

    results = await _gather_with_results(
        [
            service.preflight_resume_tool_confirmation(
                session_id=test_session_id,
                user_id=test_user_id,
                is_admin=False,
                tool_confirmation=conf,
            ),
            service.preflight_resume_tool_confirmation(
                session_id=test_session_id,
                user_id=test_user_id,
                is_admin=False,
                tool_confirmation=conf,
            ),
        ]
    )
    winners, conflicts = _split_winner_and_409(results)

    assert len(winners) == 1, f"预期 1 winner，实际 {len(winners)}: {results!r}"
    assert len(conflicts) == 1, f"预期 1 ConflictError，实际 {len(conflicts)}"
    assert conflicts[0].status_code == 409

    winner_state = winners[0]
    assert winner_state.decision_id is not None
    assert winner_state.persistent_scope is True

    # DB invariant: 1 grant + 1 audit（R5 writer.write 原子 grant+audit）
    assert await _count_grants(session_factory, tool_call_id) == 1
    assert (
        await _count_audit(session_factory, session_id=test_session_id, scope="session")
        == 1
    )


# ---------------------------------------------------------------------------
# Case 2: kickoff 失败 → sequential retry 必须能重新赢
# ---------------------------------------------------------------------------


async def test_session_scope_kickoff_rollback_allows_fresh_retry(
    service: AgentService,
    confirmation_mgr: ConfirmationManager,
    session_factory,
    test_session_id: str,
    test_user_id: str,
) -> None:
    """J2 (user approval)：winner claim 后直接调 ``_rollback_resume_claim``
    模拟 drive 阶段 ``task.resume`` 失败触发的回滚，然后 sequential retry
    必须重新成为 ``newly_created=True`` 的 winner。

    **Invariant**：
    - rollback 后 grant + audit 行数回 0，confirmation 状态回 pending
    - retry winner 有新的 ``decision_id``（与第一次不等）
    - 最终 DB 仅 1 grant（来自 retry） + 1 audit
    """
    tool_call_id = f"tc-race-rollback-{uuid.uuid4().hex[:8]}"
    detail = _make_detail(
        session_id=test_session_id, user_id=test_user_id, tool_call_id=tool_call_id
    )
    await confirmation_mgr.store(detail)

    conf = _Confirmation(action="approve", scope="session", tool_call_id=tool_call_id)

    # 第一次 preflight → winner
    first_state = await service.preflight_resume_tool_confirmation(
        session_id=test_session_id,
        user_id=test_user_id,
        is_admin=False,
        tool_confirmation=conf,
    )
    first_decision_id = first_state.decision_id
    assert first_decision_id is not None

    # 模拟 drive 阶段 task.resume 失败 → 触发 rollback 骨架
    await service._rollback_resume_claim(
        persistent_scope=True,
        decision_id=first_decision_id,
        session_id=test_session_id,
        tool_call_id=tool_call_id,
    )

    # rollback 后 grant + audit 都该 0（writer.delete_grant 对称删）
    assert await _count_grants(session_factory, tool_call_id) == 0
    assert (
        await _count_audit(session_factory, session_id=test_session_id, scope="session")
        == 0
    )
    # confirmation 回 pending（mark_pending 执行）
    detail_after = await confirmation_mgr.read(test_session_id, tool_call_id)
    assert detail_after is not None
    assert detail_after.status == "pending"

    # 第二次 preflight → 必须重新赢（新 decision_id）
    second_state = await service.preflight_resume_tool_confirmation(
        session_id=test_session_id,
        user_id=test_user_id,
        is_admin=False,
        tool_confirmation=conf,
    )
    assert second_state.decision_id is not None
    assert second_state.decision_id != first_decision_id, (
        "retry winner 必须拿到新 decision_id；若相同说明 rollback 没删 grant 行"
    )

    # 最终 DB 状态：仅 1 grant（retry 的） + 1 audit
    assert await _count_grants(session_factory, tool_call_id) == 1
    assert (
        await _count_audit(session_factory, session_id=test_session_id, scope="session")
        == 1
    )


# ---------------------------------------------------------------------------
# Case 3: once scope 并发（Redis Lua CAS / write_audit_only）
# ---------------------------------------------------------------------------


async def test_once_scope_concurrent_resume_has_one_winner(
    service: AgentService,
    confirmation_mgr: ConfirmationManager,
    session_factory,
    test_session_id: str,
    test_user_id: str,
) -> None:
    """once scope：2 个同 ``confirmation_id`` 并发 resume → 1 winner + 1 409。

    **Invariants**：
    - claim 由 Redis Lua CAS（``mark_processing_if_pending``）把守，不是 DB UNIQUE
    - winner audit 经 ``writer.write_audit_only`` 落 ``tool_approval_log``
      （Stage 1 刚接入；此 case 是合同收口后的 integration guard）
    - ``tool_approval_grants`` 行数 == 0（once 不建 grant）
    - ``tool_approval_log`` 行数 == 1，``scope='once'`` 且 ``decision_id IS NULL``
    """
    tool_call_id = f"tc-race-once-{uuid.uuid4().hex[:8]}"
    detail = _make_detail(
        session_id=test_session_id, user_id=test_user_id, tool_call_id=tool_call_id
    )
    await confirmation_mgr.store(detail)

    conf = _Confirmation(action="approve", scope="once", tool_call_id=tool_call_id)

    results = await _gather_with_results(
        [
            service.preflight_resume_tool_confirmation(
                session_id=test_session_id,
                user_id=test_user_id,
                is_admin=False,
                tool_confirmation=conf,
            ),
            service.preflight_resume_tool_confirmation(
                session_id=test_session_id,
                user_id=test_user_id,
                is_admin=False,
                tool_confirmation=conf,
            ),
        ]
    )
    winners, conflicts = _split_winner_and_409(results)

    assert len(winners) == 1, f"预期 1 winner，实际 {len(winners)}: {results!r}"
    assert len(conflicts) == 1
    assert conflicts[0].status_code == 409

    winner_state = winners[0]
    assert winner_state.persistent_scope is False
    assert winner_state.decision_id is None  # once 不建 grant

    # DB invariants：once 不写 grant，只写 audit（经 write_audit_only）
    assert await _count_grants(session_factory, tool_call_id) == 0
    async with session_factory() as s:
        rows = (
            await s.execute(
                text(
                    "SELECT scope, decision_id, approved_by "
                    "FROM tool_approval_log WHERE session_id = :sid"
                ),
                {"sid": test_session_id},
            )
        ).mappings().all()
    assert len(rows) == 1, f"预期 1 audit 行，实际 {len(rows)}"
    assert rows[0]["scope"] == "once"
    assert rows[0]["decision_id"] is None, (
        "once audit 的 decision_id 必须为 NULL——once 不建 grant，"
        "writer.write_audit_only 不应透传任何 decision_id"
    )
    assert rows[0]["approved_by"] == "user"
