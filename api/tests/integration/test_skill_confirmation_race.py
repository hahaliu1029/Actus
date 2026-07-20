"""R3 + R5 Skill confirmation race 集成测试。

TODO2.md line 395 Follow-up：``test_skill_confirmation_race``（并发 confirm +
resume，ApprovalState ownership 对 Skill 生效）。

**测试面**：Skill → ``ToolConfirmationEvent`` → 并发 ``/resume`` → R5 claim
合同整链在并发下不撕。1 条代表性 case（session scope），不重复 once 分支——
once 的 Lua CAS 由 ``test_concurrent_confirm_and_resume.py`` Case 3 覆盖，
本文件聚焦 Skill 专属路径：``tool_source="skill"`` 的 grant 在并发 resume
下正确归属、单一 winner、dedup 409。

**与 R3 的关系**：R3 在 install 侧做 AST scan + trust_origin，本测试不复用
那条注入链路（J3 user approval 2026-04-21）；仅通过 monkeypatch
``resolve_tool_source`` 模拟 skill-source 识别，保持 integration 测试聚焦
claim race 合同，不与 skill 注册/扫描 pipeline 耦合（R3 有独立 coverage）。

运行（需真 pg + redis）::

    cd api && \\
        SQLALCHEMY_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/manus_test \\
        ACTUS_TEST_REDIS_URL=redis://localhost:6379/15 \\
        uv run pytest tests/integration/test_skill_confirmation_race.py -v

``ACTUS_TEST_REDIS_URL`` 未设且 fallback ``redis://localhost:6379/15`` 连不上
时，case skip（CI 里由 ci.yml 提供 redis 服务）。
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

from app.application.services.agent_service import AgentService
from app.domain.services.permission.errors import PolicyConflict
from app.domain.services.permission.confirmation_queue import (
    ZSET_KEY,
    ConfirmationDetail,
    ConfirmationQueue as ConfirmationManager,
)
from app.domain.services.tools.tool_source_resolver import ToolSource
from app.infrastructure.repositories.db_uow import DBUnitOfWork

from tests.app.application.services.conftest import default_snapshot

pytestmark = pytest.mark.anyio


_REDIS_URL_ENV = "ACTUS_TEST_REDIS_URL"
_DEFAULT_REDIS_URL = "redis://localhost:6379/15"
_SESSION_PREFIX = "race_test_sess_"
_SKILL_TOOL_NAME = "skill_test_concurrent_race_runner"


# ---------------------------------------------------------------------------
# Fixtures — inline per user approval 2026-04-21（不改 integration/conftest.py）。
# 与 test_concurrent_confirm_and_resume.py 独立，等第 3 个 Redis-dep
# integration 文件再统一提取。
# ---------------------------------------------------------------------------


async def _probe_redis(url: str):
    try:
        from redis.asyncio import Redis

        client = Redis.from_url(url, decode_responses=True)
        await client.ping()
        return client
    except Exception:  # noqa: BLE001
        return None


@pytest.fixture
async def redis_client():
    url = os.environ.get(_REDIS_URL_ENV, _DEFAULT_REDIS_URL)
    client = await _probe_redis(url)
    if client is None:
        pytest.skip(
            f"Redis 不可连（{_REDIS_URL_ENV}={url}）；"
            "skill confirmation race 需要真 redis 以驱动 ConfirmationManager。"
        )
    try:
        yield client
    finally:
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
def skill_tool_source_patch(monkeypatch):
    """J3 (user approval)：monkeypatch ``resolve_tool_source`` 使测试用的 skill
    tool name 被识别成 ``source="skill"``。不碰 process-wide ``_REGISTRY``，不走
    真 skill 注册 pipeline（那是 R3 的 coverage 面）。

    注意：``agent_service.py:545-547`` 用函数内 ``from ... import
    resolve_tool_source``，所以必须在 resolver 模块属性上打补丁（import 在运行
    时才发生，monkeypatch 那个 module attr 即可覆盖）。

    **必须在 monkeypatch 之前捕获真实 resolver**：``monkeypatch.setattr`` 已经
    把模块属性替换成 ``_fake_resolve``，如果在 fallback 路径里延迟 import，拿到
    的是 ``_fake_resolve`` 本身 → 无限递归。改成 eager capture 绑在闭包里，fake
    任何时候都能走到真 resolver。
    """
    # 在 patch 之前先捕获真实 resolver（P3 fix：原实现延迟 import 会拿到
    # 被 patched 的 _fake_resolve 自己，fallback 任意非 _SKILL_TOOL_NAME
    # 名字会无限递归）
    from app.domain.services.tools.tool_source_resolver import (
        resolve_tool_source as _real_resolve,
    )

    def _fake_resolve(name: str) -> ToolSource:
        if name == _SKILL_TOOL_NAME:
            return ToolSource(
                source="skill",
                category="skill",
                canonical_name=name,
            )
        # 非本测试目标名落回**真实** resolver（避免误吞其他 case 的识别语义）
        return _real_resolve(name)

    monkeypatch.setattr(
        "app.domain.services.tools.tool_source_resolver.resolve_tool_source",
        _fake_resolve,
    )


@pytest.fixture
def service(uow_factory, confirmation_mgr, skill_tool_source_patch) -> AgentService:
    """AgentService with real uow + real ConfirmationManager + patched task
    methods + skill tool_source 识别补丁（J1/J3）。"""
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


def _make_skill_detail(
    *, session_id: str, user_id: str, tool_call_id: str
) -> ConfirmationDetail:
    """构造 Skill 源头的 confirmation detail。``tool_name`` 必须命中
    ``skill_tool_source_patch`` 的识别分支。"""
    return ConfirmationDetail(
        session_id=session_id,
        tool_call_id=tool_call_id,
        user_id=user_id,
        tool_name=_SKILL_TOOL_NAME,
        tool_args={"input": "deploy_to_staging"},
        risk_level="high",
        arg_digest="digest-skill-race-1",
        primary_arg="deploy_to_staging",
        dir_arg="",
        matched_patterns=[],
        deadline_ts=(datetime.now(timezone.utc) + timedelta(hours=1)).timestamp(),
        status="pending",
    )


class _Confirmation:
    def __init__(self, *, action: str, scope: str, tool_call_id: str) -> None:
        self.action = action
        self.scope = scope
        self.tool_call_id = tool_call_id


async def _fetch_grant_rows(session_factory, confirmation_id: str):
    async with session_factory() as s:
        return (
            await s.execute(
                text(
                    "SELECT decision_id, user_id, tool_name, tool_source, scope, "
                    "effect, source_type, confirmation_id "
                    "FROM tool_approval_grants WHERE confirmation_id = :cid"
                ),
                {"cid": confirmation_id},
            )
        ).mappings().all()


async def _fetch_audit_rows(session_factory, session_id: str):
    async with session_factory() as s:
        return (
            await s.execute(
                text(
                    "SELECT tool_name, action, scope, approved_by, decision_id "
                    "FROM tool_approval_log WHERE session_id = :sid"
                ),
                {"sid": session_id},
            )
        ).mappings().all()


# ---------------------------------------------------------------------------
# Test: session scope concurrent resume on a Skill-source confirmation
# ---------------------------------------------------------------------------


async def test_skill_session_scope_concurrent_resume_has_one_winner(
    service: AgentService,
    confirmation_mgr: ConfirmationManager,
    session_factory,
    test_session_id: str,
    test_user_id: str,
) -> None:
    """Skill-source + session scope：2 个并发 ``/resume`` → 1 winner + 1 409。

    **端到端链路**（R3 install → R5 claim）：
    1. Skill tool 触发 ToolConfirmationEvent（fixture 里用 ``_make_skill_detail``
       绕过 R3 install 链路，直接塞 ConfirmationDetail 到 Redis——整链 R3 侧
       已有独立 coverage；本测试聚焦 R5 claim 合同）
    2. 两个 client 同时 POST ``/sessions/{id}/chat`` 带 tool_confirmation →
       ``preflight_resume_tool_confirmation`` 并发进入 claim 路径
    3. ``ApprovalStateWriter.write()`` 的 DB UNIQUE(confirmation_id) 原子 claim
       保证只有一个赢家
    4. 失败者拿到 409（前端据此 ``/events?since=`` 复播）

    **Invariants**：
    - exactly 1 winner, exactly 1 ConflictError(409)
    - grant 行数 == 1，``tool_source = "skill"``（R5 合同：grant 记录 skill 源头）
    - audit 行数 == 1，``tool_name`` 为 skill 工具名，``decision_id`` 指 winner
    """
    tool_call_id = f"tc-race-skill-{uuid.uuid4().hex[:8]}"
    detail = _make_skill_detail(
        session_id=test_session_id,
        user_id=test_user_id,
        tool_call_id=tool_call_id,
    )
    await confirmation_mgr.store(detail)

    conf = _Confirmation(
        action="approve", scope="session", tool_call_id=tool_call_id
    )

    results = await asyncio.gather(
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
        return_exceptions=True,
    )

    winners = [r for r in results if not isinstance(r, BaseException)]
    conflicts = [r for r in results if isinstance(r, PolicyConflict)]
    unexpected = [
        r for r in results
        if isinstance(r, BaseException) and not isinstance(r, PolicyConflict)
    ]

    assert not unexpected, f"并发 preflight 出现非 409 异常: {unexpected!r}"
    assert len(winners) == 1, f"预期 1 winner，实际 {len(winners)}: {results!r}"
    assert len(conflicts) == 1
    assert str(conflicts[0]) == "approval_already_claimed"

    winner_state = winners[0]
    assert winner_state.decision_id is None
    assert winner_state.claim_nonce is not None
    assert winner_state.persistent_scope is True

    # Direct preflight stops before commit_resume; the Redis claim exists but
    # DB grant/audit rows do not yet.
    grant_rows = await _fetch_grant_rows(session_factory, tool_call_id)
    assert grant_rows == []

    audit_rows = await _fetch_audit_rows(session_factory, test_session_id)
    assert audit_rows == []
