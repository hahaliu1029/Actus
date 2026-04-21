"""Integration tests for DBUserToolApprovalPolicyRepository."""

import asyncio
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.models.user_tool_approval_policy import (
    ApprovalPolicy,
    UserToolApprovalPolicy,
)
from app.infrastructure.repositories.db_user_tool_approval_policy_repository import (
    DBUserToolApprovalPolicyRepository,
)


pytestmark = [pytest.mark.anyio, pytest.mark.integration]


@pytest.fixture
async def _user_row(db_session):
    """创建一条 users 行供 FK 引用。"""
    user_id = "policy-repo-test-user"
    await db_session.execute(text(
        "INSERT INTO users (id, username, email, password_hash, status, created_at, updated_at) "
        "VALUES (:id, :u, :e, 'x', 'active', NOW(), NOW())"
    ), {"id": user_id, "u": f"u_{user_id}", "e": f"{user_id}@t"})
    await db_session.flush()
    yield user_id


class TestUpsertAndGet:
    async def test_get_returns_none_when_absent(self, db_session):
        repo = DBUserToolApprovalPolicyRepository(db_session)
        result = await repo.get("unknown-user", "shell_execute")
        assert result is None

    async def test_upsert_creates_new_row(self, db_session, _user_row):
        repo = DBUserToolApprovalPolicyRepository(db_session)
        policy = UserToolApprovalPolicy(
            user_id=_user_row,
            tool_name="shell_execute",
            policy=ApprovalPolicy.ASK,
        )
        result = await repo.upsert(policy)
        assert result.policy == ApprovalPolicy.ASK
        fetched = await repo.get(_user_row, "shell_execute")
        assert fetched is not None
        assert fetched.policy == ApprovalPolicy.ASK

    async def test_upsert_updates_existing_row(self, db_session, _user_row):
        repo = DBUserToolApprovalPolicyRepository(db_session)
        await repo.upsert(UserToolApprovalPolicy(
            user_id=_user_row, tool_name="shell_execute", policy=ApprovalPolicy.ASK,
        ))
        await repo.upsert(UserToolApprovalPolicy(
            user_id=_user_row, tool_name="shell_execute", policy=ApprovalPolicy.AUTO,
        ))
        fetched = await repo.get(_user_row, "shell_execute")
        assert fetched.policy == ApprovalPolicy.AUTO
        count = await db_session.execute(text(
            "SELECT COUNT(*) FROM user_tool_approval_policies "
            "WHERE user_id=:u AND tool_name=:t"
        ), {"u": _user_row, "t": "shell_execute"})
        assert count.scalar() == 1


class TestListByUser:
    async def test_list_empty(self, db_session, _user_row):
        repo = DBUserToolApprovalPolicyRepository(db_session)
        assert await repo.list_by_user(_user_row) == []

    async def test_list_returns_all_rows_for_user(self, db_session, _user_row):
        repo = DBUserToolApprovalPolicyRepository(db_session)
        for tool, pol in [
            ("shell_execute", ApprovalPolicy.ASK),
            ("mcp_github_create_issue", ApprovalPolicy.AUTO),
            ("brainstorm_skill", ApprovalPolicy.DENY),
        ]:
            await repo.upsert(UserToolApprovalPolicy(
                user_id=_user_row, tool_name=tool, policy=pol,
            ))
        rows = await repo.list_by_user(_user_row)
        names = {r.tool_name for r in rows}
        assert names == {"shell_execute", "mcp_github_create_issue", "brainstorm_skill"}


class TestDelete:
    async def test_delete_returns_false_when_absent(self, db_session, _user_row):
        repo = DBUserToolApprovalPolicyRepository(db_session)
        assert await repo.delete(_user_row, "nonexistent_tool") is False

    async def test_delete_returns_true_when_present(self, db_session, _user_row):
        repo = DBUserToolApprovalPolicyRepository(db_session)
        await repo.upsert(UserToolApprovalPolicy(
            user_id=_user_row, tool_name="shell_execute", policy=ApprovalPolicy.ASK,
        ))
        assert await repo.delete(_user_row, "shell_execute") is True
        assert await repo.get(_user_row, "shell_execute") is None


class TestFkCascade:
    async def test_fk_cascade_on_user_delete(self, db_session):
        """Delete user → policies auto-removed via FK CASCADE."""
        user_id = "cascade-test-user"
        await db_session.execute(text(
            "INSERT INTO users (id, username, email, password_hash, status, created_at, updated_at) "
            "VALUES (:id, :u, :e, 'x', 'active', NOW(), NOW())"
        ), {"id": user_id, "u": f"u_{user_id}", "e": f"{user_id}@t"})
        repo = DBUserToolApprovalPolicyRepository(db_session)
        await repo.upsert(UserToolApprovalPolicy(
            user_id=user_id, tool_name="shell_execute", policy=ApprovalPolicy.ASK,
        ))
        await db_session.execute(text("DELETE FROM users WHERE id=:id"), {"id": user_id})
        assert await repo.get(user_id, "shell_execute") is None


class TestConcurrentUpsert:
    """Pin the repository-contract guarantee that ``upsert`` is atomic.

    The naive SELECT-then-INSERT pattern would let two concurrent PUTs to
    the same (user_id, tool_name) both miss, both INSERT, and one hit a
    UNIQUE(user_id, tool_name) IntegrityError. This test races two
    independent sessions so a regression away from ON CONFLICT DO UPDATE
    fails loudly here rather than surfacing as a flaky 500 under load.
    """

    async def test_concurrent_upserts_same_key_do_not_raise(
        self, async_engine
    ):
        user_id = f"concurrent-upsert-{uuid.uuid4()}"
        tool_name = "shell_execute"
        session_factory = async_sessionmaker(
            async_engine, class_=AsyncSession, expire_on_commit=False,
        )

        # Seed the parent users row in a committed session — the racing
        # upsert sessions won't see it otherwise (FK would fail).
        async with session_factory() as setup:
            async with setup.begin():
                await setup.execute(text(
                    "INSERT INTO users (id, username, email, password_hash, status, created_at, updated_at) "
                    "VALUES (:id, :u, :e, 'x', 'active', NOW(), NOW())"
                ), {"id": user_id, "u": f"u_{user_id}", "e": f"{user_id}@t"})

        async def _upsert_in_own_session(policy: ApprovalPolicy) -> None:
            async with session_factory() as sess:
                async with sess.begin():
                    repo = DBUserToolApprovalPolicyRepository(sess)
                    await repo.upsert(UserToolApprovalPolicy(
                        user_id=user_id, tool_name=tool_name, policy=policy,
                    ))

        try:
            # asyncio.gather interleaves the two coroutines on the event
            # loop — each has its own pooled connection + transaction.
            # If upsert were SELECT-then-INSERT, one branch would raise
            # IntegrityError here.
            await asyncio.gather(
                _upsert_in_own_session(ApprovalPolicy.ASK),
                _upsert_in_own_session(ApprovalPolicy.AUTO),
            )

            async with session_factory() as verify:
                count = await verify.execute(text(
                    "SELECT COUNT(*) FROM user_tool_approval_policies "
                    "WHERE user_id=:u AND tool_name=:t"
                ), {"u": user_id, "t": tool_name})
                assert count.scalar() == 1

                final = await verify.execute(text(
                    "SELECT policy FROM user_tool_approval_policies "
                    "WHERE user_id=:u AND tool_name=:t"
                ), {"u": user_id, "t": tool_name})
                # Last-writer-wins; either coroutine may have committed
                # last, so accept both valid values.
                assert final.scalar() in {"ask", "auto"}
        finally:
            # FK CASCADE removes the policy row when the user is deleted.
            async with session_factory() as cleanup:
                async with cleanup.begin():
                    await cleanup.execute(
                        text("DELETE FROM users WHERE id=:id"), {"id": user_id}
                    )
