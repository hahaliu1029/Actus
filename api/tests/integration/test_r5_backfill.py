"""R5 CS4 backfill CLI 集成测试（需要 Postgres）。

覆盖 test plan §5：
- ``test_backfill_completeness``：N 条 legacy rule → grants 表有 N 行
- ``test_backfill_idempotent``：重跑不翻倍（WHERE NOT EXISTS）
- ``test_backfill_preserves_created_at``：legacy 时间戳保留
- ``test_backfill_unknown_tool_fallback``：未知 tool_name → ``tool_source='native'`` + log warning

运行：

    cd api && SQLALCHEMY_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/manus_test \\
        uv run pytest tests/integration/test_r5_backfill.py -v
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select, text

from app.cli.backfill_approval_grants import run_backfill
from app.domain.services.approval_grant_policy import to_naive_utc
from app.infrastructure.models.tool_approval_grant import ToolApprovalGrantModel
from app.infrastructure.models.tool_approval_rule import ToolApprovalRuleModel

pytestmark = pytest.mark.anyio


async def _ensure_user(db_session, user_id: str) -> None:
    await db_session.execute(
        text("INSERT INTO users (id) VALUES (:uid) ON CONFLICT DO NOTHING"),
        {"uid": user_id},
    )


async def _insert_legacy_rule(
    db_session,
    *,
    user_id: str,
    tool_name: str = "shell_execute",
    rule: str = "always_allow",
    command_pattern: str = "ls *",
    dir_pattern: str = "",
    created_at: datetime | None = None,
) -> str:
    """插入一条 ``tool_approval_rules`` 行，返回 rule id。

    使用 ORM 而非 raw ``sa.text()``——raw text INSERT 下 asyncpg 对 naive datetime
    走 ``datetime.timestamp()`` 路径，把 naive 当**本地时区**解读为 UTC 存进
    ``TIMESTAMP WITHOUT TIME ZONE``，导致时间漂移（Codex round-4 HIGH 的根因在
    setup 而非 backfill）。ORM 路径让 SQLAlchemy + asyncpg 走 typed column 绑定，
    naive datetime 作为 wall-clock 直接存盘。
    """
    rule_id = str(uuid.uuid4())
    model = ToolApprovalRuleModel(
        id=rule_id,
        user_id=user_id,
        tool_name=tool_name,
        rule=rule,
        command_pattern=command_pattern,
        dir_pattern=dir_pattern,
    )
    if created_at is not None:
        # 归一成 naive UTC（aware → naive，naive 原样透传）
        model.created_at = to_naive_utc(created_at)
    db_session.add(model)
    await db_session.flush()
    return rule_id


async def _count_grants_for_user(db_session, user_id: str) -> int:
    stmt = select(ToolApprovalGrantModel).where(
        ToolApprovalGrantModel.user_id == user_id
    )
    rows = (await db_session.execute(stmt)).scalars().all()
    return len(rows)


# ---------------- completeness ----------------


async def test_backfill_completeness(db_session) -> None:
    """10 条 legacy rules → backfill 后 grants 表有 10 行对应。"""
    user_id = str(uuid.uuid4())
    await _ensure_user(db_session, user_id)

    # 插 10 条 legacy rules（混 allow + deny）
    for i in range(10):
        await _insert_legacy_rule(
            db_session,
            user_id=user_id,
            tool_name=f"tool_{i}",
            rule="always_deny" if i % 3 == 0 else "always_allow",
            command_pattern=f"cmd_{i} *",
            dir_pattern="" if i % 2 == 0 else f"/dir_{i}/*",
        )
    await db_session.flush()

    processed = await run_backfill(db_session, batch=100, dry_run=False, commit_each_batch=False)
    assert processed == 10

    count = await _count_grants_for_user(db_session, user_id)
    assert count == 10

    # 断言 effect 与 rule 映射：always_deny → deny，其余 approve
    stmt = select(ToolApprovalGrantModel).where(
        ToolApprovalGrantModel.user_id == user_id
    )
    grants = (await db_session.execute(stmt)).scalars().all()
    deny_count = sum(1 for g in grants if g.effect == "deny")
    approve_count = sum(1 for g in grants if g.effect == "approve")
    # i=0,3,6,9 是 deny → 4 条；其余 6 条 approve
    assert deny_count == 4
    assert approve_count == 6
    # 都是 always scope
    assert all(g.scope == "always" for g in grants)
    # 都无 session_id / 无 expires_at / 无 confirmation_id
    assert all(g.session_id is None for g in grants)
    assert all(g.expires_at is None for g in grants)
    assert all(g.confirmation_id is None for g in grants)


# ---------------- idempotency ----------------


async def test_backfill_idempotent(db_session) -> None:
    """重跑第二次行数不翻倍（``WHERE NOT EXISTS`` dedup）。"""
    user_id = str(uuid.uuid4())
    await _ensure_user(db_session, user_id)
    for i in range(3):
        await _insert_legacy_rule(
            db_session,
            user_id=user_id,
            tool_name=f"tool_{i}",
            command_pattern=f"cmd_{i} *",
        )
    await db_session.flush()

    first = await run_backfill(db_session, batch=100, dry_run=False, commit_each_batch=False)
    # 第一次：完整迁 3 条
    assert first == 3
    assert await _count_grants_for_user(db_session, user_id) == 3

    # 第二次跑：WHERE NOT EXISTS 应该全部跳过
    second = await run_backfill(db_session, batch=100, dry_run=False, commit_each_batch=False)
    assert second == 0
    assert await _count_grants_for_user(db_session, user_id) == 3


# ---------------- created_at 保留 ----------------


async def test_backfill_preserves_created_at(db_session) -> None:
    """legacy ``created_at`` 被透传到 grant ``created_at`` —— 墙钟必须完全一致。

    用 **aware UTC** 输入消除 naive-as-local 的解读歧义：
    - 测试输入：``2026-01-15 10:30:00+00:00``
    - 预期落盘：``2026-01-15 10:30:00``（naive UTC wall-clock）
    - 读回断言：``grant.created_at == 2026-01-15 10:30:00``

    若未来 DB 绑定路径回退到 naive-as-local 推断，测试会立即变红（锁死
    Codex round-4 HIGH 修复）。
    """
    user_id = str(uuid.uuid4())
    await _ensure_user(db_session, user_id)
    fixed_aware = datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    fixed_naive = fixed_aware.replace(tzinfo=None)  # DB 回读预期

    await _insert_legacy_rule(
        db_session,
        user_id=user_id,
        tool_name="shell_execute",
        command_pattern="ls *",
        created_at=fixed_aware,
    )
    await db_session.flush()

    # 立即读回 rules，确认 setup 没在 INSERT 侧漂移
    rule_ct = (
        await db_session.execute(
            text("SELECT created_at FROM tool_approval_rules WHERE user_id = :uid"),
            {"uid": user_id},
        )
    ).scalar_one()
    assert rule_ct == fixed_naive, (
        f"setup 写 rules 时已漂移：stored={rule_ct} expected={fixed_naive}"
    )

    await run_backfill(db_session, batch=100, dry_run=False, commit_each_batch=False)

    stmt = select(ToolApprovalGrantModel).where(
        ToolApprovalGrantModel.user_id == user_id
    )
    grant = (await db_session.execute(stmt)).scalar_one()
    assert grant.created_at == fixed_naive, (
        f"backfill 漂移：grant.created_at={grant.created_at} expected={fixed_naive}"
    )


# ---------------- tool_source fallback ----------------


async def test_backfill_unknown_tool_fallback_to_native(
    db_session, caplog: pytest.LogCaptureFixture
) -> None:
    """``tool_name`` 不在 registry → ``tool_source='native'`` + log warning。"""
    user_id = str(uuid.uuid4())
    await _ensure_user(db_session, user_id)
    unknown_name = "weird_tool_xyz_not_in_registry"
    await _insert_legacy_rule(
        db_session,
        user_id=user_id,
        tool_name=unknown_name,
        command_pattern="*",
    )
    await db_session.flush()

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="app.cli.backfill_approval_grants"):
        await run_backfill(db_session, batch=100, dry_run=False, commit_each_batch=False)

    stmt = select(ToolApprovalGrantModel).where(
        ToolApprovalGrantModel.user_id == user_id
    )
    grant = (await db_session.execute(stmt)).scalar_one()
    assert grant.tool_source == "native"
    # log 里应提到兜底事实
    assert any(
        unknown_name in rec.getMessage() and "native" in rec.getMessage()
        for rec in caplog.records
    ), f"expected warning about {unknown_name!r} fallback, got: {[r.getMessage() for r in caplog.records]}"


# ---------------- dry-run ----------------


async def test_backfill_dry_run_does_not_write(db_session) -> None:
    """``dry_run=True`` 只读不写，grants 表不新增行。"""
    user_id = str(uuid.uuid4())
    await _ensure_user(db_session, user_id)
    await _insert_legacy_rule(
        db_session, user_id=user_id, tool_name="shell_execute", command_pattern="ls *"
    )
    await db_session.flush()

    processed = await run_backfill(db_session, batch=100, dry_run=True)
    # dry-run 报告会扫到 1 条
    assert processed == 1
    # 但 grants 表仍然空
    assert await _count_grants_for_user(db_session, user_id) == 0
