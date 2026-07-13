"""D1a §3.6 原子路径 — DbExtensionAdmissionPort 集成测试（需 PostgreSQL）。

autouse `_migrate`（tests/integration/conftest.py）跑 alembic upgrade head →
建 extensions / plugin_memberships / extension_audit_log 表。

port 自管事务（`_record_observation` 内 `session.begin()` 提交），因此本文件
**不用** auto-rollback 的 `db_session` fixture——setup 行显式 COMMIT，断言用独立
session 重读，用例间靠**唯一 ext_id** 隔离（无跨用例污染）。

**集成未本地跑，CI 验证。**

Run:
    cd api && uv run pytest tests/integration/governance/test_db_extension_admission.py -v
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.external.extension_admission import Observation
from app.domain.models.extension_governance import HASH_SCHEMA_VERSION
from app.infrastructure.external.governance.db_extension_admission import (
    DbExtensionAdmissionPort,
)
from app.infrastructure.models.extension_governance import (
    ExtensionAuditLogModel,
    ExtensionModel,
    PluginMembershipModel,
)

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


@pytest.fixture
def factory(async_engine) -> async_sessionmaker[AsyncSession]:
    """自建 async_sessionmaker（brief Step 6）——port + setup/断言共用同一 engine。"""
    return async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)


def _eid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _tools(name: str = "t1") -> list[dict[str, Any]]:
    return [{"name": name, "description": "d", "input_schema": {"type": "object"}}]


async def _insert_ext(
    factory: async_sessionmaker[AsyncSession], *, kind: str, ext_id: str, **cols: Any,
) -> ExtensionModel:
    row = ExtensionModel(
        kind=kind,
        ext_id=ext_id,
        status=cols.pop("status", "active"),
        trust_origin=cols.pop("trust_origin", "user_installed"),
        source_type=cols.pop("source_type", "config"),
        hash_schema_version=cols.pop("hash_schema_version", HASH_SCHEMA_VERSION),
        **cols,
    )
    async with factory() as session:
        async with session.begin():
            session.add(row)
    return row


async def _fetch(
    factory: async_sessionmaker[AsyncSession], *, kind: str, ext_id: str,
) -> ExtensionModel:
    async with factory() as session:
        result = await session.execute(
            select(ExtensionModel).where(
                ExtensionModel.kind == kind, ExtensionModel.ext_id == ext_id))
        return result.scalar_one()


async def _audit_count(
    factory: async_sessionmaker[AsyncSession], *, ext_id: str, event: str,
) -> int:
    async with factory() as session:
        result = await session.execute(
            select(func.count()).select_from(ExtensionAuditLogModel).where(
                ExtensionAuditLogModel.ext_id == ext_id,
                ExtensionAuditLogModel.event == event))
        return int(result.scalar_one())


async def _audit_rows(
    factory: async_sessionmaker[AsyncSession], *, ext_id: str,
) -> list[ExtensionAuditLogModel]:
    async with factory() as session:
        result = await session.execute(
            select(ExtensionAuditLogModel).where(ExtensionAuditLogModel.ext_id == ext_id))
        return list(result.scalars().all())


def _recent() -> datetime:
    return datetime.now(timezone.utc) - timedelta(seconds=10)


# 1 ---------------------------------------------------------------------------
async def test_unknown_row(factory):
    ext_id = _eid("mcp-unknown")
    enforce = DbExtensionAdmissionPort(factory, mode="enforce")
    shadow = DbExtensionAdmissionPort(factory, mode="shadow")
    out_e = await enforce.check_many("mcp", [ext_id])
    out_s = await shadow.check_many("mcp", [ext_id])
    assert out_e[ext_id].reason == "unknown" and out_e[ext_id].admitted is False
    assert out_s[ext_id].reason == "unknown" and out_s[ext_id].admitted is True


# 2 ---------------------------------------------------------------------------
async def test_first_observation_persists_and_audits(factory):
    ext_id = _eid("skill-first")
    await _insert_ext(factory, kind="skill", ext_id=ext_id, artifact_hash=None)
    port = DbExtensionAdmissionPort(factory, mode="shadow")
    obs = Observation(category="artifact", payload="h1", schema_version=HASH_SCHEMA_VERSION)
    decision = await port.verify_observation("skill", ext_id, obs)
    row = await _fetch(factory, kind="skill", ext_id=ext_id)
    assert row.observed_artifact_hash == "h1"
    assert row.observed_hash_schema_version == HASH_SCHEMA_VERSION
    assert row.row_revision == 1
    assert await _audit_count(factory, ext_id=ext_id, event="observed_first") == 1
    assert decision.observation_outcome == "persisted"


# 3 ---------------------------------------------------------------------------
async def test_steady_state_window_zero_write(factory):
    ext_id = _eid("skill-steady")
    seeded_at = _recent()
    await _insert_ext(
        factory, kind="skill", ext_id=ext_id, artifact_hash="h1",
        observed_artifact_hash="h1", observed_hash_schema_version=HASH_SCHEMA_VERSION,
        last_observed_at=seeded_at, row_revision=5)
    port = DbExtensionAdmissionPort(factory, mode="shadow")  # window default 3600s
    obs = Observation(category="artifact", payload="h1", schema_version=HASH_SCHEMA_VERSION)
    decision = await port.verify_observation("skill", ext_id, obs)
    row = await _fetch(factory, kind="skill", ext_id=ext_id)
    assert row.row_revision == 5                       # 零 bump
    assert row.last_observed_at == seeded_at           # 未刷时间戳
    assert await _audit_count(factory, ext_id=ext_id, event="observed_first") == 0
    assert decision.observation_outcome == "unchanged"


# 4 ---------------------------------------------------------------------------
async def test_shadow_mismatch_dedup(factory):
    ext_id = _eid("skill-shadowmm")
    await _insert_ext(
        factory, kind="skill", ext_id=ext_id, artifact_hash="h0",
        observed_artifact_hash="h0", observed_hash_schema_version=HASH_SCHEMA_VERSION,
        last_observed_at=_recent(), row_revision=1)
    port = DbExtensionAdmissionPort(factory, mode="shadow")
    obs = Observation(category="artifact", payload="h1", schema_version=HASH_SCHEMA_VERSION)
    await port.verify_observation("skill", ext_id, obs)
    row = await _fetch(factory, kind="skill", ext_id=ext_id)
    assert row.observed_artifact_hash == "h1" and row.status == "active"
    assert row.row_revision == 2
    assert await _audit_count(factory, ext_id=ext_id, event="pin_mismatch") == 1
    # 再观测同值 → 稳态失配 dedup（R40#1）：零新 audit、零 revision 增长
    await port.verify_observation("skill", ext_id, obs)
    row2 = await _fetch(factory, kind="skill", ext_id=ext_id)
    assert row2.row_revision == 2
    assert await _audit_count(factory, ext_id=ext_id, event="pin_mismatch") == 1


# 5 ---------------------------------------------------------------------------
async def test_enforce_mismatch_quarantines_active(factory):
    ext_id = _eid("skill-enfmm")
    await _insert_ext(
        factory, kind="skill", ext_id=ext_id, artifact_hash="h0",
        observed_artifact_hash="h0", observed_hash_schema_version=HASH_SCHEMA_VERSION,
        last_observed_at=_recent(), row_revision=1)
    port = DbExtensionAdmissionPort(factory, mode="enforce")
    obs = Observation(category="artifact", payload="h1", schema_version=HASH_SCHEMA_VERSION)
    decision = await port.verify_observation("skill", ext_id, obs)
    row = await _fetch(factory, kind="skill", ext_id=ext_id)
    assert row.status == "quarantined" and row.quarantine_reason == "pin_mismatch"
    assert row.last_mismatch_at is not None and row.observed_artifact_hash == "h1"
    assert decision.reason == "quarantined" and decision.admitted is False
    assert await _audit_count(factory, ext_id=ext_id, event="pin_mismatch") == 1
    assert await _audit_count(factory, ext_id=ext_id, event="quarantined") == 1
    corrs = {r.event: r.correlation_id for r in await _audit_rows(factory, ext_id=ext_id)
             if r.event in ("pin_mismatch", "quarantined")}
    assert corrs["pin_mismatch"] is not None
    assert corrs["pin_mismatch"] == corrs["quarantined"]     # 共享 correlation_id


# 6 ---------------------------------------------------------------------------
async def test_enforce_mismatch_disabled_row_keeps_status(factory):
    ext_id = _eid("skill-disabled")
    await _insert_ext(
        factory, kind="skill", ext_id=ext_id, status="disabled", artifact_hash="h0",
        observed_artifact_hash="h0", observed_hash_schema_version=HASH_SCHEMA_VERSION,
        last_observed_at=_recent(), row_revision=1)
    port = DbExtensionAdmissionPort(factory, mode="enforce")
    obs = Observation(category="artifact", payload="h1", schema_version=HASH_SCHEMA_VERSION)
    decision = await port.verify_observation("skill", ext_id, obs)
    row = await _fetch(factory, kind="skill", ext_id=ext_id)
    assert row.status == "disabled" and row.observed_artifact_hash == "h1"
    assert decision.reason == "disabled" and decision.admitted is False
    assert await _audit_count(factory, ext_id=ext_id, event="quarantined") == 0


# 7 ---------------------------------------------------------------------------
async def test_version_gate_clears_and_rewrites(factory):
    ext_id = _eid("skill-vergate")
    await _insert_ext(
        factory, kind="skill", ext_id=ext_id, artifact_hash="a",
        observed_surface_hash="s", observed_artifact_hash="a", observed_config_fingerprint="c",
        observed_hash_schema_version=0, last_observed_at=_recent(), row_revision=2)
    port = DbExtensionAdmissionPort(factory, mode="shadow")
    # digest 未变（="a"）且窗口内 → 仍强制持久化（条件④ 版本门）
    obs = Observation(category="artifact", payload="a", schema_version=HASH_SCHEMA_VERSION)
    await port.verify_observation("skill", ext_id, obs)
    row = await _fetch(factory, kind="skill", ext_id=ext_id)
    assert row.observed_surface_hash is None and row.observed_config_fingerprint is None
    assert row.observed_artifact_hash == "a"
    assert row.observed_hash_schema_version == HASH_SCHEMA_VERSION
    assert row.row_revision == 3
    assert await _audit_count(factory, ext_id=ext_id, event="observed_first") == 1


# 8 ---------------------------------------------------------------------------
async def test_surface_persistence_gate_early_return(factory):
    ext_id = _eid("mcp-surfgate")
    await _insert_ext(
        factory, kind="mcp", ext_id=ext_id, surface_hash="SF",
        observed_config_fingerprint="C1", last_observed_at=_recent(), row_revision=3)
    port = DbExtensionAdmissionPort(factory, mode="enforce")
    obs = Observation(category="surface", payload=_tools(), schema_version=HASH_SCHEMA_VERSION,
                      under_config_fingerprint="C0")
    decision = await port.verify_observation("mcp", ext_id, obs)
    row = await _fetch(factory, kind="mcp", ext_id=ext_id)
    assert row.observed_surface_hash is None and row.row_revision == 3    # 零写
    assert decision.observation_outcome == "conflict" and decision.reason == "config_drift"
    assert len(await _audit_rows(factory, ext_id=ext_id)) == 0            # 零 audit


# 9 ---------------------------------------------------------------------------
async def test_config_drift_branch_never_quarantines(factory):
    # 可比 pin 行（config_fingerprint=F0，pin 版本=当前）
    ext_a = _eid("mcp-drift")
    await _insert_ext(
        factory, kind="mcp", ext_id=ext_a, config_fingerprint="F0",
        surface_hash="SF", observed_surface_hash="SF", observed_config_fingerprint="F0",
        observed_hash_schema_version=HASH_SCHEMA_VERSION, last_observed_at=_recent(),
        row_revision=1)
    port = DbExtensionAdmissionPort(factory, mode="enforce")
    out = await port.check_many("mcp", [ext_a], {ext_a: "F1"})
    row_a = await _fetch(factory, kind="mcp", ext_id=ext_a)
    assert out[ext_a].reason == "config_drift" and out[ext_a].config_drift_detected is True
    assert row_a.status == "active" and row_a.last_mismatch_at is None
    assert await _audit_count(factory, ext_id=ext_a, event="config_drift_detected") == 1
    assert await _audit_count(factory, ext_id=ext_a, event="quarantined") == 0
    assert await _audit_count(factory, ext_id=ext_a, event="pin_mismatch") == 0

    # unpinned 行同路 → 字段 False 零 drift audit（R48#4）
    ext_b = _eid("mcp-unpinned")
    await _insert_ext(factory, kind="mcp", ext_id=ext_b, config_fingerprint=None)
    out_b = await port.check_many("mcp", [ext_b], {ext_b: "F1"})
    assert out_b[ext_b].config_drift_detected is False
    assert await _audit_count(factory, ext_id=ext_b, event="config_drift_detected") == 0


# 10 --------------------------------------------------------------------------
async def test_config_change_invalidates_stale_surface(factory):
    ext_id = _eid("mcp-invsurf")
    await _insert_ext(
        factory, kind="mcp", ext_id=ext_id,
        observed_surface_hash="S0", observed_config_fingerprint="C0",
        observed_hash_schema_version=HASH_SCHEMA_VERSION, last_observed_at=_recent(),
        row_revision=1)
    port = DbExtensionAdmissionPort(factory, mode="shadow")
    obs = Observation(category="config_fingerprint", payload="C1", schema_version=HASH_SCHEMA_VERSION)
    await port.verify_observation("mcp", ext_id, obs)
    row = await _fetch(factory, kind="mcp", ext_id=ext_id)
    assert row.observed_surface_hash is None            # 同事务失效陈旧 surface
    assert row.observed_config_fingerprint == "C1"


# 11 --------------------------------------------------------------------------
async def test_cas_loser_rereads_no_quarantine(factory, monkeypatch):
    ext_id = _eid("skill-cas")
    await _insert_ext(
        factory, kind="skill", ext_id=ext_id, artifact_hash="h0",
        observed_artifact_hash="h0", observed_hash_schema_version=HASH_SCHEMA_VERSION,
        last_observed_at=_recent(), row_revision=5)
    port = DbExtensionAdmissionPort(factory, mode="enforce")

    real_load = port._load_row
    calls = {"n": 0}

    async def _stale_first(session, kind, ext):
        row = await real_load(session, kind, ext)
        calls["n"] += 1
        if calls["n"] == 1 and row is not None:
            return dict(row, row_revision=row["row_revision"] - 1)   # 过期快照 → CAS 落败
        return row

    monkeypatch.setattr(port, "_load_row", _stale_first)
    obs = Observation(category="artifact", payload="h1", schema_version=HASH_SCHEMA_VERSION)
    decision = await port.verify_observation("skill", ext_id, obs)
    row = await _fetch(factory, kind="skill", ext_id=ext_id)
    assert decision.observation_outcome == "conflict"
    assert row.status == "active" and row.observed_artifact_hash == "h0"   # 未写
    assert await _audit_count(factory, ext_id=ext_id, event="quarantined") == 0
    # F1：CAS 输家对 fresh 行（pin=h0，本次 digest=h1 仍失配）重算**完整判定**——
    # enforce 下 admitted=False / reason=pin_mismatch（非旧实现的 pin_presence-only fail-open）；
    # 且输家零写、零 pin_mismatch audit（迁移事件只由 quarantine 赢家写）。
    assert decision.reason == "pin_mismatch" and decision.admitted is False
    assert await _audit_count(factory, ext_id=ext_id, event="pin_mismatch") == 0


# 12 --------------------------------------------------------------------------
async def test_parent_blocked_via_membership(factory):
    parent_id = _eid("plugin-parent")
    child_id = _eid("mcp-child")
    parent = await _insert_ext(
        factory, kind="plugin", ext_id=parent_id, status="disabled", artifact_hash="pa")
    child = await _insert_ext(factory, kind="mcp", ext_id=child_id)
    async with factory() as session:
        async with session.begin():
            session.add(PluginMembershipModel(
                plugin_id=parent.id, child_extension_id=child.id,
                declared_component_id="c1"))
    shadow = DbExtensionAdmissionPort(factory, mode="shadow")
    out = await shadow.check_many("mcp", [child_id])
    # 行政类：shadow 也不准入（INV-D1-8）
    assert out[child_id].reason == "parent_blocked" and out[child_id].admitted is False


# 13 --------------------------------------------------------------------------
async def test_unpinned_first_observation_leaves_last_verified_null(factory):
    """F8：unpinned 扩展首次观测落库但**不打 last_verified_at**（无 pin 无「verified」语义，
    §5.2 写矩阵仅匹配的 pinned 比对打戳）。"""
    ext_id = _eid("skill-unpinnedobs")
    await _insert_ext(factory, kind="skill", ext_id=ext_id, artifact_hash=None)  # unpinned
    port = DbExtensionAdmissionPort(factory, mode="shadow")
    obs = Observation(category="artifact", payload="h1", schema_version=HASH_SCHEMA_VERSION)
    decision = await port.verify_observation("skill", ext_id, obs)
    row = await _fetch(factory, kind="skill", ext_id=ext_id)
    assert row.observed_artifact_hash == "h1"          # 观测落库
    assert row.last_verified_at is None                # F8：unpinned 不打 verified 戳
    assert row.last_mismatch_at is None
    assert decision.reason == "unpinned" and decision.observation_outcome == "persisted"
