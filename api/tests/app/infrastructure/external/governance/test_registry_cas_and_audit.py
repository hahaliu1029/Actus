"""F4/F5 registry-port 单测（fake session——无需 PG）。

- F4：record_delete 必须带 row_revision CAS（读-CAS-重试 3 次，镜像 record_install）——
  UPDATE 带 WHERE row_revision 守卫；行消失/已删→幂等成功；耗尽→RevisionConflictError。
- F5：行政动作 audit 的 `before` 快照必须在 UPDATE **之前**捕获进纯局部——否则 ORM
  synchronize_session 已把 in-memory row 同步到新值，before==after。

集成侧对应 [CI-only] 用例在 tests/integration/governance/test_db_extension_registry.py。
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from app.domain.external.extension_admission import UninstallContext
from app.domain.models.extension_governance import HASH_SCHEMA_VERSION, RevisionConflictError
from app.application.errors.exceptions import BadRequestError
from app.infrastructure.external.governance import db_extension_registry as reg_mod
from app.infrastructure.external.governance.db_extension_registry import (
    DbExtensionRegistryReadPort,
    DbExtensionRegistryWritePort,
)


# ---------- fake session 基建 ----------

class _ACM:
    def __init__(self, value: Any = None) -> None:
        self._value = value

    async def __aenter__(self) -> Any:
        return self._value

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class _Result:
    def __init__(self, scalar: Any) -> None:
        self._scalar = scalar

    def scalar_one_or_none(self) -> Any:
        return self._scalar


class _FakeSession:
    def __init__(self, execute_results: list[Any]) -> None:
        self._results = list(execute_results)
        self.executed: list[Any] = []
        self.added: list[Any] = []

    def begin(self) -> _ACM:
        return _ACM(None)

    async def execute(self, stmt: Any) -> _Result:
        self.executed.append(stmt)
        return self._results.pop(0) if self._results else _Result(None)

    def add(self, obj: Any) -> None:
        self.added.append(obj)


class _FakeFactory:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    def __call__(self) -> _ACM:
        return _ACM(self._session)


def _where_cols(stmt: Any) -> set[str]:
    cols: set[str] = set()
    for crit in stmt._where_criteria:
        left = getattr(crit, "left", None)
        if left is not None and hasattr(left, "key"):
            cols.add(left.key)
    return cols


def _uctx(correlation_id: uuid.UUID | None = None) -> UninstallContext:
    return UninstallContext(correlation_id=correlation_id or uuid.uuid4(), actor_user_id="admin-1")


# ---------- F4：record_delete row_revision CAS ----------

@pytest.mark.anyio
async def test_record_delete_update_carries_revision_guard(monkeypatch):
    """审计 repro 防线：软删 UPDATE 必须带 WHERE row_revision 守卫（否则陈旧 delete 覆盖
    并发 reinstall）。CAS-miss 首次→重试→第二次成功。"""
    session = _FakeSession(execute_results=[_Result(None), _Result(9)])  # miss, hit
    port = DbExtensionRegistryWritePort(_FakeFactory(session))

    async def fake_load_live(sess, kind, ext_id):
        return SimpleNamespace(id=uuid.uuid4(), row_revision=7)

    monkeypatch.setattr(reg_mod, "_load_live", fake_load_live)
    await port.record_delete("skill", "e1", uninstall_context=_uctx())

    assert len(session.executed) == 2                 # 首次 CAS miss → 重试
    assert "row_revision" in _where_cols(session.executed[0])   # 守卫存在
    assert len([a for a in session.added]) == 1        # 成功后恰一条 uninstalled audit


@pytest.mark.anyio
async def test_record_delete_row_gone_mid_retry_idempotent(monkeypatch):
    """CAS miss 后重读发现行已消失（并发软删）→ 幂等成功返回，零异常零 audit。"""
    session = _FakeSession(execute_results=[_Result(None)])  # attempt1 CAS miss
    port = DbExtensionRegistryWritePort(_FakeFactory(session))
    calls = {"n": 0}

    async def fake_load_live(sess, kind, ext_id):
        calls["n"] += 1
        if calls["n"] == 1:
            return SimpleNamespace(id=uuid.uuid4(), row_revision=7)
        return None  # 并发已软删

    monkeypatch.setattr(reg_mod, "_load_live", fake_load_live)
    await port.record_delete("skill", "e1", uninstall_context=_uctx())  # 不抛
    assert session.added == []                          # 幂等零 audit


@pytest.mark.anyio
async def test_record_delete_cas_exhausted_raises(monkeypatch):
    """持久 CAS miss（行一直 live 但 rev 一直变）→ 3 次耗尽 → RevisionConflictError。"""
    session = _FakeSession(execute_results=[])  # 恒 None → 恒 CAS miss
    port = DbExtensionRegistryWritePort(_FakeFactory(session))

    async def fake_load_live(sess, kind, ext_id):
        return SimpleNamespace(id=uuid.uuid4(), row_revision=7)

    monkeypatch.setattr(reg_mod, "_load_live", fake_load_live)
    with pytest.raises(RevisionConflictError):
        await port.record_delete("skill", "e1", uninstall_context=_uctx())
    assert len(session.executed) == 3                  # 恰 3 次尝试


# ---------- F5：audit before 在 UPDATE 之前捕获 ----------

def _sync_cas(monkeypatch):
    """把 _cas_transition 换成模拟 ORM synchronize_session 的桩：UPDATE 后 in-memory row
    已同步到新值（复现 F5 时序——若 before 在 UPDATE 后读则 before==after）。"""
    async def fake_cas(session, row, expected_row_revision, **values):
        for key, value in values.items():
            setattr(row, key, value)
        row.row_revision = expected_row_revision + 1
        return expected_row_revision + 1

    monkeypatch.setattr(reg_mod, "_cas_transition", fake_cas)


@pytest.mark.anyio
async def test_quarantine_before_snapshot_is_pre_update(monkeypatch):
    """审计 repro：active→quarantined，audit before.status 必须是 'active'（转移前），
    不能因 ORM 同步读成 'quarantined'。"""
    session = _FakeSession(execute_results=[])
    port = DbExtensionRegistryWritePort(_FakeFactory(session))
    row = SimpleNamespace(id=uuid.uuid4(), status="active", row_revision=0)

    async def fake_load_live(sess, kind, ext_id):
        return row

    monkeypatch.setattr(reg_mod, "_load_live", fake_load_live)
    _sync_cas(monkeypatch)

    await port.quarantine("skill", "e1", expected_row_revision=0, actor_user_id="admin-1")
    audit = session.added[0]
    assert audit.before == {"status": "active"}         # 转移前快照
    assert audit.after["status"] == "quarantined"


@pytest.mark.anyio
async def test_disable_before_snapshot_is_pre_update(monkeypatch):
    """active→disabled，audit before.status 必须是 'active'。"""
    session = _FakeSession(execute_results=[])
    port = DbExtensionRegistryWritePort(_FakeFactory(session))
    row = SimpleNamespace(id=uuid.uuid4(), status="active", row_revision=3)

    async def fake_load_live(sess, kind, ext_id):
        return row

    monkeypatch.setattr(reg_mod, "_load_live", fake_load_live)
    _sync_cas(monkeypatch)

    await port.set_governance_enabled(
        "skill", "e1", enabled=False, expected_row_revision=3, actor_user_id="admin-1")
    audit = session.added[0]
    assert audit.before == {"status": "active"}
    assert audit.after["status"] == "disabled"


@pytest.mark.anyio
async def test_enable_before_snapshot_is_pre_update(monkeypatch):
    """G5 补洞：disabled→active（set_governance_enabled enable path），audit before.status
    必须是 'disabled'（转移前），不能因 ORM 同步读成 'active'。"""
    session = _FakeSession(execute_results=[])
    port = DbExtensionRegistryWritePort(_FakeFactory(session))
    row = SimpleNamespace(id=uuid.uuid4(), status="disabled", row_revision=4)

    async def fake_load_live(sess, kind, ext_id):
        return row

    monkeypatch.setattr(reg_mod, "_load_live", fake_load_live)
    _sync_cas(monkeypatch)

    await port.set_governance_enabled(
        "skill", "e1", enabled=True, expected_row_revision=4, actor_user_id="admin-1")
    audit = session.added[0]
    assert audit.before == {"status": "disabled"}       # 转移前快照
    assert audit.after["status"] == "active"


@pytest.mark.anyio
async def test_reapprove_before_snapshot_is_pre_update(monkeypatch):
    """G5 补洞：quarantined→active（reapprove），audit before.status 必须是 'quarantined'
    （转移前），不能因 ORM 同步读成 'active'。"""
    session = _FakeSession(execute_results=[])
    port = DbExtensionRegistryWritePort(_FakeFactory(session))
    # reapprove 前置：全类别 observed 非 NULL 且版本=当前（skill 仅需 artifact）
    row = SimpleNamespace(
        id=uuid.uuid4(), status="quarantined", row_revision=2,
        observed_artifact_hash="h-art", observed_surface_hash=None,
        observed_config_fingerprint=None,
        observed_hash_schema_version=HASH_SCHEMA_VERSION)

    async def fake_load_live(sess, kind, ext_id):
        return row

    monkeypatch.setattr(reg_mod, "_load_live", fake_load_live)
    _sync_cas(monkeypatch)

    await port.reapprove("skill", "e1", expected_row_revision=2, actor_user_id="admin-1")
    audit = session.added[0]                             # 首条=reapproved（次条=pin_established）
    assert audit.before == {"status": "quarantined"}    # 转移前快照
    assert audit.after["status"] == "active"


# ---------- R2b#2：list_audit 畸形游标 → BadRequestError（400，非全局 500）----------

class TestListAuditCursorDecode:
    """``_decode_audit_cursor`` 对畸形 ``?cursor=`` 做 ``rsplit('|')`` + ``fromisoformat``
    + ``uuid.UUID``——任一失败抛裸 ``ValueError``（无 handler → 全局 500）。修为 ReadPort.
    ``list_audit`` 包裹 decode → ``BadRequestError``（映射 400）。decode 在 DB session
    **之前**执行（畸形游标零 DB 触碰），故 fake factory 永不被调用即可断言。"""

    @pytest.mark.anyio
    @pytest.mark.parametrize("bad_cursor", [
        "garbage",                       # 无 '|' → unpack 失败
        "not-a-date|not-a-uuid",         # fromisoformat 失败
        "2026-07-11T00:00:00+00:00|xyz", # uuid.UUID 失败
    ])
    async def test_invalid_cursor_raises_bad_request(self, bad_cursor):
        session = _FakeSession(execute_results=[])
        port = DbExtensionRegistryReadPort(_FakeFactory(session))
        with pytest.raises(BadRequestError):
            await port.list_audit(cursor=bad_cursor)
        assert session.executed == []    # 畸形游标零 DB 触碰（decode 早于 session）


# ---------- R2b#4：record_reconciled_missing 软删 WHERE 携时间宽限门 ----------

class TestReconciledMissingGraceGate:
    """两次确认软删的 UPDATE WHERE 必须除 `source_missing_at IS NOT NULL`（#4-R1 守卫）外
    再携 `source_missing_at < now - GRACE` 时间门——否则多 pod（A 标记→释锁→B 抢锁即删）
    宽限压缩为零经过时间。时间语义由 DB 求值（CI 集成锁），本单测锁 WHERE 结构存在两门。"""

    @pytest.mark.anyio
    async def test_soft_delete_where_carries_time_grace(self, monkeypatch):
        session = _FakeSession(execute_results=[_Result(None)])  # WHERE miss → no-op（宽限内）
        port = DbExtensionRegistryWritePort(_FakeFactory(session))
        row = SimpleNamespace(
            id=uuid.uuid4(), source_missing_at=datetime.now(timezone.utc), row_revision=1)

        async def fake_load_live(sess, kind, ext_id):
            return row

        monkeypatch.setattr(reg_mod, "_load_live", fake_load_live)
        await port.record_reconciled_missing("mcp", "e1")

        assert len(session.executed) == 1          # 仅 UPDATE（WHERE miss → 无 audit add）
        assert session.added == []
        upd = session.executed[0]
        ops = [
            getattr(crit.operator, "__name__", str(crit.operator))
            for crit in upd._where_criteria
            if getattr(getattr(crit, "left", None), "key", None) == "source_missing_at"
        ]
        assert "is_not" in ops                     # #4-R1 守卫保留
        assert "lt" in ops                         # R2b#4 时间宽限门新增
