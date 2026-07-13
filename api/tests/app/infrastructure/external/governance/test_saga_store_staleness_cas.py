"""R2b#1 saga-store 单测（fake session——无需 PG）：多 pod 崩溃恢复的终态 CAS + staleness。

- (a) compensate 终态 CAS 前置守卫：CAS miss（op 已被并发 complete）→ **不 teardown 不覆盖**。
- (a) fail UPDATE 携 `state='in_progress'` 守卫（不把 completed/compensated 覆盖成 failed）。
- (b) load_inflight_operations SELECT WHERE 携 `updated_at < cutoff` staleness 门（不采纳活 pod
  正在运行的新鲜 op）。

DB 时间/CAS 语义的端到端保真归 CI-only integration
（tests/integration/governance/test_plugin_saga_recovery_db.py）；本文件锁控制流与 WHERE 结构。
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from app.infrastructure.external.governance.plugin_saga_store import PluginSagaStore


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _ACM:
    def __init__(self, value: Any = None) -> None:
        self._value = value

    async def __aenter__(self) -> Any:
        return self._value

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class _Result:
    def __init__(self, scalar: Any = None, scalars_list: list[Any] | None = None) -> None:
        self._scalar = scalar
        self._scalars_list = scalars_list

    def scalar_one_or_none(self) -> Any:
        return self._scalar

    def scalars(self) -> Any:
        return SimpleNamespace(all=lambda: list(self._scalars_list or []))


class _FakeSession:
    def __init__(self, *, get_values: list[Any] | None = None,
                 execute_results: list[Any] | None = None) -> None:
        self._get_values = list(get_values or [])
        self._results = list(execute_results or [])
        self.executed: list[Any] = []
        self.added: list[Any] = []

    def begin(self) -> _ACM:
        return _ACM(None)

    async def get(self, model: Any, ident: Any) -> Any:
        return self._get_values.pop(0) if self._get_values else None

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


def _where_ops(stmt: Any) -> list[tuple[str | None, str]]:
    out: list[tuple[str | None, str]] = []
    for crit in stmt._where_criteria:
        left = getattr(crit, "left", None)
        out.append((getattr(left, "key", None),
                    getattr(crit.operator, "__name__", str(crit.operator))))
    return out


def _fake_op() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(), plugin_extension_id=uuid.uuid4(), initiated_by="admin-1")


# ---------- (b) load_inflight_operations staleness WHERE ----------

@pytest.mark.anyio
async def test_load_inflight_where_carries_staleness_gate():
    """SELECT WHERE 必须携 `state=='in_progress'`（eq）AND `updated_at < cutoff`（lt）——
    否则多 pod closure 采纳活 pod 正在运行的新鲜 op 当孤儿。"""
    session = _FakeSession(execute_results=[_Result(scalars_list=[])])  # 零 op
    store = PluginSagaStore(_FakeFactory(session))
    result = await store.load_inflight_operations()
    assert result == []
    assert len(session.executed) == 1
    ops = _where_ops(session.executed[0])
    assert ("state", "eq") in ops             # state == 'in_progress'
    assert ("updated_at", "lt") in ops        # R2b#1b staleness 门


# ---------- (a) fail 终态 CAS WHERE ----------

@pytest.mark.anyio
async def test_fail_update_carries_in_progress_cas():
    """fail UPDATE WHERE 必须携 `state=='in_progress'` 守卫（除 id 外）——多 pod 下不把
    已 completed/compensated 的 op 覆盖成 failed。"""
    session = _FakeSession()
    store = PluginSagaStore(_FakeFactory(session))
    await store.fail(uuid.uuid4(), error="boom")
    assert len(session.executed) == 1
    ops = _where_ops(session.executed[0])
    assert ("id", "eq") in ops
    assert ("state", "eq") in ops             # R2b#1a 终态 CAS 守卫


# ---------- (a) compensate 终态 CAS 前置守卫 ----------

@pytest.mark.anyio
async def test_compensate_cas_miss_skips_teardown():
    """核心安全属性：op 已被并发 complete_install（CAS `WHERE state='in_progress'` miss）→
    compensate **不 teardown 不写 audit**（否则拆掉已成功安装 + completed→compensated 覆盖）。
    首个 execute=CAS UPDATE 返 None（miss）→ 早退，无后续 execute、无 audit add。"""
    session = _FakeSession(
        get_values=[_fake_op()],           # get(op) → 非 None
        execute_results=[_Result(None)])   # CAS UPDATE returning None = miss
    store = PluginSagaStore(_FakeFactory(session))
    await store.compensate(uuid.uuid4(), details={})
    assert len(session.executed) == 1      # 仅 CAS，无 teardown 执行
    assert session.added == []             # 无 audit（未 teardown）
    # 且首个 execute 就是携 state 守卫的 op-state UPDATE（CAS 先行）
    assert ("state", "eq") in _where_ops(session.executed[0])


@pytest.mark.anyio
async def test_compensate_cas_hit_proceeds_teardown():
    """CAS hit（op 仍 in_progress）→ 补偿照常 teardown（membership 删/父行软删）+ 写
    plugin_expand_compensated audit。锁「CAS 先行不阻断赢家」。"""
    plugin_row = SimpleNamespace(id=uuid.uuid4(), ext_id="org.test.pack", deleted_at=None)
    session = _FakeSession(
        get_values=[_fake_op(), plugin_row],
        execute_results=[
            _Result(uuid.uuid4()),            # CAS hit
            _Result(scalars_list=[]),         # member_ids select → 空
            _Result(None),                    # membership delete
            _Result(None),                    # 父行软删
        ])
    store = PluginSagaStore(_FakeFactory(session))
    await store.compensate(uuid.uuid4(), details={})
    assert len(session.executed) > 1          # teardown 进行
    events = [getattr(a, "event", None) for a in session.added]
    assert "plugin_expand_compensated" in events
