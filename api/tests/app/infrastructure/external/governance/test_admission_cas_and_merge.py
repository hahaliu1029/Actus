"""F1/F6/F8 admission-port 单测（fake session——无需 PG）。

- F1：观测 CAS 输家必须重算**完整判定**（digest 与新 pin 比对），非仅 pin_presence/status；
- F6：check_many 双侧 admitted 合并按 reason 严重度（detection 胜 ok），不吞检测原因；
- F8：last_verified_at 仅在**匹配的 pinned 比对**时打戳（unpinned 首观测保持 NULL）。

集成侧对应 [CI-only] 用例在 tests/integration/governance/test_db_extension_admission.py。
"""
from __future__ import annotations

import uuid
from typing import Any

import pytest

from app.domain.external.extension_admission import AdmissionDecision, Observation
from app.infrastructure.external.governance.db_extension_admission import (
    DbExtensionAdmissionPort,
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


def _row(**cols: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(
        id=uuid.uuid4(), ext_id="e1", status="active", row_revision=0,
        hash_schema_version=1, artifact_hash=None, surface_hash=None,
        config_fingerprint=None, observed_surface_hash=None,
        observed_artifact_hash=None, observed_config_fingerprint=None,
        observed_hash_schema_version=None, last_observed_at=None,
        parent_blocked=False,
    )
    base.update(cols)
    return base


def _set_keys(stmt: Any) -> set[str]:
    return {c.key for c in stmt._values}


# ---------- F1：CAS 输家完整重判 ----------

@pytest.mark.anyio
async def test_cas_loser_mismatch_returns_pin_mismatch_no_quarantine(monkeypatch):
    """审计 repro：enforce 观测输掉 observed CAS，重读 fresh 行 pin 仍与本次 digest 失配 →
    必须 admitted=False / reason='pin_mismatch'（非旧代码的 'ok' fail-open），且输家不写 quarantine。"""
    session = _FakeSession(execute_results=[_Result(None)])  # observed UPDATE → CAS 落败
    port = DbExtensionAdmissionPort(_FakeFactory(session), mode="enforce")

    r0 = _row(row_revision=5, artifact_hash="h0", observed_artifact_hash="h0",
              observed_hash_schema_version=1)
    fresh = _row(row_revision=6, artifact_hash="h0", observed_artifact_hash="h0",
                 observed_hash_schema_version=1)     # 并发 approve 后 rev bump；pin 仍 h0
    seq = [r0, fresh]
    calls = {"n": 0}

    async def fake_load_row(sess, kind, ext_id):
        r = seq[calls["n"]]
        calls["n"] += 1
        return r

    monkeypatch.setattr(port, "_load_row", fake_load_row)
    obs = Observation(category="artifact", payload="h1", schema_version=1)
    decision = await port.verify_observation("skill", "e1", obs)

    assert decision.observation_outcome == "conflict"
    assert decision.admitted is False
    assert decision.reason == "pin_mismatch"
    assert decision.row_revision == 6                # fresh 行 revision
    assert len(session.executed) == 1               # 仅 observed UPDATE，无 quarantine UPDATE
    assert session.added == []                       # 输家零 audit


@pytest.mark.anyio
async def test_cas_loser_matched_pin_returns_ok(monkeypatch):
    """CAS 输家但本次 digest 与 fresh pin 一致 → reason='ok'（准入），仍 outcome='conflict'。"""
    session = _FakeSession(execute_results=[_Result(None)])
    port = DbExtensionAdmissionPort(_FakeFactory(session), mode="enforce")

    r0 = _row(row_revision=5, artifact_hash="h1", observed_artifact_hash="h0",
              observed_hash_schema_version=1)
    fresh = _row(row_revision=6, artifact_hash="h1", observed_artifact_hash="h1",
                 observed_hash_schema_version=1)     # 并发把 pin 转正为 h1
    seq = [r0, fresh]
    calls = {"n": 0}

    async def fake_load_row(sess, kind, ext_id):
        r = seq[calls["n"]]
        calls["n"] += 1
        return r

    monkeypatch.setattr(port, "_load_row", fake_load_row)
    obs = Observation(category="artifact", payload="h1", schema_version=1)
    decision = await port.verify_observation("skill", "e1", obs)

    assert decision.observation_outcome == "conflict"
    assert decision.reason == "ok" and decision.admitted is True
    assert decision.row_revision == 6


# ---------- F6：check_many 双侧 admitted 按严重度合并 ----------

@pytest.mark.anyio
async def test_check_many_shadow_detection_reason_wins_over_ok(monkeypatch):
    """审计 repro：config pin 匹配(ok) + surface pin 缺失(unpinned)，shadow 双侧 admitted →
    返回 reason 必须是检测原因 'unpinned'，不能被 config 侧 'ok' 吞掉。"""
    session = _FakeSession(execute_results=[])
    port = DbExtensionAdmissionPort(_FakeFactory(session), mode="shadow")

    async def fake_record_obs(kind, ext_id, *, category, digest,
                              under_config_fingerprint, obs_schema_version):
        return AdmissionDecision(admitted=True, reason="ok", row_revision=7,
                                 observation_outcome="unchanged")

    async def fake_load_rows(sess, kind, ext_ids):
        return {ext_ids[0]: _row(ext_id=ext_ids[0], status="active", row_revision=7,
                                 surface_hash=None, config_fingerprint="F0",
                                 observed_config_fingerprint="F0",
                                 observed_hash_schema_version=1)}

    monkeypatch.setattr(port, "_record_observation", fake_record_obs)
    monkeypatch.setattr(port, "_load_rows", fake_load_rows)

    out = await port.check_many("mcp", ["e1"], {"e1": "F0"})
    assert out["e1"].reason == "unpinned"
    assert out["e1"].admitted is True
    assert out["e1"].observation_outcome == "unchanged"  # 携带 config 观测 outcome


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["shadow", "enforce"])
async def test_check_many_administrative_obs_reason_wins_over_row_detection(monkeypatch, mode):
    """G4 补洞（F6 严重度合并的行政面）：config 观测侧返回行政原因（disabled——行政类两
    模式均不准入）+ 行状态侧检测原因（surface unpinned）→ 合并结果必须是**行政原因**、
    admitted=False，shadow 与 enforce 皆然（行政 > 检测严重度；行政在 shadow 也拒，
    不因 shadow 对 unpinned fail-open 而放行）。"""
    session = _FakeSession(execute_results=[])
    port = DbExtensionAdmissionPort(_FakeFactory(session), mode=mode)

    async def fake_record_obs(kind, ext_id, *, category, digest,
                              under_config_fingerprint, obs_schema_version):
        # config 观测侧：行政原因 disabled（ADMINISTRATIVE_REASONS，双 mode 不准入）
        return AdmissionDecision(admitted=False, reason="disabled", row_revision=9,
                                 observation_outcome="unchanged")

    async def fake_load_rows(sess, kind, ext_ids):
        # 行状态侧：active 但 surface 未 pin → status_decision.reason='unpinned'（检测类）
        return {ext_ids[0]: _row(ext_id=ext_ids[0], status="active", row_revision=9,
                                 surface_hash=None, config_fingerprint="F0",
                                 observed_config_fingerprint="F0",
                                 observed_hash_schema_version=1)}

    monkeypatch.setattr(port, "_record_observation", fake_record_obs)
    monkeypatch.setattr(port, "_load_rows", fake_load_rows)

    out = await port.check_many("mcp", ["e1"], {"e1": "F0"})
    assert out["e1"].reason == "disabled"       # 行政原因胜出（非被 unpinned/ok 吞）
    assert out["e1"].admitted is False           # 两模式均不准入


# ---------- F8：last_verified_at 仅匹配 pinned 打戳 ----------

@pytest.mark.anyio
async def test_unpinned_first_observation_does_not_stamp_last_verified(monkeypatch):
    """审计 repro：unpinned 扩展首次观测不得打 last_verified_at（无 pin 无 'verified' 语义）。"""
    session = _FakeSession(execute_results=[_Result(1)])  # observed UPDATE 成功
    port = DbExtensionAdmissionPort(_FakeFactory(session), mode="shadow")

    r0 = _row(row_revision=0, artifact_hash=None, observed_artifact_hash=None,
              observed_hash_schema_version=None)

    async def fake_load_row(sess, kind, ext_id):
        return r0

    monkeypatch.setattr(port, "_load_row", fake_load_row)
    obs = Observation(category="artifact", payload="h1", schema_version=1)
    decision = await port.verify_observation("skill", "e1", obs)

    keys = _set_keys(session.executed[0])
    assert "observed_artifact_hash" in keys       # 观测确实落库
    assert "last_verified_at" not in keys          # 但不打 verified 戳
    assert "last_mismatch_at" not in keys          # 也非 mismatch
    assert decision.reason == "unpinned" and decision.observation_outcome == "persisted"


@pytest.mark.anyio
async def test_matched_pin_observation_stamps_last_verified(monkeypatch):
    """匹配的 pinned 比对（digest==pin）→ 打 last_verified_at（正向对照，防过度收紧）。"""
    session = _FakeSession(execute_results=[_Result(6)])
    port = DbExtensionAdmissionPort(_FakeFactory(session), mode="shadow")

    # 值变化触发写：stored 'old' → new 'h0'（==pin），窗口条件无关
    r0 = _row(row_revision=5, artifact_hash="h0", observed_artifact_hash="old",
              observed_hash_schema_version=1)

    async def fake_load_row(sess, kind, ext_id):
        return r0

    monkeypatch.setattr(port, "_load_row", fake_load_row)
    obs = Observation(category="artifact", payload="h0", schema_version=1)
    await port.verify_observation("skill", "e1", obs)

    keys = _set_keys(session.executed[0])
    assert "last_verified_at" in keys
    assert "last_mismatch_at" not in keys
