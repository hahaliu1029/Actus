"""D1a §6.1-2 startup 全量 reconcile（四段全序第④段）单元测试。

Fake ports（write/admission/read）+ fake skill repo + tmp_path skill/plugin 目录。
覆盖 6 路：缺行补建 / config 观测 drift→reset / 源缺失两段确认 / 源恢复 /
skill artifact 观测 / plugin bundle artifact + 缺目录 missing（不 quarantine）。
"""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.application.services.extension_reconciler import ExtensionReconciler
from app.domain.external.extension_admission import AdmissionDecision
from app.domain.models.app_config import (
    A2AConfig,
    A2AServerConfig,
    MCPConfig,
    MCPServerConfig,
)
from app.domain.models.extension_governance import HASH_SCHEMA_VERSION
from app.domain.services.skills_guard import SkillsGuard


class FakeWrite:
    """记录所有写 port 调用（reset_pins_after_config_drift 恒返回 True=CAS 成功）。"""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        async def _rec(*a, **k):
            self.calls.append((name, a, k))
            if name == "reset_pins_after_config_drift":
                return True

        return _rec


class FakeAdmission:
    def __init__(self, *, check_decision=None, verify_decision=None):
        self._check = check_decision or AdmissionDecision(True, "ok", 1)
        self._verify = verify_decision or AdmissionDecision(True, "ok", 1)
        self.check_calls = []
        self.verify_calls = []

    async def check_many(self, kind, ext_ids, config_fingerprints=None):
        self.check_calls.append((kind, list(ext_ids), dict(config_fingerprints or {})))
        return {e: self._check for e in ext_ids}

    async def verify_observation(self, kind, ext_id, obs):
        self.verify_calls.append((kind, ext_id, obs))
        return self._verify


class FakeRead:
    def __init__(self, rows=None):
        self._rows = rows or []

    async def list_live_rows(self):
        return list(self._rows)


class FakeSkillRepo:
    def __init__(self, skills=None, dirs=None):
        self._skills = skills or []
        self._dirs = dirs or {}

    async def list(self):
        return list(self._skills)

    def get_skill_dir(self, skill_id):
        return self._dirs.get(skill_id)


def _row(kind, ext_id, *, source_missing_at=None, version=None):
    return SimpleNamespace(
        kind=kind, ext_id=ext_id, source_missing_at=source_missing_at, version=version
    )


def _skill(skill_id, *, source_type="local", source_ref=None, version=None,
           trust_origin="user_installed"):
    return SimpleNamespace(
        id=skill_id,
        source_type=source_type,
        source_ref=source_ref,
        manifest={"version": version} if version is not None else {},
        trust_origin=trust_origin,
    )


def _mcp(**servers):
    return MCPConfig(mcpServers={k: MCPServerConfig(**v) for k, v in servers.items()})


def _a2a(*ids_urls):
    return A2AConfig(a2a_servers=[A2AServerConfig(id=i, base_url=u) for i, u in ids_urls])


def _appcfg(*, mcp=None, a2a=None):
    return SimpleNamespace(mcp_config=mcp, a2a_config=a2a)


def _reconciler(w, a):
    return ExtensionReconciler(w, a)


def _seen(w):
    return [(c[1][0], c[1][1]) for c in w.calls if c[0] == "record_reconciled_seen"]


@pytest.mark.anyio
async def test_builds_missing_rows_unpinned(tmp_path):
    # config 有 s1/a1、registry 空 → mcp/a2a 各补一 seen；skill 仓有 k1 无行 → skill 一条。
    w, a = FakeWrite(), FakeAdmission()
    r = _reconciler(w, a)
    cfg = _appcfg(mcp=_mcp(s1={"url": "https://x"}), a2a=_a2a(("a1", "https://a")))
    skills = FakeSkillRepo(skills=[_skill("k1", version="9.9.9")], dirs={"k1": None})
    await r.run_startup_reconcile(
        app_config=cfg, skill_repository=skills, read_port=FakeRead([]),
    )
    seen = _seen(w)
    assert ("mcp", "s1") in seen
    assert ("a2a", "a1") in seen
    assert ("skill", "k1") in seen
    # provenance 取 skill 字段（R5#5/R7#4）
    skill_call = [c for c in w.calls
                  if c[0] == "record_reconciled_seen" and c[1][1] == "k1"][0]
    assert skill_call[2]["source_type"] == "local"
    assert skill_call[2]["version"] == "9.9.9"
    assert skill_call[2]["trust_origin"] == "user_installed"


@pytest.mark.anyio
async def test_config_observation_with_reset():
    # registry 行存在 + admission config_drift_detected=True/row_revision=9 → reset 恰一次。
    w = FakeWrite()
    a = FakeAdmission(check_decision=AdmissionDecision(
        False, "disabled", 9, config_drift_detected=True))
    r = _reconciler(w, a)
    cfg = _appcfg(mcp=_mcp(s1={"url": "https://x"}))
    rows = FakeRead([_row("mcp", "s1")])
    await r.run_startup_reconcile(
        app_config=cfg, skill_repository=FakeSkillRepo(), read_port=rows,
    )
    reset = [c for c in w.calls if c[0] == "reset_pins_after_config_drift"]
    assert len(reset) == 1
    assert reset[0][2]["row_revision"] == 9


@pytest.mark.anyio
async def test_source_missing_two_phase():
    # s-gone 首次(source_missing_at None)→mark_source_missing；
    # s-gone2 已置位→record_reconciled_missing（分流在 T8）。
    w, a = FakeWrite(), FakeAdmission()
    r = _reconciler(w, a)
    cfg = _appcfg(mcp=_mcp())  # 空 config → 两行皆缺源
    rows = FakeRead([
        _row("mcp", "s-gone", source_missing_at=None),
        _row("mcp", "s-gone2", source_missing_at=datetime.now(timezone.utc)),
    ])
    await r.run_startup_reconcile(
        app_config=cfg, skill_repository=FakeSkillRepo(), read_port=rows,
    )
    assert [c for c in w.calls
            if c[0] == "mark_source_missing" and c[1] == ("mcp", "s-gone")]
    assert [c for c in w.calls
            if c[0] == "record_reconciled_missing" and c[1] == ("mcp", "s-gone2")]


@pytest.mark.anyio
async def test_source_restored():
    # 行 source_missing_at 置位 + config 含该条 → mark_source_restored。
    w, a = FakeWrite(), FakeAdmission()
    r = _reconciler(w, a)
    cfg = _appcfg(mcp=_mcp(s1={"url": "https://x"}))
    rows = FakeRead([_row("mcp", "s1", source_missing_at=datetime.now(timezone.utc))])
    await r.run_startup_reconcile(
        app_config=cfg, skill_repository=FakeSkillRepo(), read_port=rows,
    )
    assert [c for c in w.calls
            if c[0] == "mark_source_restored" and c[1] == ("mcp", "s1")]


@pytest.mark.anyio
async def test_skill_artifact_observed(tmp_path):
    # R4-03：skill 目录存在 → verify_observation category="artifact"、payload=content_hash。
    w, a = FakeWrite(), FakeAdmission()
    r = _reconciler(w, a)
    skill_dir = tmp_path / "k1"
    skill_dir.mkdir()
    (skill_dir / "manifest.json").write_text('{"name": "k1"}', encoding="utf-8")
    expected_hash = SkillsGuard.compute_content_hash(skill_dir)
    skills = FakeSkillRepo(skills=[_skill("k1")], dirs={"k1": skill_dir})
    rows = FakeRead([_row("skill", "k1")])  # 已有行 → 只观测不补 seen
    await r.run_startup_reconcile(
        app_config=_appcfg(), skill_repository=skills, read_port=rows,
    )
    assert len(a.verify_calls) == 1
    kind, ext_id, obs = a.verify_calls[0]
    assert kind == "skill" and ext_id == "k1"
    assert obs.category == "artifact"
    assert obs.payload == expected_hash
    assert obs.schema_version == HASH_SCHEMA_VERSION


@pytest.mark.anyio
async def test_plugin_bundle_artifact_and_missing(tmp_path):
    # plugin 行 + bundle 目录存在 → artifact 观测；目录缺 → missing 流（不 quarantine，R32#5）。
    w, a = FakeWrite(), FakeAdmission()
    r = _reconciler(w, a)
    plugins_root = tmp_path / "plugins"
    p1_bundle = plugins_root / "p1" / "1.0.0"
    p1_bundle.mkdir(parents=True)
    (p1_bundle / "manifest.json").write_text('{"name": "p1"}', encoding="utf-8")
    expected_hash = SkillsGuard.compute_content_hash(p1_bundle)
    rows = FakeRead([
        _row("plugin", "p1", version="1.0.0"),
        _row("plugin", "p2", version="2.0.0", source_missing_at=None),  # 目录缺
    ])
    await r.run_startup_reconcile(
        app_config=_appcfg(), skill_repository=FakeSkillRepo(), read_port=rows,
        plugins_root=plugins_root,
    )
    plugin_verify = [v for v in a.verify_calls if v[0] == "plugin"]
    assert len(plugin_verify) == 1
    assert plugin_verify[0][1] == "p1"
    assert plugin_verify[0][2].category == "artifact"
    assert plugin_verify[0][2].payload == expected_hash
    # p2 目录缺 → 首次 mark_source_missing；R32#5 不 quarantine
    assert [c for c in w.calls
            if c[0] == "mark_source_missing" and c[1] == ("plugin", "p2")]
    assert not [c for c in w.calls if c[0] == "quarantine"]
