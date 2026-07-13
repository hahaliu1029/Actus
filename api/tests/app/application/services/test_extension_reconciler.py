"""D1a §6.1 delta 语义（二分/守卫/drift reset 交接）。"""
import pytest

from app.domain.external.extension_admission import (
    AdmissionDecision, InstallContext, UninstallContext,
)
from app.domain.models.app_config import A2AConfig, A2AServerConfig, MCPConfig, MCPServerConfig
from app.application.services.extension_reconciler import ExtensionReconciler


class FakeWrite:
    def __init__(self):
        self.calls = []
    def __getattr__(self, name):
        async def _rec(*a, **k):
            self.calls.append((name, a, k))
            if name == "reset_pins_after_config_drift":
                return True
        return _rec


class FakeAdmission:
    def __init__(self, decision):
        self.decision = decision
        self.calls = []
    async def check_many(self, kind, ext_ids, config_fingerprints=None):
        self.calls.append((kind, list(ext_ids), dict(config_fingerprints or {})))
        return {e: self.decision for e in ext_ids}


def _mcp(**servers):
    return MCPConfig(mcpServers={k: MCPServerConfig(**v) for k, v in servers.items()})


def _ctx():
    return InstallContext(actor_user_id="admin-1", correlation_id=None,
                          source_type="config", source_ref=None, version=None,
                          trust_origin="user_installed", artifact_hash=None,
                          surface_hash="sh", config_fingerprint="fp",
                          hash_schema_version=1, scan=None)


@pytest.mark.anyio
async def test_added_entry_upserts_unpinned():
    w, a = FakeWrite(), FakeAdmission(AdmissionDecision(True, "ok", 1))
    r = ExtensionReconciler(w, a)
    await r.reconcile_mcp_delta(_mcp(), _mcp(s1={"url": "https://x"}))
    assert w.calls[0][0] == "record_reconciled_seen"
    assert w.calls[0][2]["source_type"] == "config"


@pytest.mark.anyio
async def test_removed_without_context_raises():
    # R28#1：删除 diff 而 uninstall_context is None → ValueError（负向锁定）
    w, a = FakeWrite(), FakeAdmission(AdmissionDecision(True, "ok", 1))
    r = ExtensionReconciler(w, a)
    with pytest.raises(ValueError):
        await r.reconcile_mcp_delta(_mcp(s1={"url": "https://x"}), _mcp())


@pytest.mark.anyio
async def test_removed_with_context_soft_deletes():
    w, a = FakeWrite(), FakeAdmission(AdmissionDecision(True, "ok", 1))
    r = ExtensionReconciler(w, a)
    ctx = UninstallContext(correlation_id=None, actor_user_id="admin-1")
    await r.reconcile_mcp_delta(_mcp(s1={"url": "https://x"}), _mcp(), uninstall_context=ctx)
    name, args, kwargs = w.calls[0]
    assert name == "record_delete" and kwargs["uninstall_context"] is ctx


@pytest.mark.anyio
async def test_install_context_branch_is_diffless_record_install():
    # R47#3+#8 断言③：内容相同 diff 为空 → 仍显式 record_install；零 drift 观测零 reset
    w, a = FakeWrite(), FakeAdmission(AdmissionDecision(True, "ok", 1))
    r = ExtensionReconciler(w, a)
    same = _mcp(s1={"url": "https://x"})
    await r.reconcile_mcp_delta(same, same, install_context=_ctx(), target_ext_id="s1")
    assert [c[0] for c in w.calls] == ["record_install"]
    assert a.calls == []          # 不做带外 drift 观测（安装即新 TOFU 基线）


@pytest.mark.anyio
async def test_changed_entry_observes_and_resets_on_drift():
    # §5.2 reset 交接：判据=config_drift_detected 字段非 reason（R46#5——disabled 行也触发）
    decision = AdmissionDecision(False, "disabled", 7, config_drift_detected=True)
    w, a = FakeWrite(), FakeAdmission(decision)
    r = ExtensionReconciler(w, a)
    await r.reconcile_mcp_delta(
        _mcp(s1={"url": "https://old"}), _mcp(s1={"url": "https://new"}))
    assert a.calls[0][0] == "mcp" and "s1" in a.calls[0][2]   # 观测经 Port（R14#6）
    reset = [c for c in w.calls if c[0] == "reset_pins_after_config_drift"]
    assert len(reset) == 1 and reset[0][2]["row_revision"] == 7


@pytest.mark.anyio
async def test_changed_no_drift_no_reset():
    w, a = FakeWrite(), FakeAdmission(AdmissionDecision(True, "ok", 3, config_drift_detected=False))
    r = ExtensionReconciler(w, a)
    await r.reconcile_mcp_delta(
        _mcp(s1={"url": "https://old"}), _mcp(s1={"url": "https://new"}))
    assert not [c for c in w.calls if c[0] == "reset_pins_after_config_drift"]


@pytest.mark.anyio
async def test_a2a_delta_by_id():
    w, a = FakeWrite(), FakeAdmission(AdmissionDecision(True, "ok", 1))
    r = ExtensionReconciler(w, a)
    old = A2AConfig(a2a_servers=[A2AServerConfig(id="i1", base_url="https://a")])
    new = A2AConfig(a2a_servers=[A2AServerConfig(id="i1", base_url="https://a"),
                                 A2AServerConfig(id="i2", base_url="https://b")])
    await r.reconcile_a2a_delta(old, new)
    assert w.calls[0][0] == "record_reconciled_seen" and w.calls[0][1][1] == "i2"
