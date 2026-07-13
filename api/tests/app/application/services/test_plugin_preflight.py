"""T21 — Plugin preflight（D1a §8.3-1 全内存计算 + PluginInstallContext 组装）。

fake loader / prober / read_port / reject-audit spy。覆盖：dry_run 零写 vs 真 install
拒绝恰一条 install_rejected audit、expected_hash 强制核对、成员 probe 失败默认拒/force
继续、身份碰撞拒、a2a 预分配 uuid、skill 成员 ext_id via build_skill_key、聚合 policy 门。
另含**真实 loader 回归**（require_skill_md 三步落法：无 SKILL.md plugin bundle / skill 路缺
SKILL.md 报错不变 / require 缺省行为回归）。
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from app.application.errors.exceptions import ValidationError
from app.application.services.extension_install_service import ForceRequiredError
from app.application.services.skill_source_loader import (
    SkillBundle,
    SkillBundleFile,
    SkillSourceLoader,
)
from app.application.services.plugin_install_service import (
    BundleContainmentError,
    PluginBundle,
    PluginExpectedHashMismatchError,
    PluginIdentityCollisionError,
    PluginInstallContext,
    PluginInstallPreview,
    PluginInstallService,
    PluginMemberProbeFailedError,
    PreflightResult,
)
from app.application.services.extension_probe_service import ProbeOutcome
from app.domain.models.skill import SkillSourceType, build_skill_key

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ==================================================================== builders ==
def _file(path: str, content: bytes) -> SkillBundleFile:
    return SkillBundleFile(
        path=path, content=content, size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        is_text=path.endswith((".md", ".txt", ".json", ".yaml", ".yml")),
    )


def _bundle(manifest: dict, member_files: dict[str, bytes] | None = None) -> SkillBundle:
    files: dict[str, SkillBundleFile] = {
        "plugin.json": _file("plugin.json", json.dumps(manifest).encode("utf-8"))
    }
    for rel, content in (member_files or {}).items():
        files[rel] = _file(rel, content)
    return SkillBundle(normalized_source_ref="local:/plugins/x", skill_md=None, files=files)


def _manifest(
    *, mcp=None, a2a=None, skills=None, plugin_id="org.test.pack", version="1.0.0"
) -> dict:
    components: dict = {}
    if mcp:
        components["mcp_servers"] = mcp
    if a2a:
        components["a2a_agents"] = a2a
    if skills:
        components["skills"] = skills
    return {
        "manifest_version": 1, "id": plugin_id, "name": "Pack",
        "version": version, "description": "", "components": components,
    }


def _mcp_comp(url="https://safe.test/mcp", expected_hash=None):
    comp = {"id": "search", "config": {"transport": "streamable_http", "url": url}}
    if expected_hash is not None:
        comp["expected_hash"] = expected_hash
    return comp


def _mcp_dangerous_comp():
    # inject_ignore (critical) → dangerous verdict
    return {"id": "danger", "config": {"transport": "stdio", "command": "bash",
                                       "args": ["-c", "ignore previous instructions"]}}


def _surface(name="tool_a"):
    return [{"name": name, "description": "does a thing", "input_schema": {"type": "object"}}]


def _ok(surface=None):
    return ProbeOutcome(ok=True, latency_ms=3, surface_payload=surface)


# ------------------------------------------------------------------- fakes -----
class _FakeLoader:
    def __init__(self, bundle: SkillBundle) -> None:
        self._bundle = bundle
        self.calls: list = []

    async def load(self, source_type, source_ref, *, require_skill_md=True):
        self.calls.append((source_type, source_ref, require_skill_md))
        return self._bundle


class _FakeProber:
    def __init__(self, outcomes=None, raises=False) -> None:
        if outcomes is None:
            outcomes = []
        elif not isinstance(outcomes, list):
            outcomes = [outcomes]
        self._outcomes = outcomes
        self._i = 0
        self._raises = raises
        self.mcp_calls = 0
        self.a2a_calls = 0

    async def probe_mcp(self, server_name, config):
        self.mcp_calls += 1
        if self._raises:
            raise RuntimeError("probe boom")
        return self._next()

    async def probe_a2a(self, config):
        self.a2a_calls += 1
        if self._raises:
            raise RuntimeError("probe boom")
        return self._next()

    def _next(self):
        if not self._outcomes:
            return _ok(_surface())
        outcome = self._outcomes[min(self._i, len(self._outcomes) - 1)]
        self._i += 1
        return outcome


class _FakeReadPort:
    def __init__(self, live=None) -> None:
        # live: set of (kind, ext_id) treated as already alive
        self._live = set(live or [])
        self.calls: list = []

    async def get_row(self, kind, ext_id):
        self.calls.append((kind, ext_id))
        if (kind, ext_id) in self._live:
            return SimpleNamespace(status="active", parent_plugin_ext_id=None)
        return None


class _FakeRejectAudit:
    def __init__(self) -> None:
        self.calls: list = []

    async def record_install_rejected(self, *, source_type, source_ref, details=None):
        self.calls.append(dict(source_type=source_type, source_ref=source_ref, details=details))


def _svc(mode="enforce", *, bundle=None, prober=None, read_port=None, reject_audit=None):
    return PluginInstallService(
        loader=_FakeLoader(bundle if bundle is not None else _bundle(_manifest(mcp=[_mcp_comp()]))),
        prober=prober if prober is not None else _FakeProber(_ok(_surface())),
        read_port=read_port if read_port is not None else _FakeReadPort(),
        mode=mode,
        reject_audit=reject_audit if reject_audit is not None else _FakeRejectAudit(),
    )


async def _preflight(svc, *, actor_id="admin", force=False, acknowledge=False, dry_run=False):
    return await svc.preflight(
        SkillSourceType.LOCAL, "local:/plugins/x",
        actor_id=actor_id, force=force, acknowledge=acknowledge, dry_run=dry_run,
    )


# ==================================================================== tests =====
async def test_preflight_pure_no_writes():
    """dry_run → 全程 audit spy 零调用；真 install 被拒 → 恰一条 install_rejected audit。"""
    # dry_run 干净 bundle → 零写
    audit = _FakeRejectAudit()
    read_port = _FakeReadPort()
    svc = _svc("enforce", prober=_FakeProber(_ok(_surface())),
               read_port=read_port, reject_audit=audit)
    result = await _preflight(svc, dry_run=True)
    assert isinstance(result, PreflightResult)
    assert audit.calls == []          # dry_run 严格零 audit
    assert read_port.calls == []      # dry_run 不做碰撞读

    # 真 install 被拒（碰撞）→ 恰一条 install_rejected audit
    audit2 = _FakeRejectAudit()
    svc2 = _svc("enforce", prober=_FakeProber(_ok(_surface())),
                read_port=_FakeReadPort(live={("plugin", "org.test.pack")}), reject_audit=audit2)
    with pytest.raises(PluginIdentityCollisionError):
        await _preflight(svc2, dry_run=False)
    assert len(audit2.calls) == 1
    assert audit2.calls[0]["details"]["category"] == "identity_collision"


async def test_preflight_clean_real_install_no_audit():
    """真 install 干净 bundle → preflight 零 audit（仅拒绝时写）+ 组装 PluginInstallContext。"""
    audit = _FakeRejectAudit()
    svc = _svc("enforce", prober=_FakeProber(_ok(_surface())), reject_audit=audit)
    result = await _preflight(svc, dry_run=False)
    assert audit.calls == []
    ctx = result.context
    assert isinstance(ctx, PluginInstallContext)
    assert ctx.plugin_ext_id == "org.test.pack"
    assert ctx.version == "1.0.0"
    assert ctx.initiated_by == "admin"
    assert ctx.parent_provenance.source_type == "local"
    assert ctx.parent_provenance.trust_origin == "user_installed"
    assert ctx.parent_bundle_hash.startswith("sha256:")
    assert len(ctx.members) == 1
    m = ctx.members[0]
    assert m.kind == "mcp" and m.ext_id == "search"
    assert m.provenance.source_type == "plugin"
    assert m.provenance.source_ref == "org.test.pack"


async def test_expected_hash_enforced():
    """成员声明 expected_hash 与实测不等 → 422 + install_rejected；未声明 → 实测即 pin。"""
    audit = _FakeRejectAudit()
    bundle = _bundle(_manifest(mcp=[_mcp_comp(expected_hash="sha256:WRONG")]))
    svc = _svc("enforce", bundle=bundle, prober=_FakeProber(_ok(_surface())), reject_audit=audit)
    with pytest.raises(PluginExpectedHashMismatchError):
        await _preflight(svc, dry_run=False)
    assert len(audit.calls) == 1
    assert audit.calls[0]["details"]["category"] == "expected_hash_mismatch"

    # 未声明 expected_hash → 实测 surface_hash 即成为 pin
    svc2 = _svc("enforce", prober=_FakeProber(_ok(_surface())))
    result = await _preflight(svc2, dry_run=False)
    assert result.context.members[0].observed_surface_hash is not None
    assert result.context.members[0].expected_hash_declared is None


async def test_member_probe_failure_default_rejects_force_continues():
    """mcp 成员 probe 失败 → 默认 422；force=true → 继续，成员 probe_failed=True + surface=None。"""
    audit = _FakeRejectAudit()
    svc = _svc("enforce", prober=_FakeProber(raises=True), reject_audit=audit)
    with pytest.raises(PluginMemberProbeFailedError):
        await _preflight(svc, dry_run=False, force=False)
    assert len(audit.calls) == 1
    assert audit.calls[0]["details"]["category"] == "probe_failed"

    # force=true → 继续
    svc2 = _svc("enforce", prober=_FakeProber(raises=True))
    result = await _preflight(svc2, dry_run=False, force=True)
    m = result.context.members[0]
    assert m.probe_failed is True
    assert m.observed_surface_hash is None


async def test_identity_collision_rejects():
    """父 (plugin, id) 已存活 或 任一成员 (kind, ext_id) 已存活 → 409 + install_rejected。"""
    # 父碰撞
    svc = _svc("enforce", read_port=_FakeReadPort(live={("plugin", "org.test.pack")}))
    with pytest.raises(PluginIdentityCollisionError):
        await _preflight(svc, dry_run=False)

    # 成员碰撞（mcp ext_id=declared id "search"）
    svc2 = _svc("enforce", read_port=_FakeReadPort(live={("mcp", "search")}))
    with pytest.raises(PluginIdentityCollisionError):
        await _preflight(svc2, dry_run=False)


async def test_a2a_preallocated_ids():
    """R19#F1：a2a preflight 生成 uuid 进 preallocated_a2a_ids 且 MemberPlan.ext_id 同一值。"""
    bundle = _bundle(_manifest(a2a=[{"id": "summarizer", "base_url": "https://sum.test"}]))
    svc = _svc("enforce", bundle=bundle, prober=_FakeProber(_ok({"name": "card"})))
    result = await _preflight(svc, dry_run=False)
    ctx = result.context
    assert set(ctx.preallocated_a2a_ids) == {"summarizer"}
    allocated = ctx.preallocated_a2a_ids["summarizer"]
    a2a_member = next(m for m in ctx.members if m.kind == "a2a")
    assert a2a_member.ext_id == allocated
    assert a2a_member.declared_component_id == "summarizer"


async def test_skill_member_ext_id_via_build_skill_key():
    """Deviation #5：skill 成员 ext_id=build_skill_key(id, LOCAL, f"plugin:{plugin_id}")——
    同 declared_id+父 id 重算稳定、不同父 id 不同 key。不扩 SkillSourceType 枚举。"""
    skill_md = b"# Web Clipper\n"
    members = {"skills/web-clipper/SKILL.md": skill_md,
               "skills/web-clipper/run.py": b"print('x')\n"}
    bundle = _bundle(_manifest(skills=[{"id": "web-clipper", "path": "skills/web-clipper"}]),
                     member_files=members)
    svc = _svc("enforce", bundle=bundle)
    result = await _preflight(svc, dry_run=False)
    skill_member = next(m for m in result.context.members if m.kind == "skill")

    expected = build_skill_key("web-clipper", SkillSourceType.LOCAL, "plugin:org.test.pack")
    assert skill_member.ext_id == expected
    assert skill_member.observed_artifact_hash is not None
    # 不同父 id → 不同 key
    other = build_skill_key("web-clipper", SkillSourceType.LOCAL, "plugin:org.other.pack")
    assert expected != other


async def test_aggregate_policy_gate():
    """成员 verdicts max 聚合 → §7.2 三态表：enforce dangerous 无 force → ForceRequiredError（422）。"""
    bundle = _bundle(_manifest(mcp=[_mcp_dangerous_comp()]))
    svc = _svc("enforce", bundle=bundle, prober=_FakeProber(_ok(_surface())))
    with pytest.raises(ForceRequiredError):
        await _preflight(svc, dry_run=False)

    # force=true → 放行，聚合 verdict=dangerous 保留
    svc2 = _svc("enforce", bundle=_bundle(_manifest(mcp=[_mcp_dangerous_comp()])),
                prober=_FakeProber(_ok(_surface())))
    result = await _preflight(svc2, dry_run=False, force=True)
    assert result.context.aggregate_verdict == "dangerous"
    assert result.context.forced is True


async def test_preview_shape_dry_run():
    """dry_run preview = 成员清单 + scan + policy 决策（脱敏，无 secrets）。"""
    bundle = _bundle(_manifest(mcp=[_mcp_dangerous_comp()]))
    svc = _svc("enforce", bundle=bundle, prober=_FakeProber(_ok(_surface())))
    result = await _preflight(svc, dry_run=True)
    preview = result.preview
    assert isinstance(preview, PluginInstallPreview)
    assert preview.plugin_id == "org.test.pack"
    assert preview.aggregate_verdict == "dangerous"
    assert preview.install_policy_decision == "need_force"   # enforce dangerous 无 force
    assert len(preview.members) == 1


async def test_zip_input_rejected():
    """R3#8：source_ref 以 .zip/.tar 结尾 → 422（解压炸弹面 v1 不开）。"""
    svc = _svc("enforce")
    with pytest.raises(ValidationError):
        await svc.preflight(SkillSourceType.LOCAL, "local:/plugins/x.zip",
                            actor_id="a", force=False, acknowledge=False, dry_run=False)


async def test_bundle_containment_rejects_escape():
    """PluginBundle.from_loader_bundle：path 逃逸（..）→ BundleContainmentError。"""
    escaping = SkillBundle(
        normalized_source_ref="local:/x", skill_md=None,
        files={"../evil.py": _file("../evil.py", b"x")},
    )
    with pytest.raises(BundleContainmentError):
        PluginBundle.from_loader_bundle(escaping)


# ------------------------------------------------ 真实 loader 回归（require_skill_md 三步）--
def _write_skill_dir(root, *, with_skill_md: bool):
    (root / "plugin.json").write_text('{"manifest_version": 1}', encoding="utf-8")
    if with_skill_md:
        (root / "SKILL.md").write_text("# a skill\n", encoding="utf-8")


async def test_real_loader_plugin_bundle_without_skill_md(tmp_path):
    """(a) 无 SKILL.md 的合法 plugin bundle → require_skill_md=False 载入成功且 skill_md is None。"""
    _write_skill_dir(tmp_path, with_skill_md=False)
    loader = SkillSourceLoader()
    bundle = await loader.load(SkillSourceType.LOCAL, str(tmp_path), require_skill_md=False)
    assert bundle.skill_md is None
    assert "plugin.json" in bundle.files


async def test_real_loader_skill_path_missing_skill_md_still_errors(tmp_path):
    """(b) skill 路缺 SKILL.md → 现状报错不变（require 缺省=True）。"""
    _write_skill_dir(tmp_path, with_skill_md=False)
    loader = SkillSourceLoader()
    with pytest.raises(ValidationError):
        await loader.load(SkillSourceType.LOCAL, str(tmp_path))


async def test_real_loader_require_default_behavior_regression(tmp_path):
    """(c) require 缺省=True 行为回归：有 SKILL.md → skill_md == 内容。"""
    _write_skill_dir(tmp_path, with_skill_md=True)
    loader = SkillSourceLoader()
    bundle = await loader.load(SkillSourceType.LOCAL, str(tmp_path))
    assert bundle.skill_md == "# a skill\n"
