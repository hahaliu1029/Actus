"""T22 — Plugin install saga（D1a §8.3 骨架→内容写入→复验→发布→补偿）状态机/顺序合同。

fake store/services——**不**依赖真 PG（DB 事务保真度归 CI-only integration）。覆盖 spec §11.3
十二用例：锁集合、锁内双源重检、骨架状态、写前置标 attempting、config/skill/bundle create 语义
碰撞、补偿 scope、补偿走 delta hooks 带 saga ctx（+自失败→failed 父行 disabled）、复验失败补偿、
发布成功事务、成员 InstallContext 合成、staging→rename 进位。

fs 隔离：autouse fixture monkeypatch PLUGIN_*_ROOT → tmp。config：单一 ``_FakeConfig`` 同时充当
AppConfigService（写）与 config_loader（读 occupied/fingerprint/entries），保证 reverify/rollback
三分与 preflight 观测天然同源。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.application.services.plugin_install_service as saga_mod
from app.application.services.plugin_install_service import (
    InstallResult,
    PluginIdentityCollisionError,
    PluginInstallPreview,
    PluginInstallService,
    _bundle_key,
)
from app.application.services.extension_probe_service import ProbeOutcome
from app.application.services.skill_source_loader import SkillBundle, SkillBundleFile
from app.domain.external.extension_admission import InstallContext
from app.domain.models.skill import SkillSourceType, build_skill_key
from app.domain.services.extension_hashing import (
    a2a_config_fingerprint,
    entry_content_hash,
    mcp_config_fingerprint,
)
from app.infrastructure.external.governance.plugin_saga_store import (
    SkeletonResult,
    build_saga_steps,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _saga_roots(tmp_path, monkeypatch):
    """每测试隔离 fs 根（staging/plugins/skills 同 tmp 下——同 fs 保证 os.replace 原子）。"""
    monkeypatch.setattr(saga_mod, "PLUGIN_STAGING_ROOT", tmp_path / ".plugin-staging")
    monkeypatch.setattr(saga_mod, "PLUGIN_STORE_ROOT", tmp_path / "plugins")
    monkeypatch.setattr(saga_mod, "SKILL_STORE_ROOT", tmp_path / "skills")
    return tmp_path


# ==================================================================== builders ==
def _file(path: str, content: bytes) -> SkillBundleFile:
    return SkillBundleFile(
        path=path, content=content, size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        is_text=path.endswith((".md", ".txt", ".json", ".yaml", ".yml")))


def _bundle(manifest: dict, member_files: dict[str, bytes] | None = None) -> SkillBundle:
    files: dict[str, SkillBundleFile] = {
        "plugin.json": _file("plugin.json", json.dumps(manifest).encode("utf-8"))}
    for rel, content in (member_files or {}).items():
        files[rel] = _file(rel, content)
    return SkillBundle(normalized_source_ref="local:/plugins/x", skill_md=None, files=files)


def _manifest(*, mcp=None, a2a=None, skills=None, plugin_id="org.test.pack", version="1.0.0") -> dict:
    components: dict = {}
    if mcp:
        components["mcp_servers"] = mcp
    if a2a:
        components["a2a_agents"] = a2a
    if skills:
        components["skills"] = skills
    return {"manifest_version": 1, "id": plugin_id, "name": "Pack",
            "version": version, "description": "", "components": components}


def _mcp_comp(cid="search", url="https://safe.test/mcp"):
    return {"id": cid, "config": {"transport": "streamable_http", "url": url}}


def _skill_files(name="web-clipper"):
    return {f"skills/{name}/SKILL.md": b"# Web Clipper\n",
            f"skills/{name}/run.py": b"print('x')\n"}


def _surface(name="tool_a"):
    return [{"name": name, "description": "does a thing", "input_schema": {"type": "object"}}]


def _ok(surface=None):
    return ProbeOutcome(ok=True, latency_ms=3, surface_payload=surface)


# ------------------------------------------------------------------- fakes -----
class _FakeLoader:
    def __init__(self, bundle):
        self._bundle = bundle

    async def load(self, source_type, source_ref, *, require_skill_md=True):
        return self._bundle


class _FakeProber:
    async def probe_mcp(self, server_name, config):
        return _ok(_surface())

    async def probe_a2a(self, config):
        return _ok({"name": "card"})


class _FakeReadPort:
    def __init__(self, live=None):
        self._live = set(live or [])
        self.calls: list = []

    async def get_row(self, kind, ext_id):
        self.calls.append((kind, ext_id))
        if (kind, ext_id) in self._live:
            return SimpleNamespace(status="active", parent_plugin_ext_id=None)
        return None


class _FakeRejectAudit:
    def __init__(self):
        self.calls: list = []

    async def record_install_rejected(self, *, source_type, source_ref, details=None):
        self.calls.append(dict(source_type=source_type, source_ref=source_ref, details=details))


class _FakeLocks:
    def __init__(self):
        self.acquired: list = []

    def acquire_all(self, identities):
        self.acquired.append(list(identities))
        return _NullCtx()


class _NullCtx:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


class _FakeConfig:
    """AppConfigService(写) + config_loader(读) 合一——reverify/rollback 与 preflight 同源。"""

    def __init__(self, *, prepopulated=None, load_raises=False):
        self.mcp: dict = {}
        self.a2a: dict = {}
        for kind, ext_id in (prepopulated or []):
            (self.mcp if kind == "mcp" else self.a2a)[ext_id] = SimpleNamespace()
        self.create_calls: list = []
        self.delete_calls: list = []
        self._delete_raises = False
        self._load_raises = load_raises

    # ---- AppConfigService 写面 ----
    async def update_and_create_mcp_servers(self, mcp_config, *, actor_id=None,
                                            install_context=None, target_server=None):
        for name, cfg in mcp_config.mcpServers.items():
            self.mcp[name] = cfg
        self.create_calls.append(SimpleNamespace(
            kind="mcp", ext_id=target_server, install_context=install_context, actor_id=actor_id))

    async def create_a2a_server(self, base_url, *, actor_id=None,
                                install_context=None, preallocated_id=None):
        self.a2a[preallocated_id] = base_url
        self.create_calls.append(SimpleNamespace(
            kind="a2a", ext_id=preallocated_id, base_url=base_url,
            install_context=install_context, actor_id=actor_id))

    async def delete_mcp_server(self, server_name, *, actor_id=None,
                                uninstall_context=None, missing_ok=False):
        self.delete_calls.append(SimpleNamespace(
            kind="mcp", ext_id=server_name, uninstall_context=uninstall_context, missing_ok=missing_ok))
        if self._delete_raises:
            raise RuntimeError("delete boom")
        self.mcp.pop(server_name, None)

    async def delete_a2a_server(self, a2a_id, *, actor_id=None,
                                uninstall_context=None, missing_ok=False):
        self.delete_calls.append(SimpleNamespace(
            kind="a2a", ext_id=a2a_id, uninstall_context=uninstall_context, missing_ok=missing_ok))
        if self._delete_raises:
            raise RuntimeError("delete boom")
        self.a2a.pop(a2a_id, None)

    # ---- config_loader 读面 ----
    def occupied(self, kind, ext_id):
        return ext_id in (self.mcp if kind == "mcp" else self.a2a)

    def current_fingerprint(self, kind, ext_id):
        if kind == "mcp" and ext_id in self.mcp:
            return mcp_config_fingerprint(self.mcp[ext_id])
        if kind == "a2a" and ext_id in self.a2a:
            return a2a_config_fingerprint(self.a2a[ext_id])
        return None

    def load_entries(self):
        if self._load_raises:
            raise RuntimeError("config unreadable")
        # 带外占用哨兵（无 model_dump）以 stub 表示——不崩溃；collided step 在 rollback 恒被跳过。
        mcp = {n: (c.model_dump(mode="json") if hasattr(c, "model_dump") else {"_stub": n})
               for n, c in self.mcp.items()}
        return {"mcp": mcp, "a2a": {}}


class _FakeSkillService:
    """install_skill = 复制 staging → SKILL_STORE_ROOT/{name}（令 skill reverify 天然通过）。"""

    def __init__(self, *, install_raises=False):
        self.install_calls: list = []
        self.delete_calls: list = []
        self._install_raises = install_raises

    async def install_skill(self, source_type, source_ref, manifest, skill_md, installed_by, *,
                            trust_origin, force, actor_id, governance_install_context):
        self.install_calls.append(SimpleNamespace(
            source_type=source_type, source_ref=source_ref, installed_by=installed_by,
            trust_origin=trust_origin, force=force, actor_id=actor_id,
            ctx=governance_install_context))
        if self._install_raises:
            raise RuntimeError("install boom")
        name = Path(source_ref).name
        dest = saga_mod.SKILL_STORE_ROOT / name
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(source_ref, dest)
        return SimpleNamespace(id=name)

    async def delete_skill(self, skill_id, *, actor_id=None, uninstall_context=None, missing_ok=False):
        self.delete_calls.append(SimpleNamespace(
            skill_id=skill_id, uninstall_context=uninstall_context, missing_ok=missing_ok))
        shutil.rmtree(saga_mod.SKILL_STORE_ROOT / skill_id, ignore_errors=True)


class _FakeWritePort:
    def __init__(self):
        self.install_calls: list = []
        self.delete_calls: list = []

    async def record_install(self, kind, ext_id, install_context):
        self.install_calls.append(SimpleNamespace(kind=kind, ext_id=ext_id, ctx=install_context))

    async def record_delete(self, kind, ext_id, *, uninstall_context):
        self.delete_calls.append(SimpleNamespace(kind=kind, ext_id=ext_id, uninstall_context=uninstall_context))


class _FakeSagaStore:
    """in-memory 忠实建模骨架/步进/发布/补偿 DB 语义（真 DB 保真度归 integration）。"""

    def __init__(self):
        self.ops: dict = {}
        self.parent_rows: dict = {}
        self.member_rows: dict = {}
        self.memberships: list = []
        self.audits: list = []
        self.mark_calls: list = []
        self.skeleton_calls: list = []
        self.skeleton_snapshot = None

    async def create_install_skeleton(self, ctx):
        op_id = uuid.uuid4()
        plugin_row_id = uuid.uuid4()
        self.parent_rows[ctx.plugin_ext_id] = {"status": "disabled", "deleted": False, "id": plugin_row_id}
        for m in ctx.members:
            self.member_rows[(m.kind, m.ext_id)] = {"status": "active", "deleted": False}
            self.memberships.append((ctx.plugin_ext_id, m.kind, m.ext_id))
        steps = build_saga_steps(ctx)
        self.ops[op_id] = dict(
            id=op_id, plugin_ext_id=ctx.plugin_ext_id, plugin_extension_id=plugin_row_id,
            initiated_by=ctx.initiated_by, state="in_progress", steps=steps, error=None)
        self.audits.append(dict(event="plugin_expand_started", ext_id=ctx.plugin_ext_id,
                                correlation_id=op_id, actor=ctx.initiated_by, details=None))
        self.skeleton_calls.append(ctx)
        self.skeleton_snapshot = dict(
            parent_status="disabled",
            member_statuses={(m.kind, m.ext_id): "active" for m in ctx.members},
            membership_count=len(ctx.members),
            steps=[dict(s) for s in steps], op_state="in_progress")
        return SkeletonResult(operation_id=op_id, plugin_row_id=plugin_row_id)

    async def mark_step(self, operation_id, seq, state):
        self.mark_calls.append((seq, state))
        for s in self.ops[operation_id]["steps"]:
            if s["seq"] == seq:
                s["state"] = state
                break

    async def complete_install(self, operation_id):
        op = self.ops[operation_id]
        self.parent_rows[op["plugin_ext_id"]]["status"] = "active"
        op["state"] = "completed"
        self.audits.append(dict(event="plugin_expand_completed", ext_id=op["plugin_ext_id"],
                                correlation_id=operation_id, actor=op["initiated_by"], details=None))

    async def compensate(self, operation_id, *, details):
        op = self.ops[operation_id]
        self.parent_rows[op["plugin_ext_id"]]["deleted"] = True
        for (pid, kind, ext_id) in list(self.memberships):
            if pid == op["plugin_ext_id"]:
                self.member_rows[(kind, ext_id)]["deleted"] = True
        self.memberships = [m for m in self.memberships if m[0] != op["plugin_ext_id"]]
        op["state"] = "compensated"
        collided = details.get("collided_targets") or []
        if collided:
            self.audits.append(dict(event="install_rejected", ext_id=op["plugin_ext_id"],
                                    correlation_id=operation_id, actor=op["initiated_by"],
                                    details={"stage": "recovery_collision", "collided_targets": collided}))
        self.audits.append(dict(event="plugin_expand_compensated", ext_id=op["plugin_ext_id"],
                                correlation_id=operation_id, actor=op["initiated_by"],
                                details={"failed_step": details.get("failed_step"),
                                         "compensated_targets": details.get("compensated_targets")}))

    async def fail(self, operation_id, *, error):
        self.ops[operation_id]["state"] = "failed"
        self.ops[operation_id]["error"] = error

    async def load_operation(self, operation_id):
        op = self.ops.get(operation_id)
        if op is None:
            return None
        return SimpleNamespace(
            id=op["id"], operation_type="plugin_install",
            plugin_extension_id=op["plugin_extension_id"], plugin_ext_id=op["plugin_ext_id"],
            initiated_by=op["initiated_by"], state=op["state"],
            steps=[dict(s) for s in op["steps"]], error=op["error"])

    async def record_content_collision_audit(self, operation_id, *, stage, member=None,
                                              target_key=None, collided_targets=None):
        op = self.ops[operation_id]
        details = {"stage": stage}
        if member is not None:
            details["member"] = member
        if target_key is not None:
            details["target_key"] = target_key
        if collided_targets is not None:
            details["collided_targets"] = collided_targets
        self.audits.append(dict(event="install_rejected", ext_id=op["plugin_ext_id"],
                                correlation_id=operation_id, actor=op["initiated_by"], details=details))

    # ---- 断言 helper ----
    def events(self):
        return [a["event"] for a in self.audits]


# ------------------------------------------------------------------- harness ---
def _make_svc(bundle, *, mode="enforce", store=None, skill=None, config=None,
              write_port=None, read_port=None, reject=None, locks=None):
    config = config if config is not None else _FakeConfig()
    return PluginInstallService(
        loader=_FakeLoader(bundle), prober=_FakeProber(),
        read_port=read_port if read_port is not None else _FakeReadPort(),
        mode=mode, reject_audit=reject or _FakeRejectAudit(),
        identity_locks=locks or _FakeLocks(),
        store=store or _FakeSagaStore(), skill_service=skill or _FakeSkillService(),
        app_config_service=config, write_port=write_port or _FakeWritePort(),
        config_loader=config)


async def _install(svc, *, force=False, acknowledge=False, dry_run=False, actor="admin"):
    return await svc.install(SkillSourceType.LOCAL, "local:/plugins/x", actor_id=actor,
                             force=force, acknowledge=acknowledge, dry_run=dry_run)


async def _preflight_ctx(svc):
    return (await svc.preflight(SkillSourceType.LOCAL, "local:/plugins/x",
                                actor_id="admin", force=False, acknowledge=False,
                                dry_run=False)).context


# ==================================================================== tests =====
async def test_lock_set_includes_parent_and_all_members():
    """§3.6 R36#5：acquire_all 收到 {父 plugin}∪{全部成员} 排序键集。"""
    bundle = _bundle(_manifest(mcp=[_mcp_comp("search"), _mcp_comp("fetch")]))
    locks = _FakeLocks()
    svc = _make_svc(bundle, locks=locks)
    await _install(svc)
    assert len(locks.acquired) == 1
    keys = locks.acquired[0]
    assert keys == sorted([("plugin", "org.test.pack"), ("mcp", "search"), ("mcp", "fetch")])


async def test_in_lock_recheck_dual_source_registry():
    """R46#3 断言②(a)：registry 存活行 → 409 + install_rejected，骨架零建。"""
    bundle = _bundle(_manifest(mcp=[_mcp_comp("search")]))
    store = _FakeSagaStore()
    reject = _FakeRejectAudit()
    svc = _make_svc(bundle, store=store, reject=reject,
                    read_port=_FakeReadPort(live={("mcp", "search")}))
    with pytest.raises(PluginIdentityCollisionError):
        await _install(svc)
    assert store.skeleton_calls == []                       # 骨架零建
    assert len(reject.calls) == 1
    assert reject.calls[0]["details"]["category"] == "identity_collision"


async def test_in_lock_recheck_dual_source_native_config():
    """R46#3 断言②(b)：原生 config 同名键占用（config_loader）→ 409，骨架零建。"""
    bundle = _bundle(_manifest(mcp=[_mcp_comp("search")]))
    store = _FakeSagaStore()
    config = _FakeConfig(prepopulated=[("mcp", "search")])
    svc = _make_svc(bundle, store=store, config=config)
    with pytest.raises(PluginIdentityCollisionError):
        await _install(svc)
    assert store.skeleton_calls == []


async def test_in_lock_recheck_dual_source_skill_and_bundle_dirs(_saga_roots):
    """R47#5：skill 目录 / 父 bundle 目录已存在（原生占用）→ 409，骨架零建。"""
    # skill 目录占用
    ext_id = build_skill_key("web-clipper", SkillSourceType.LOCAL, "plugin:org.test.pack")
    (saga_mod.SKILL_STORE_ROOT / ext_id).mkdir(parents=True)
    bundle = _bundle(_manifest(skills=[{"id": "web-clipper", "path": "skills/web-clipper"}]),
                     member_files=_skill_files())
    store = _FakeSagaStore()
    svc = _make_svc(bundle, store=store)
    with pytest.raises(PluginIdentityCollisionError):
        await _install(svc)
    assert store.skeleton_calls == []

    # 父 bundle 目录占用
    (saga_mod.PLUGIN_STORE_ROOT / _bundle_key("org.test.pack", "1.0.0")).mkdir(parents=True)
    store2 = _FakeSagaStore()
    svc2 = _make_svc(_bundle(_manifest(mcp=[_mcp_comp("search")])), store=store2)
    with pytest.raises(PluginIdentityCollisionError):
        await _install(svc2)
    assert store2.skeleton_calls == []


async def test_skeleton_insert_order_and_states():
    """骨架后父行 disabled、成员行 active、membership 齐、op in_progress、steps 全 planned +
    每 target expected_hash（skill/bundle=快照 / mcp,a2a=entry_content_hash）。"""
    bundle = _bundle(
        _manifest(mcp=[_mcp_comp("search")],
                  skills=[{"id": "web-clipper", "path": "skills/web-clipper"}]),
        member_files=_skill_files())
    svc = _make_svc(bundle)
    ctx = await _preflight_ctx(svc)

    # 纯函数 build_saga_steps：全 planned + expected_hash per target
    steps = build_saga_steps(ctx)
    assert [s["state"] for s in steps] == ["planned"] * len(steps)
    bundle_step = steps[0]
    assert bundle_step["target"]["type"] == "plugin_bundle"
    assert bundle_step["expected_hash"] == ctx.parent_bundle_hash
    by_key = {s["target"]["key"]: s for s in steps[1:]}
    for m in ctx.members:
        st = by_key[m.ext_id]
        if m.kind == "skill":
            assert st["target"]["type"] == "skill_dir"
            assert st["expected_hash"] == m.observed_artifact_hash
        else:
            assert st["target"]["type"] == "mcp_config"
            assert st["expected_hash"] == entry_content_hash(m.entry_dump)

    # 骨架快照（fake store 于 create 时捕获）：父 disabled / 成员 active / membership 齐 / op in_progress
    store = _FakeSagaStore()
    svc2 = _make_svc(bundle, store=store)
    await _install(svc2)
    snap = store.skeleton_snapshot
    assert snap["parent_status"] == "disabled"
    assert set(snap["member_statuses"].values()) == {"active"}
    assert snap["membership_count"] == len(ctx.members)
    assert snap["op_state"] == "in_progress"
    assert [s["state"] for s in snap["steps"]] == ["planned"] * len(snap["steps"])


async def test_content_write_marks_attempting_before_external_write():
    """R48#2：每 target 顺序 = mark_step(attempting) → 外部写 → mark_step(done)。"""
    bundle = _bundle(_manifest(mcp=[_mcp_comp("search")]))
    store = _FakeSagaStore()
    svc = _make_svc(bundle, store=store)
    await _install(svc)
    # bundle(seq1) + mcp(seq2) 各自 attempting 先于 done；同 seq attempting 在 done 之前
    marks = store.mark_calls
    for seq in (1, 2):
        seq_marks = [state for (s, state) in marks if s == seq]
        assert seq_marks == ["attempting", "done"], seq_marks


async def test_config_member_create_semantics():
    """R48#3：写入时刻 config 键已存在 → 该 step collided + abort 转补偿 + content_write_collision
    audit（config 键零覆盖）。构造：occupied 命中在**写入**而非重检——用 config 写面预置。"""
    bundle = _bundle(_manifest(mcp=[_mcp_comp("search")]))
    store = _FakeSagaStore()
    config = _FakeConfig()
    svc = _make_svc(bundle, store=store, config=config)
    # 骨架建成后、写 mcp 之前令 config 键出现——用 monkeypatch 令 _native_occupied 对 mcp 返 True
    # 但对 recheck 阶段（骨架前）返 False：通过在 create_install_skeleton 后注入 occupancy。
    orig_create = store.create_install_skeleton

    async def _create(ctx):
        result = await orig_create(ctx)
        config.mcp["search"] = SimpleNamespace()             # 骨架后带外占用 mcp 键
        return result

    store.create_install_skeleton = _create
    result = await _install(svc)
    assert isinstance(result, InstallResult)
    assert result.status == "compensated"
    assert "mcp_config:search" in result.collided_targets
    assert "content_write_collision" in [
        a["details"].get("stage") for a in store.audits
        if a["event"] == "install_rejected" and a["details"]]
    assert config.mcp["search"].__class__ is SimpleNamespace   # 零覆盖（未被 saga 内容替换）
    assert store.parent_rows["org.test.pack"]["deleted"] is True


async def test_skill_and_bundle_dir_collision_same_semantics(_saga_roots):
    """R47#5：skill 成员目标目录（写入时刻已存在）→ 同 config 语义 collided + abort。"""
    ext_id = build_skill_key("web-clipper", SkillSourceType.LOCAL, "plugin:org.test.pack")
    bundle = _bundle(_manifest(skills=[{"id": "web-clipper", "path": "skills/web-clipper"}]),
                     member_files=_skill_files())
    store = _FakeSagaStore()
    svc = _make_svc(bundle, store=store)
    orig_create = store.create_install_skeleton

    async def _create(ctx):
        result = await orig_create(ctx)
        (saga_mod.SKILL_STORE_ROOT / ext_id).mkdir(parents=True)   # 骨架后带外占用 skill 目录
        return result

    store.create_install_skeleton = _create
    result = await _install(svc)
    assert result.status == "compensated"
    assert f"skill_dir:{ext_id}" in result.collided_targets


async def test_compensation_scope():
    """R48#2/R47#4：第 k 步 collided abort → 补偿只处理 {attempting,done}；k+1 planned 恒不删；
    collided target 零删除零 uninstalled。三成员：mcp1(done)→mcp2(collided)→mcp3(planned)。"""
    bundle = _bundle(_manifest(mcp=[_mcp_comp("m1"), _mcp_comp("m2"), _mcp_comp("m3")]))
    store = _FakeSagaStore()
    config = _FakeConfig()
    svc = _make_svc(bundle, store=store, config=config)
    orig_create = store.create_install_skeleton

    async def _create(ctx):
        result = await orig_create(ctx)
        config.mcp["m2"] = SimpleNamespace()                 # m2 写入时碰撞
        return result

    store.create_install_skeleton = _create
    result = await _install(svc)
    assert result.status == "compensated"
    # 只 m1 被删（done）；m2 collided 零删；m3 planned 零删
    deleted = [c.ext_id for c in config.delete_calls if c.kind == "mcp"]
    assert deleted == ["m1"]
    assert "mcp_config:m2" in result.collided_targets


async def test_compensation_uses_delta_hooks_with_saga_ctx():
    """R30#2/R1#8：已写成员补偿删除经 delete hooks 带 UninstallContext(op.id, initiated_by)——
    每成员恰一条；成员行与父行均软删；末尾 plugin_expand_compensated。自失败注入 → op failed +
    父行保持 disabled（不软删）。"""
    bundle = _bundle(_manifest(mcp=[_mcp_comp("m1"), _mcp_comp("m2")]))
    store = _FakeSagaStore()
    config = _FakeConfig()
    svc = _make_svc(bundle, store=store, config=config)
    op_ids: list = []
    orig_create = store.create_install_skeleton

    async def _create(ctx):
        result = await orig_create(ctx)
        op_ids.append(result.operation_id)
        config.mcp["m2"] = SimpleNamespace()                 # m2 碰撞 → 补偿 m1
        return result

    store.create_install_skeleton = _create
    result = await _install(svc)
    assert result.status == "compensated"
    m1_deletes = [c for c in config.delete_calls if c.ext_id == "m1"]
    assert len(m1_deletes) == 1
    ctx_used = m1_deletes[0].uninstall_context
    assert ctx_used.correlation_id == op_ids[0]
    assert ctx_used.actor_user_id == "admin"
    assert m1_deletes[0].missing_ok is True
    assert store.parent_rows["org.test.pack"]["deleted"] is True
    assert all(r["deleted"] for r in store.member_rows.values())
    assert store.events()[-1] == "plugin_expand_compensated"

    # 自失败注入：delete hook raise → op failed + 父行保持 disabled（未软删）
    store2 = _FakeSagaStore()
    config2 = _FakeConfig()
    config2._delete_raises = True
    svc2 = _make_svc(_bundle(_manifest(mcp=[_mcp_comp("m1"), _mcp_comp("m2")])),
                     store=store2, config=config2)
    orig_create2 = store2.create_install_skeleton

    async def _create2(ctx):
        r = await orig_create2(ctx)
        config2.mcp["m2"] = SimpleNamespace()
        return r

    store2.create_install_skeleton = _create2
    result2 = await _install(svc2)
    assert result2.status == "failed"
    assert store2.parent_rows["org.test.pack"]["deleted"] is False
    assert store2.parent_rows["org.test.pack"]["status"] == "disabled"


async def test_publish_reverify_failure_compensates():
    """R33#4/§11.3 R34③：发布前复验 config fingerprint ≠ ctx pin → install_rejected
    (stage=publish_reverify) → 补偿序列。构造：写后带外篡改 config 令 fingerprint 漂移。"""
    bundle = _bundle(_manifest(mcp=[_mcp_comp("search", url="https://safe.test/mcp")]))
    store = _FakeSagaStore()
    config = _FakeConfig()
    svc = _make_svc(bundle, store=store, config=config)

    # 令 current_fingerprint 复验期返回漂移值（≠ preflight 观测）
    orig_fp = config.current_fingerprint
    config.current_fingerprint = lambda kind, ext_id: "sha256:DRIFTED"
    result = await _install(svc)
    config.current_fingerprint = orig_fp
    assert result.status == "compensated"
    stages = [a["details"].get("stage") for a in store.audits
              if a["event"] == "install_rejected" and a["details"]]
    assert "publish_reverify" in stages
    # 补偿删除已写 mcp
    assert any(c.ext_id == "search" for c in config.delete_calls)


async def test_publish_success_transaction():
    """复验过 → 父行 active + op completed + plugin_expand_completed（correlation 贯穿）；
    发布前全程父行 disabled（骨架，INV-D1-8）。"""
    bundle = _bundle(_manifest(mcp=[_mcp_comp("search")]))
    store = _FakeSagaStore()
    write_port = _FakeWritePort()
    svc = _make_svc(bundle, store=store, write_port=write_port)
    result = await _install(svc)
    assert isinstance(result, InstallResult)
    assert result.status == "completed"
    assert store.parent_rows["org.test.pack"]["status"] == "active"
    assert store.parent_rows["org.test.pack"]["deleted"] is False
    # correlation 贯穿：started + completed 同 op id
    op_id = store.skeleton_calls and next(iter(store.ops))
    started = next(a for a in store.audits if a["event"] == "plugin_expand_started")
    completed = next(a for a in store.audits if a["event"] == "plugin_expand_completed")
    assert started["correlation_id"] == completed["correlation_id"] == op_id
    # 父行 pin 经 write_port.record_install("plugin", ...)（disabled 骨架 update-only）
    assert any(c.kind == "plugin" and c.ext_id == "org.test.pack" for c in write_port.install_calls)


async def test_member_install_context_synthesis():
    """R25#2/R2#F1：per-member InstallContext = 成员观测/scan/provenance + actor=Admin +
    correlation=op.id + acknowledged/forced/probe_failed 取成员级字段（safe 成员三者 False）。"""
    bundle = _bundle(_manifest(mcp=[_mcp_comp("search")]))
    store = _FakeSagaStore()
    config = _FakeConfig()
    svc = _make_svc(bundle, store=store, config=config)
    await _install(svc)
    create = next(c for c in config.create_calls if c.kind == "mcp")
    ctx: InstallContext = create.install_context
    assert ctx.actor_user_id == "admin"
    assert ctx.correlation_id == next(iter(store.ops))
    assert ctx.source_type == "plugin"
    assert ctx.source_ref == "org.test.pack"
    assert ctx.trust_origin == "user_installed"
    # safe 成员：三 policy 事实字段恒 False（不伪产 acknowledged/force_installed）
    assert ctx.acknowledged is False and ctx.forced is False and ctx.probe_failed is False
    assert ctx.config_fingerprint is not None      # 观测 pin 逐字落


async def test_staging_rename_promotion(_saga_roots):
    """R50#3：skill/bundle 内容先落 PLUGIN_STAGING_ROOT/{op}/... 再 rename 进位。"""
    bundle = _bundle(_manifest(skills=[{"id": "web-clipper", "path": "skills/web-clipper"}]),
                     member_files=_skill_files())
    store = _FakeSagaStore()
    skill = _FakeSkillService()
    svc = _make_svc(bundle, store=store, skill=skill)
    result = await _install(svc)
    assert result.status == "completed"
    # bundle 进位到最终位（staging → rename）
    final_bundle = saga_mod.PLUGIN_STORE_ROOT / _bundle_key("org.test.pack", "1.0.0")
    assert (final_bundle / "plugin.json").exists()
    # skill install_skill 的 source_ref 位于 staging 根下（先 staging）
    assert len(skill.install_calls) == 1
    src = skill.install_calls[0].source_ref
    assert str(saga_mod.PLUGIN_STAGING_ROOT) in src


async def test_dry_run_returns_preview_zero_saga():
    """§8.3-1：dry_run → PluginInstallPreview，严格零 saga（store 零调用）。"""
    bundle = _bundle(_manifest(mcp=[_mcp_comp("search")]))
    store = _FakeSagaStore()
    svc = _make_svc(bundle, store=store)
    result = await _install(svc, dry_run=True)
    assert isinstance(result, PluginInstallPreview)
    assert store.skeleton_calls == []
    assert store.mark_calls == []


# ------------------------------------------------------ 硬化 #1/#2（防御守卫）-----
async def test_compensate_on_completed_op_is_noop():
    """硬化 #1（INV-D1-8 崩溃一致性守卫）：complete_install 已成功 COMMIT，但随后在 session
    teardown 抛错（COMMIT 后连接重置）→ install() 外层 ``except`` 携 **completed** op 进补偿。
    守卫令补偿为 no-op：成员**不删**、父行**不软删**、op 保持 completed、返回 completed 终态结果
    （不加守卫会拆掉一个已成功的安装 → 成功安装静默消失）。"""
    bundle = _bundle(_manifest(mcp=[_mcp_comp("search")]))
    store = _FakeSagaStore()
    config = _FakeConfig()
    svc = _make_svc(bundle, store=store, config=config)
    ctx = await _preflight_ctx(svc)
    skeleton = await store.create_install_skeleton(ctx)
    # 模拟成功安装既成事实：全部 step 标 done + op completed（父行翻 active）
    for seq in range(1, len(ctx.members) + 2):
        await store.mark_step(skeleton.operation_id, seq, "done")
    await store.complete_install(skeleton.operation_id)
    assert store.parent_rows["org.test.pack"]["status"] == "active"

    # complete_install 之后 teardown 抛错 → 外层 except 携 completed op 调 _compensate_install
    guard_result = await svc._compensate_install(
        ctx, skeleton, RuntimeError("connection reset after COMMIT"))

    assert config.delete_calls == []                                 # 成员未删
    assert store.parent_rows["org.test.pack"]["deleted"] is False    # 父行未软删
    assert store.parent_rows["org.test.pack"]["status"] == "active"  # 保持 active
    assert store.ops[skeleton.operation_id]["state"] == "completed"  # op 保持 completed
    assert guard_result is not None and guard_result.status == "completed"


async def test_preflight_malformed_source_ref_no_bare_valueerror():
    """R2b#3：畸形 source_ref（`https://[invalid` → urlsplit `Invalid IPv6 URL`）——
    ``canonicalize_source_ref`` 在 preflight try 块**之前**调用（line 323），修前抛裸
    ValueError 冒到全局 handler=500。修后 canonicalize 安全兜底，preflight 不再抛裸
    ValueError（fake loader 忽略 ref → dry_run 正常返回 preview；真 loader 路对畸形
    ref 抛 ValidationError∈_REJECT_EXCEPTIONS → 422，两路皆非 500）。"""
    bundle = _bundle(_manifest(mcp=[_mcp_comp("search")]))
    svc = _make_svc(bundle)
    # 不得抛 ValueError（畸形 URL 的 urlsplit 崩溃已被 canonicalize 吸收）
    result = await svc.preflight(
        SkillSourceType.GITHUB, "https://[invalid", actor_id="admin",
        force=False, acknowledge=False, dry_run=True)
    assert isinstance(result.preview, PluginInstallPreview)


async def test_result_of_none_op_returns_failed():
    """硬化 #2：``_result_of`` 遇 ``load_operation`` 返 None（op 行缺失/被删）→ 不解引用 None 崩
    ``AttributeError``，返回合同内 ``failed`` 终态结果。"""
    bundle = _bundle(_manifest(mcp=[_mcp_comp("search")]))
    store = _FakeSagaStore()
    svc = _make_svc(bundle, store=store)
    result = await svc._result_of(uuid.uuid4())     # 未知 op id → load_operation 返 None
    assert isinstance(result, InstallResult)
    assert result.status == "failed"
    assert result.collided_targets == []
