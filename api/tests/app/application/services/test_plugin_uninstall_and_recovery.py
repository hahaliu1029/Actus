"""T23 — Plugin uninstall saga（首步阻断 → 逐成员 forward 删 → 最终事务）+ 崩溃恢复链
（startup ②段 saga 收尾）+ failed-uninstall CAS 复活 的状态机/顺序合同。

fake store/services——**不**依赖真 PG（DB 事务保真度归 CI-only integration，见
``tests/integration/governance/test_plugin_saga_recovery_db.py``）。fake store 忠实建模
``begin_uninstall``/``finalize_uninstall``/``load_inflight_operations`` 的 DB 语义（与 T22
fake 同一手法：unit 测服务/模块编排，真 DB 保真度归 integration）。恢复路 ``_rollback_steps``
是 **T22 共享内核**（本文件通过 closure 真实驱动它——非重实现）。

fs 隔离：autouse fixture monkeypatch PLUGIN_*_ROOT → tmp。
"""
from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.application.services.plugin_install_service as saga_mod
from app.application.errors.exceptions import NotFoundError
from app.application.services.plugin_install_service import (
    PluginInstallService,
    _bundle_key,
    _sweep_staging_dirs,
    run_startup_saga_closure,
)
from app.domain.external.extension_admission import UninstallContext
from app.domain.models.extension_governance import (
    OperationPendingError,
    RevisionConflictError,
)
from app.domain.services.extension_hashing import entry_content_hash
from app.infrastructure.external.governance.plugin_saga_store import (
    build_uninstall_steps,
    make_step,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _saga_roots(tmp_path, monkeypatch):
    """每测试隔离 fs 根（staging/plugins/skills 同 tmp 下）。"""
    monkeypatch.setattr(saga_mod, "PLUGIN_STAGING_ROOT", tmp_path / ".plugin-staging")
    monkeypatch.setattr(saga_mod, "PLUGIN_STORE_ROOT", tmp_path / "plugins")
    monkeypatch.setattr(saga_mod, "SKILL_STORE_ROOT", tmp_path / "skills")
    return tmp_path


# ==================================================================== fakes =====
_TARGET_TYPE = {"skill": "skill_dir", "mcp": "mcp_config", "a2a": "a2a_config"}


class _NullCtx:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


class _FakeLocks:
    def __init__(self):
        self.acquired: list = []

    def acquire_all(self, identities):
        self.acquired.append(list(identities))
        return _NullCtx()


class _FakeConfig:
    """AppConfigService(写) + config_loader(读 occupied) 合一。"""

    def __init__(self, *, present=None):
        self.mcp: dict = {}
        self.a2a: dict = {}
        for kind, ext_id in (present or []):
            (self.mcp if kind == "mcp" else self.a2a)[ext_id] = SimpleNamespace()
        self.delete_calls: list = []

    async def delete_mcp_server(self, server_name, *, actor_id=None,
                                uninstall_context=None, missing_ok=False):
        self.delete_calls.append(SimpleNamespace(
            kind="mcp", ext_id=server_name, uninstall_context=uninstall_context,
            missing_ok=missing_ok))
        self.mcp.pop(server_name, None)

    async def delete_a2a_server(self, a2a_id, *, actor_id=None,
                                uninstall_context=None, missing_ok=False):
        self.delete_calls.append(SimpleNamespace(
            kind="a2a", ext_id=a2a_id, uninstall_context=uninstall_context,
            missing_ok=missing_ok))
        self.a2a.pop(a2a_id, None)

    # ---- config_loader 读面 ----
    def occupied(self, kind, ext_id):
        return ext_id in (self.mcp if kind == "mcp" else self.a2a)


class _FakeSkillService:
    def __init__(self):
        self.delete_calls: list = []

    async def delete_skill(self, skill_id, *, actor_id=None, uninstall_context=None,
                           missing_ok=False):
        self.delete_calls.append(SimpleNamespace(
            skill_id=skill_id, uninstall_context=uninstall_context, missing_ok=missing_ok))


class _FakeWritePort:
    def __init__(self):
        self.delete_calls: list = []

    async def record_delete(self, kind, ext_id, *, uninstall_context):
        self.delete_calls.append(SimpleNamespace(
            kind=kind, ext_id=ext_id, uninstall_context=uninstall_context))


class _FakeUninstallStore:
    """in-memory 忠实建模 begin/finalize/mark/fail/compensate/collision-audit + inflight。

    seed(plugin_ext_id, members, *, status, row_revision) 预置一个已安装 plugin。
    """

    def __init__(self):
        self.parents: dict = {}      # plugin_ext_id -> dict(status,row_revision,deleted,id,version)
        self.members: dict = {}      # plugin_ext_id -> list[(kind, ext_id, managed_by)]
        self.ops: dict = {}          # op_id -> dict
        self.audits: list = []

    # ---- seeding ----
    def seed(self, plugin_ext_id, members, *, status="active", row_revision=0, version="1.0.0"):
        self.parents[plugin_ext_id] = dict(
            status=status, row_revision=row_revision, deleted=False,
            id=uuid.uuid4(), version=version)
        self.members[plugin_ext_id] = list(members)   # (kind, ext_id, managed_by)
        return self.parents[plugin_ext_id]

    def seed_op(self, plugin_ext_id, *, operation_type, state, steps=None, error=None):
        op_id = uuid.uuid4()
        self.ops[op_id] = dict(
            id=op_id, operation_type=operation_type, plugin_ext_id=plugin_ext_id,
            plugin_extension_id=self.parents[plugin_ext_id]["id"] if plugin_ext_id in self.parents
            else uuid.uuid4(),
            initiated_by="admin", state=state, steps=steps or [], error=error)
        return op_id

    def _snap(self, op):
        return SimpleNamespace(
            id=op["id"], operation_type=op["operation_type"],
            plugin_extension_id=op["plugin_extension_id"], plugin_ext_id=op["plugin_ext_id"],
            initiated_by=op["initiated_by"], state=op["state"],
            steps=[dict(s) for s in op["steps"]], error=op["error"])

    # ---- begin_uninstall（首步单一事务建模）----
    async def begin_uninstall(self, plugin_ext_id, *, initiated_by, expected_row_revision):
        parent = self.parents.get(plugin_ext_id)
        if parent is None or parent["deleted"]:
            raise NotFoundError(f"plugin {plugin_ext_id} not found")
        pid = parent["id"]
        # 既有 in_progress → 409（部分唯一索引语义）
        if any(o["plugin_extension_id"] == pid and o["state"] == "in_progress"
               for o in self.ops.values()):
            raise OperationPendingError("plugin has in_progress operation")
        # 既有 failed uninstall → CAS 复活原 operation（保 id/steps，清 error）
        failed_un = [o for o in self.ops.values()
                     if o["plugin_extension_id"] == pid and o["state"] == "failed"
                     and o["operation_type"] == "plugin_uninstall"]
        if failed_un:
            op = failed_un[0]
            op["state"] = "in_progress"
            op["error"] = None
            return self._snap(op)
        # 新建路（failed install 不阻断）：父行 CAS-bump + active→disabled
        if parent["row_revision"] != expected_row_revision:
            raise RevisionConflictError("plugin row_revision mismatch")
        was_active = parent["status"] == "active"
        if was_active:
            parent["status"] = "disabled"
        parent["row_revision"] += 1
        managed = [(k, e) for (k, e, mg) in self.members.get(plugin_ext_id, []) if mg]
        steps = build_uninstall_steps(managed)
        op_id = uuid.uuid4()
        self.ops[op_id] = dict(
            id=op_id, operation_type="plugin_uninstall", plugin_ext_id=plugin_ext_id,
            plugin_extension_id=pid, initiated_by=initiated_by, state="in_progress",
            steps=steps, error=None)
        # disabled audit 在 op 建后发，correlation=op_id（与真 store + integration 同契约：
        # 该 disabled 迁移由本 uninstall op 因果驱动，须关联，不得留 NULL）
        if was_active:
            self.audits.append(dict(event="disabled", ext_id=plugin_ext_id,
                                    extension_id=pid, actor=initiated_by, correlation_id=op_id))
        return self._snap(self.ops[op_id])

    async def mark_step(self, operation_id, seq, state):
        for s in self.ops[operation_id]["steps"]:
            if s["seq"] == seq:
                s["state"] = state
                break

    async def finalize_uninstall(self, operation_id):
        op = self.ops[operation_id]
        parent = self.parents.get(op["plugin_ext_id"])
        if parent is not None and not parent["deleted"]:
            parent["deleted"] = True
            parent["row_revision"] += 1
            self.audits.append(dict(event="uninstalled", ext_id=op["plugin_ext_id"],
                                    extension_id=op["plugin_extension_id"],
                                    actor=op["initiated_by"], correlation_id=operation_id))
        self.members[op["plugin_ext_id"]] = []
        op["state"] = "completed"
        # 旧 failed install → compensated + error 追加 cleaned_up_via
        for o in self.ops.values():
            if (o["plugin_extension_id"] == op["plugin_extension_id"]
                    and o["operation_type"] == "plugin_install" and o["state"] == "failed"):
                o["state"] = "compensated"
                o["error"] = (o["error"] or "") + f" cleaned_up_via={operation_id}"

    async def fail(self, operation_id, *, error):
        self.ops[operation_id]["state"] = "failed"
        self.ops[operation_id]["error"] = error

    async def compensate(self, operation_id, *, details):
        op = self.ops[operation_id]
        parent = self.parents.get(op["plugin_ext_id"])
        if parent is not None:
            parent["deleted"] = True
        op["state"] = "compensated"
        collided = details.get("collided_targets") or []
        if collided:
            self.audits.append(dict(event="install_rejected", ext_id=op["plugin_ext_id"],
                                    correlation_id=operation_id,
                                    details={"stage": "recovery_collision",
                                             "collided_targets": collided}))
        self.audits.append(dict(event="plugin_expand_compensated", ext_id=op["plugin_ext_id"],
                                correlation_id=operation_id, details=None))

    async def record_content_collision_audit(self, operation_id, *, stage, member=None,
                                              target_key=None, collided_targets=None):
        details = {"stage": stage}
        if collided_targets is not None:
            details["collided_targets"] = collided_targets
        self.audits.append(dict(event="install_rejected",
                                ext_id=self.ops[operation_id]["plugin_ext_id"],
                                correlation_id=operation_id, details=details))

    async def load_operation(self, operation_id):
        op = self.ops.get(operation_id)
        return self._snap(op) if op else None

    async def load_inflight_operations(self):
        return [self._snap(o) for o in self.ops.values() if o["state"] == "in_progress"]

    async def sweep_terminal_staging(self):
        # 真 store 走 DB 查询；fake 走 op 注册表——共用 REAL _sweep_staging_dirs（fs 逻辑单源）
        _sweep_staging_dirs({o["id"]: o["state"] for o in self.ops.values()})

    def events(self):
        return [a["event"] for a in self.audits]

    def events_for(self, op_id):
        """镜像 integration ``_events_for``（``WHERE correlation_id == op_id``）——unit 与
        integration 对 correlation 契约同口径：仅关联到该 op 的 audit events。"""
        return [a["event"] for a in self.audits if a.get("correlation_id") == op_id]


# ------------------------------------------------------------------- harness ---
def _svc(store, *, config=None, skill=None, write_port=None, locks=None, mode="enforce"):
    config = config if config is not None else _FakeConfig()
    return PluginInstallService(
        loader=None, prober=None, read_port=None, mode=mode,
        identity_locks=locks or _FakeLocks(), store=store,
        skill_service=skill or _FakeSkillService(), app_config_service=config,
        write_port=write_port or _FakeWritePort(), config_loader=config)


# ==================================================================== tests =====
# ---- Step 2a: uninstall saga ----
async def test_uninstall_first_tx():
    """§8.3-6/R47#2 断言②：begin_uninstall 首步单一事务=op in_progress（steps 预写 planned）+
    父行 active→disabled（audit disabled）+ row_revision CAS-bump；已 disabled 父行保持原状态
    但仍 bump（并发 enable 持旧 revision 由此失效 409）。"""
    store = _FakeUninstallStore()
    parent = store.seed("org.test.pack", [("mcp", "search", True)], row_revision=5)
    op = await store.begin_uninstall("org.test.pack", initiated_by="admin", expected_row_revision=5)
    assert op.operation_type == "plugin_uninstall" and op.state == "in_progress"
    assert [s["state"] for s in op.steps] == ["planned"]
    assert op.steps[0]["target"] == {"type": "mcp_config", "key": "search"}
    assert parent["status"] == "disabled"                     # active → disabled
    assert parent["row_revision"] == 6                        # CAS-bump（并发旧 rev 必 409）
    # correlation 契约（与 integration test_begin_uninstall_first_tx 同口径）：disabled audit
    # 必须关联到本 uninstall op.id——非 correlation-filtered 的 events() 会漏检 NULL 关联缺陷。
    assert "disabled" in store.events_for(op.id)

    # 已 disabled 父行：保持状态但仍 bump（quarantined 同理，此处用 disabled）
    store2 = _FakeUninstallStore()
    p2 = store2.seed("org.two", [("mcp", "s", True)], status="disabled", row_revision=2)
    op2 = await store2.begin_uninstall("org.two", initiated_by="admin", expected_row_revision=2)
    assert p2["status"] == "disabled" and p2["row_revision"] == 3
    assert "disabled" not in store2.events()                  # 无迁移 → 不发 disabled audit
    assert op2.state == "in_progress"


async def test_uninstall_members_via_services():
    """§8.3-6：仅 managed_by_plugin=true 成员；mcp/a2a 经 delete_*(missing_ok=True,
    uninstall_context=ctx(op_id, actor))、skill 经 delete_skill；每成员恰一条（内容源在→delta 路）；
    每步 done 按 seq 更新不追加（R9#1）。"""
    store = _FakeUninstallStore()
    store.seed("org.test.pack",
               [("mcp", "search", True), ("a2a", "agent-x", True), ("skill", "clip", True),
                ("mcp", "unmanaged", False)],       # managed_by=false → 不删
               row_revision=0)
    config = _FakeConfig(present=[("mcp", "search"), ("a2a", "agent-x")])
    skill = _FakeSkillService()
    # skill 内容源在（目录存在）→ 走 delete_skill 而非缺源收尾
    (saga_mod.SKILL_STORE_ROOT / "clip").mkdir(parents=True)
    write_port = _FakeWritePort()
    svc = _svc(store, config=config, skill=skill, write_port=write_port)
    await svc.uninstall("org.test.pack", actor_id="admin", expected_row_revision=0)

    assert {c.ext_id for c in config.delete_calls} == {"search", "agent-x"}
    assert all(c.missing_ok is True for c in config.delete_calls)
    assert [c.skill_id for c in skill.delete_calls] == ["clip"]
    assert "unmanaged" not in {c.ext_id for c in config.delete_calls}     # managed_by=false 跳过
    # ctx 双字段（correlation=op.id + actor）
    op_id = next(iter(store.ops))
    for c in config.delete_calls:
        assert isinstance(c.uninstall_context, UninstallContext)
        assert c.uninstall_context.correlation_id == op_id
        assert c.uninstall_context.actor_user_id == "admin"
    # 内容源在 → 不走 write_port 缺源路
    assert write_port.delete_calls == []
    # op 收敛 completed（finalize）+ 父行软删
    assert store.ops[op_id]["state"] == "completed"
    assert store.parents["org.test.pack"]["deleted"] is True


async def test_missing_source_closure():
    """R22#F1：成员内容源已缺（重试/failed-install 未落盘）→ WritePort.record_delete 幂等收尾
    （传同一 ctx）→ 才标 done；service delete 不触（delta/hooks 不可达的显式例外）。"""
    store = _FakeUninstallStore()
    store.seed("org.gone", [("mcp", "ghost", True)], row_revision=0)
    config = _FakeConfig(present=[])                 # config 中无 ghost → 缺源
    write_port = _FakeWritePort()
    svc = _svc(store, config=config, write_port=write_port)
    await svc.uninstall("org.gone", actor_id="admin", expected_row_revision=0)

    assert [c.ext_id for c in write_port.delete_calls] == ["ghost"]    # WritePort 收尾
    assert config.delete_calls == []                                  # service delete 不触
    op_id = next(iter(store.ops))
    assert write_port.delete_calls[0].uninstall_context.correlation_id == op_id
    assert store.ops[op_id]["steps"][0]["state"] == "done"            # 收尾后才标 done
    assert store.ops[op_id]["state"] == "completed"


async def test_final_tx_converts_old_failed_install():
    """R52#4 断言③/R53#2：存在旧 failed plugin_install → uninstall 最终事务同事务把旧 op 转
    compensated 且 error 列追加 cleaned_up_via=<uninstall op id>（零 schema 变更）→ 无 failed 残留。"""
    store = _FakeUninstallStore()
    store.seed("org.test.pack", [("mcp", "search", True)], row_revision=0)
    config = _FakeConfig(present=[("mcp", "search")])
    old_failed = store.seed_op("org.test.pack", operation_type="plugin_install",
                               state="failed", error="compensation failed")
    svc = _svc(store, config=config)
    await svc.uninstall("org.test.pack", actor_id="admin", expected_row_revision=0)

    assert store.ops[old_failed]["state"] == "compensated"           # 旧 failed → compensated
    uninstall_op_id = next(o for o in store.ops if store.ops[o]["operation_type"] == "plugin_uninstall")
    assert f"cleaned_up_via={uninstall_op_id}" in store.ops[old_failed]["error"]
    # 无 failed 残留
    assert not any(o["state"] == "failed" for o in store.ops.values())


async def test_failed_uninstall_retry_cas_resurrect():
    """§3.4：failed uninstall → 重试=CAS 复活**原** operation（id/steps 保留，correlation 连续）；
    存在 failed uninstall 时同 plugin 禁新建（复活而非新建）；in_progress → 409；failed install
    的 plugin 允许新建 uninstall（唯一例外）。"""
    store = _FakeUninstallStore()
    store.seed("org.test.pack", [("mcp", "search", True)], status="disabled", row_revision=3)
    orig_steps = build_uninstall_steps([("mcp", "search")])
    orig = store.seed_op("org.test.pack", operation_type="plugin_uninstall",
                         state="failed", steps=[dict(s) for s in orig_steps],
                         error="uninstall step failed")
    svc = _svc(store)
    # retry 经同一 uninstall 入口 → begin_uninstall detect failed → CAS 复活原 op
    await svc.retry_failed_uninstall("org.test.pack", actor_id="admin", expected_row_revision=3)
    assert store.ops[orig]["state"] == "completed"           # 复活的原 op 走完 → completed
    # 复活=原 op（未新建第二个 uninstall op）
    uninstall_ops = [o for o in store.ops.values() if o["operation_type"] == "plugin_uninstall"]
    assert len(uninstall_ops) == 1 and uninstall_ops[0]["id"] == orig

    # in_progress → OperationPendingError（409）
    store2 = _FakeUninstallStore()
    store2.seed("org.p2", [("mcp", "s", True)], row_revision=0)
    store2.seed_op("org.p2", operation_type="plugin_uninstall", state="in_progress",
                   steps=[dict(s) for s in orig_steps])
    with pytest.raises(OperationPendingError):
        await store2.begin_uninstall("org.p2", initiated_by="admin", expected_row_revision=0)

    # failed install 的 plugin 允许新建 uninstall（唯一例外）
    store3 = _FakeUninstallStore()
    store3.seed("org.p3", [("mcp", "s", True)], row_revision=0)
    store3.seed_op("org.p3", operation_type="plugin_install", state="failed")
    op3 = await store3.begin_uninstall("org.p3", initiated_by="admin", expected_row_revision=0)
    assert op3.operation_type == "plugin_uninstall" and op3.state == "in_progress"


# ---- Step 2b: startup 收尾（run_startup_saga_closure）----
async def test_startup_closure_install_orphan():
    """§3.4 恢复算法：in_progress install 孤儿 → 逆序对 {attempting, done} 回滚（复用 T22
    _rollback_steps）→ compensated；planned 恒不删。此处 done→删、attempting(hash==expected)→删。"""
    store = _FakeUninstallStore()
    store.seed("org.test.pack", [("mcp", "m1", True)], status="disabled", row_revision=1)
    dump = {"transport": "streamable_http", "url": "https://safe.test/mcp"}
    steps = [
        make_step(1, "write_plugin_bundle", "plugin_bundle", "org.test.pack/1.0.0", "sha256:bundle"),
        make_step(2, "write_mcp", "mcp_config", "m1", entry_content_hash(dump)),
    ]
    steps[0]["state"] = "done"
    steps[1]["state"] = "done"
    op_id = store.seed_op("org.test.pack", operation_type="plugin_install",
                          state="in_progress", steps=steps)
    # bundle 目录存在（done 删）
    (saga_mod.PLUGIN_STORE_ROOT / _bundle_key("org.test.pack", "1.0.0")).mkdir(parents=True)
    config = _FakeConfig()
    config.mcp["m1"] = SimpleNamespace(model_dump=lambda mode=None: dump)   # config 现值 m1
    skill = _FakeSkillService()
    write_port = _FakeWritePort()

    def _entries(_path):
        return {"mcp": {"m1": dump}, "a2a": {}}

    import app.application.services.plugin_install_service as m
    orig = m._tolerant_load_config_entries
    m._tolerant_load_config_entries = _entries
    try:
        await run_startup_saga_closure(
            saga_store=store, app_config_service=config, skill_service=skill,
            write_port=write_port, config_path="/x/config.yaml")
    finally:
        m._tolerant_load_config_entries = orig

    assert store.ops[op_id]["state"] == "compensated"        # install 孤儿 → 补偿
    assert [c.ext_id for c in config.delete_calls] == ["m1"]  # done mcp 删


async def test_startup_closure_uninstall_orphan():
    """§3.4：in_progress uninstall 孤儿 → 标 failed（父级首步已阻断——不回滚）。"""
    store = _FakeUninstallStore()
    store.seed("org.test.pack", [("mcp", "m1", True)], status="disabled", row_revision=1)
    steps = build_uninstall_steps([("mcp", "m1")])
    op_id = store.seed_op("org.test.pack", operation_type="plugin_uninstall",
                          state="in_progress", steps=[dict(s) for s in steps])
    config = _FakeConfig()
    svc_write = _FakeWritePort()
    await run_startup_saga_closure(
        saga_store=store, app_config_service=config, skill_service=_FakeSkillService(),
        write_port=svc_write, config_path="/x/config.yaml")
    assert store.ops[op_id]["state"] == "failed"             # 标 failed（不回滚）
    assert config.delete_calls == []                         # 无成员删除（uninstall 孤儿不回滚）


async def test_startup_closure_orphan_isolation():
    """Fix #2：单孤儿终态迁移抛错须被隔离——不得阻断其余孤儿，更不得跳过末尾 staging sweep
    （§8.3-3 R51#3 "secrets 不过夜"）。构造两 uninstall 孤儿（boom 先/ok 后）：boom 的 fail() 抛
    RuntimeError → 断言 ok 仍被 fail + sweep 仍执行（ok 终态 staging 删、boom 仍 in_progress 保留）。"""
    store = _FakeUninstallStore()
    store.seed("org.boom", [("mcp", "m1", True)], status="disabled", row_revision=1)
    store.seed("org.ok", [("mcp", "m2", True)], status="disabled", row_revision=1)
    boom_op = store.seed_op("org.boom", operation_type="plugin_uninstall", state="in_progress",
                            steps=build_uninstall_steps([("mcp", "m1")]))
    ok_op = store.seed_op("org.ok", operation_type="plugin_uninstall", state="in_progress",
                          steps=build_uninstall_steps([("mcp", "m2")]))
    for op_id in (boom_op, ok_op):
        (saga_mod.PLUGIN_STAGING_ROOT / str(op_id)).mkdir(parents=True)

    # boom_op 的终态迁移（fail）抛错——模拟 DB 终态写失败
    orig_fail = store.fail

    async def _boom_fail(operation_id, *, error):
        if operation_id == boom_op:
            raise RuntimeError("terminal transition failed")
        await orig_fail(operation_id, error=error)

    store.fail = _boom_fail
    await run_startup_saga_closure(
        saga_store=store, app_config_service=_FakeConfig(), skill_service=_FakeSkillService(),
        write_port=_FakeWritePort(), config_path="/x/config.yaml")

    # boom 抛错被隔离（仍 in_progress，未推进），其余孤儿仍处理 + sweep 必达
    assert store.ops[boom_op]["state"] == "in_progress"       # 终态迁移失败 → 未推进
    assert store.ops[ok_op]["state"] == "failed"              # 隔离后其余孤儿仍处理
    assert not (saga_mod.PLUGIN_STAGING_ROOT / str(ok_op)).exists()    # sweep 必达：终态删
    assert (saga_mod.PLUGIN_STAGING_ROOT / str(boom_op)).exists()      # in_progress → 保留


async def test_config_unreadable_three_state():
    """R51#2/R52#3/R53#1：恢复读 config 失败 → config target 全 collided 不删 + 非 config target
    （skill 本地盘）照常三分回滚 + op 转 failed（非 compensated）+ recovery_collision audit 恰一条；
    收尾在 normal load 之前且不因不可读跳过（结构断言见 test_off_mode）。"""
    store = _FakeUninstallStore()
    store.seed("org.test.pack",
               [("mcp", "m1", True), ("skill", "clip", True)], status="disabled", row_revision=1)
    steps = [
        make_step(1, "write_plugin_bundle", "plugin_bundle", "org.test.pack/1.0.0", "sha256:bundle"),
        make_step(2, "write_mcp", "mcp_config", "m1", "sha256:mcp"),
        make_step(3, "write_skill", "skill_dir", "clip", None),
    ]
    for s in steps:
        s["state"] = "done"
    # skill 内容源在（本地盘）→ done 照删；bundle done 照删；mcp=config target → 不可读 collided
    (saga_mod.SKILL_STORE_ROOT / "clip").mkdir(parents=True)
    (saga_mod.PLUGIN_STORE_ROOT / _bundle_key("org.test.pack", "1.0.0")).mkdir(parents=True)
    op_id = store.seed_op("org.test.pack", operation_type="plugin_install",
                          state="in_progress", steps=steps)
    config = _FakeConfig()
    skill = _FakeSkillService()
    write_port = _FakeWritePort()

    import app.application.services.plugin_install_service as m
    orig = m._tolerant_load_config_entries
    m._tolerant_load_config_entries = lambda _p: None        # config 不可读
    try:
        await run_startup_saga_closure(
            saga_store=store, app_config_service=config, skill_service=skill,
            write_port=write_port, config_path="/x/config.yaml")
    finally:
        m._tolerant_load_config_entries = orig

    assert store.ops[op_id]["state"] == "failed"             # 非 compensated
    assert [c.skill_id for c in skill.delete_calls] == ["clip"]   # 非 config target 照删
    rc = [a for a in store.audits if a.get("details", {}).get("stage") == "recovery_collision"]
    assert len(rc) == 1                                      # recovery_collision 恰一条
    assert "mcp_config:m1" in rc[0]["details"]["collided_targets"]   # config target collided


async def test_staging_cleanup():
    """R50#3 断言③：终态（completed/compensated/failed）op 的 PLUGIN_STAGING_ROOT/{op_id}/
    被收尾整删；非终态（in_progress）保留。经 REAL _sweep_staging_dirs（fs 逻辑单源）。"""
    terminal = {uuid.uuid4(): "completed", uuid.uuid4(): "compensated", uuid.uuid4(): "failed"}
    live = uuid.uuid4()
    for op_id in list(terminal) + [live]:
        (saga_mod.PLUGIN_STAGING_ROOT / str(op_id)).mkdir(parents=True)
    states = dict(terminal)
    states[live] = "in_progress"
    _sweep_staging_dirs(states)
    for op_id in terminal:
        assert not (saga_mod.PLUGIN_STAGING_ROOT / str(op_id)).exists()   # 终态删
    assert (saga_mod.PLUGIN_STAGING_ROOT / str(live)).exists()            # 非终态保留


async def test_off_mode_closure_not_run():
    """§6.0/INV-D1-0：mode=off → run_startup_saga_closure 不被 lifespan 调用（结构断言：
    main.py ②段收尾在 `if _governance_mode != "off"` gate 内 + 位于 ①sweep 之后、
    normal `_load_app_config()` 之前）。"""
    main_src = Path(saga_mod.__file__).resolve().parents[3] / "app" / "main.py"
    src = main_src.read_text(encoding="utf-8")
    assert "run_startup_saga_closure" in src, "②段收尾未接线到 lifespan"
    # ②段收尾在 mode!=off gate 内
    idx_closure = src.index("run_startup_saga_closure")
    gate = src.rindex('if _governance_mode != "off":', 0, idx_closure)
    sweep = src.index("sweep_orphan_config_temps(")
    # ③段 normal load = lifespan 内首个 eager config load（不可读即抛 ServerRequestsError 中止 boot）
    normal_load = src.index("_app_config = _load_app_config()")
    # ①sweep < ②closure gate < ②closure < ③normal load（四段全序②在①之后、③之前，R52#2）
    assert sweep < gate < idx_closure < normal_load, "②段位序错（须①sweep 后、normal load 前）"


# ---- 9c: staging 对 B9 不可见 + 路径不在 skill 树下 ----
def test_plugin_staging_invisible_to_b9():
    """R3#14/spec R51#4：PLUGIN_STAGING_ROOT 不在 skill 根目录树下（纯路径 unit）——
    FileSkillRepository 枚举 skills_root 天然不可达 .plugin-staging（staging 非 skills 后代）。"""
    from app.application.services import plugin_install_service as real
    # 容纳当前（含测试 monkeypatch tmp）常量的树-containment 不变式：skills 非 staging 祖先
    staging = real.PLUGIN_STAGING_ROOT
    skills = real.SKILL_STORE_ROOT
    assert skills not in staging.parents, "staging 落在 skill 根树下——B9 枚举会误投影"
    assert staging != skills
    # 生产默认（源码级，不受 monkeypatch 影响）：staging=/app/data/.plugin-staging（skills 的兄弟）
    src = Path(real.__file__).read_text(encoding="utf-8")
    assert 'PLUGIN_STAGING_ROOT = Path("/app/data/.plugin-staging")' in src
    assert 'SKILL_STORE_ROOT = Path("/app/data/skills")' in src


# ---- _tolerant_load_config_entries 直测（load-bearing：test_config_unreadable 依赖 None 语义）----
def test_tolerant_load_config_entries_unreadable_returns_none(monkeypatch):
    """§8.3-2：读 config 失败（repo.load 抛）→ None（不抛）。"""
    import app.application.services.plugin_install_service as m

    class _BoomRepo:
        def __init__(self, path):
            pass

        def load(self):
            raise RuntimeError("config unreadable")

    monkeypatch.setattr(
        "app.infrastructure.repositories.file_app_config_repository.FileAppConfigRepository",
        _BoomRepo)
    assert m._tolerant_load_config_entries("/x/config.yaml") is None
