"""T16 — SkillService 治理挂钩单测（D1a §6.1-3 / §8.4 / §3.6）。

fake repo/ports：standalone 占用预检、二次 upsert 后 hook（实际磁盘 hash）、
hook 失败上抛不被吞、off 跳过、delete 三态 ctx、plugin 成员拒删、
InstallContext 字段投影、saga ctx 逐字、identity-lock 包裹治理路径。
"""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pytest

from app.application.services import skill_service as skill_service_module
from app.application.services.skill_service import SkillService
from app.domain.external.extension_admission import InstallContext, UninstallContext
from app.domain.models.extension_governance import (
    HASH_SCHEMA_VERSION,
    GovernanceScanSummary,
    InvalidStateTransitionError,
    ManagedByPluginError,
)
from app.domain.models.skill import SkillSourceType

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


_SKILL_MD = """---
name: Gov Demo
description: benign skill for governance hook test
---
# Gov Demo
Just a doc, nothing dangerous here.
"""


# ---------------------------------------------------------------- fakes ----
class _FakeSkillRepo:
    def __init__(
        self,
        events: list[str] | None = None,
        with_dir: bool = False,
        delete_returns: bool = True,
    ) -> None:
        self._items: dict = {}
        self._events = events
        self._with_dir = with_dir
        self._base = Path(tempfile.mkdtemp()) if with_dir else None
        self._delete_returns = delete_returns
        self.upsert_count = 0

    async def get_by_slug(self, slug: str):
        return None

    async def get_by_id(self, skill_id: str):
        return self._items.get(skill_id)

    async def list(self):
        return list(self._items.values())

    async def upsert(self, skill):
        self.upsert_count += 1
        if self._events is not None:
            self._events.append("upsert")
        self._items[skill.id] = skill
        if self._with_dir and self._base is not None:
            d = self._base / skill.id
            d.mkdir(parents=True, exist_ok=True)
            (d / "SKILL.md").write_text(
                (skill.manifest or {}).get("skill_md", "# x"), encoding="utf-8"
            )
        return skill

    def get_skill_dir(self, skill_id: str):
        if not self._with_dir or self._base is None:
            return None
        return self._base / skill_id

    async def delete(self, skill_id: str) -> bool:
        self._items.pop(skill_id, None)
        return self._delete_returns


class _FakeRow:
    def __init__(self, parent_plugin_ext_id: str | None = None, status: str = "active") -> None:
        self.parent_plugin_ext_id = parent_plugin_ext_id
        self.status = status


class _FakeReadPort:
    def __init__(self, row: _FakeRow | None = None, events: list[str] | None = None) -> None:
        self._row = row
        self.calls: list[tuple[str, str]] = []
        self._events = events

    async def get_row(self, kind: str, ext_id: str):
        if self._events is not None:
            self._events.append("precheck")
        self.calls.append((kind, ext_id))
        return self._row


class _FakeWritePort:
    def __init__(self, events: list[str] | None = None, fail: bool = False) -> None:
        self.installs: list[tuple] = []
        self.deletes: list[tuple] = []
        self._events = events
        self._fail = fail

    async def record_install(self, kind, ext_id, install_context) -> None:
        if self._events is not None:
            self._events.append("hook")
        self.installs.append((kind, ext_id, install_context))
        if self._fail:
            raise RuntimeError("record_install boom")

    async def record_delete(self, kind, ext_id, *, uninstall_context) -> None:
        self.deletes.append((kind, ext_id, uninstall_context))


class _FakeIdentityLocks:
    def __init__(self, events: list[str] | None = None) -> None:
        self.acquired: list = []
        self._events = events

    def acquire_all(self, identities):
        events = self._events
        acquired = self.acquired

        class _CM:
            async def __aenter__(self):
                acquired.append(list(identities))
                if events is not None:
                    events.append("acquire")

            async def __aexit__(self, *exc):
                if events is not None:
                    events.append("release")
                return False

        return _CM()


async def _install(service: SkillService, **kwargs):
    return await service.install_skill(
        source_type=SkillSourceType.LOCAL,
        source_ref="local:/tmp/gov-hook-demo",
        manifest={},
        skill_md=_SKILL_MD,
        installed_by="admin-1",
        **kwargs,
    )


# ---------------------------------------------------------------- cases ----
async def test_precheck_membership_409() -> None:
    repo = _FakeSkillRepo()
    read_port = _FakeReadPort(_FakeRow(parent_plugin_ext_id="plugin-x"))
    write_port = _FakeWritePort()
    svc = SkillService(repo, registry_write_port=write_port, registry_read_port=read_port)

    with pytest.raises(ManagedByPluginError):
        await _install(svc, actor_id="admin-1")
    assert repo.upsert_count == 0        # 磁盘零触碰
    assert write_port.installs == []


async def test_precheck_quarantined_disabled_409() -> None:
    for status in ("quarantined", "disabled"):
        repo = _FakeSkillRepo()
        read_port = _FakeReadPort(_FakeRow(status=status))
        svc = SkillService(repo, registry_write_port=_FakeWritePort(), registry_read_port=read_port)
        with pytest.raises(InvalidStateTransitionError):
            await _install(svc, actor_id="admin-1")
        assert repo.upsert_count == 0

    # active 行 → 放行（重装/更新语义）
    repo_ok = _FakeSkillRepo()
    read_ok = _FakeReadPort(_FakeRow(status="active"))
    svc_ok = SkillService(repo_ok, registry_write_port=_FakeWritePort(), registry_read_port=read_ok)
    result = await _install(svc_ok, actor_id="admin-1")
    assert result.id is not None


async def test_hook_after_second_upsert_with_actual_hash() -> None:
    events: list[str] = []
    repo = _FakeSkillRepo(events=events, with_dir=True)
    write_port = _FakeWritePort(events=events)
    svc = SkillService(repo, registry_write_port=write_port)

    result = await _install(svc, actor_id="admin-1")

    # 二次 upsert 后 hook；hash 为实际磁盘 hash（非 tmpdir）
    assert events == ["upsert", "upsert", "hook"]
    final_hash = (result.scan_report or {}).get("content_hash")
    assert final_hash is not None
    ctx = write_port.installs[0][2]
    assert ctx.artifact_hash == final_hash


async def test_hook_failure_propagates_not_swallowed() -> None:
    repo = _FakeSkillRepo()
    write_port = _FakeWritePort(fail=True)
    svc = SkillService(repo, registry_write_port=write_port)
    with pytest.raises(RuntimeError, match="record_install boom"):
        await _install(svc, actor_id="admin-1")


async def test_hook_skipped_when_port_none() -> None:
    repo = _FakeSkillRepo()
    svc = SkillService(repo)   # 全 None → off
    result = await _install(svc, actor_id="admin-1")
    assert result.id is not None
    assert repo.upsert_count >= 1


async def test_delete_skill_contexts() -> None:
    # standalone → 构造 ctx（correlation None + actor 非空）
    repo = _FakeSkillRepo()
    write_port = _FakeWritePort()
    svc = SkillService(repo, registry_write_port=write_port)
    await svc.delete_skill("s1", actor_id="admin-1")
    kind, ext_id, ctx = write_port.deletes[0]
    assert (kind, ext_id) == ("skill", "s1")
    assert ctx.correlation_id is None
    assert ctx.actor_user_id == "admin-1"

    # 传 uninstall_context → 逐字透传
    repo2 = _FakeSkillRepo()
    write_port2 = _FakeWritePort()
    svc2 = SkillService(repo2, registry_write_port=write_port2)
    saga_ctx = UninstallContext(correlation_id=uuid.uuid4(), actor_user_id="admin-2")
    await svc2.delete_skill("s2", uninstall_context=saga_ctx)
    assert write_port2.deletes[0][2] is saga_ctx

    # missing_ok=True + repo.delete False → 不抛且零 record_delete
    repo3 = _FakeSkillRepo(delete_returns=False)
    write_port3 = _FakeWritePort()
    svc3 = SkillService(repo3, registry_write_port=write_port3)
    await svc3.delete_skill("s3", actor_id="admin-1", missing_ok=True)
    assert write_port3.deletes == []


async def test_delete_plugin_member_refused() -> None:
    # standalone delete，行带 parent → ManagedByPluginError
    repo = _FakeSkillRepo()
    read_port = _FakeReadPort(_FakeRow(parent_plugin_ext_id="plugin-x"))
    write_port = _FakeWritePort()
    svc = SkillService(repo, registry_write_port=write_port, registry_read_port=read_port)
    with pytest.raises(ManagedByPluginError):
        await svc.delete_skill("s1", actor_id="admin-1")
    assert write_port.deletes == []

    # saga ctx（correlation 非 None）→ 守卫跳过、放行
    repo2 = _FakeSkillRepo()
    read_port2 = _FakeReadPort(_FakeRow(parent_plugin_ext_id="plugin-x"))
    write_port2 = _FakeWritePort()
    svc2 = SkillService(repo2, registry_write_port=write_port2, registry_read_port=read_port2)
    saga_ctx = UninstallContext(correlation_id=uuid.uuid4(), actor_user_id="admin-2")
    await svc2.delete_skill("s1", uninstall_context=saga_ctx)
    assert write_port2.deletes[0][2] is saga_ctx


async def test_install_context_fields() -> None:
    repo = _FakeSkillRepo(with_dir=True)
    write_port = _FakeWritePort()
    svc = SkillService(repo, registry_write_port=write_port)
    result = await _install(svc, actor_id="admin-1", trust_origin="agent_created")

    ctx = write_port.installs[0][2]
    assert isinstance(ctx, InstallContext)
    assert ctx.actor_user_id == "admin-1"
    assert ctx.correlation_id is None
    assert ctx.trust_origin == "agent_created"          # 透传（在 TRUST_ORIGINS 内）
    assert ctx.source_type == SkillSourceType.LOCAL      # skill 既有 source
    assert ctx.hash_schema_version == HASH_SCHEMA_VERSION
    # scan 投影为 GovernanceScanSummary（≤50 findings、无 match 原文字段）
    assert isinstance(ctx.scan, GovernanceScanSummary)
    assert len(ctx.scan.findings) <= 50
    for f in ctx.scan.findings:
        assert not hasattr(f, "match")
    _ = result


async def test_saga_override_context_verbatim() -> None:
    repo = _FakeSkillRepo()
    read_port = _FakeReadPort(_FakeRow(status="active"))
    write_port = _FakeWritePort()
    svc = SkillService(repo, registry_write_port=write_port, registry_read_port=read_port)

    saga_ctx = InstallContext(
        actor_user_id="saga-actor",
        correlation_id=uuid.uuid4(),
        source_type="plugin",
        source_ref="plugin:demo",
        version="9.9.9",
        trust_origin="user_installed",
        artifact_hash="sha256:preexisting-pin",
        surface_hash=None,
        config_fingerprint=None,
        hash_schema_version=HASH_SCHEMA_VERSION,
        scan=None,
    )
    await _install(svc, actor_id="admin-1", governance_install_context=saga_ctx)

    # hook 收到 ctx 逐字（pin/scan 不重算不覆盖）
    assert write_port.installs[0][2] is saga_ctx
    # saga 成员路 → standalone 占用预检被跳过（read_port.get_row 零调用）
    assert read_port.calls == []


async def test_identity_lock_wraps_governance_path(monkeypatch) -> None:
    # (a) 注入 fake locks：acquire → 预检 → upsert → hook → release
    events: list[str] = []
    repo = _FakeSkillRepo(events=events)
    read_port = _FakeReadPort(row=None, events=events)   # None row → 不抛、只记 precheck
    write_port = _FakeWritePort(events=events)
    locks = _FakeIdentityLocks(events=events)
    svc = SkillService(
        repo, registry_write_port=write_port, registry_read_port=read_port, identity_locks=locks
    )

    factory_calls = {"n": 0}

    def _spy_factory():
        factory_calls["n"] += 1
        raise AssertionError("factory must not be touched when identity_locks injected")

    monkeypatch.setattr(skill_service_module, "get_identity_locks", _spy_factory)

    await _install(svc, actor_id="admin-1")
    assert events == ["acquire", "precheck", "upsert", "hook", "release"]
    assert factory_calls["n"] == 0
    assert locks.acquired == [[("skill", svc_last_skill_id(repo))]]

    # (b) identity_locks=None + write_port 非 None → 走 get_identity_locks() 恰一次
    from app.application.services.extension_identity_locks import IdentityLockRegistry

    factory_calls_b = {"n": 0}
    real_reg = IdentityLockRegistry()

    def _spy_factory_b():
        factory_calls_b["n"] += 1
        return real_reg

    monkeypatch.setattr(skill_service_module, "get_identity_locks", _spy_factory_b)
    repo_b = _FakeSkillRepo()
    svc_b = SkillService(repo_b, registry_write_port=_FakeWritePort())
    await _install(svc_b, actor_id="admin-1")
    assert factory_calls_b["n"] == 1

    # (c) off（write_port=None）→ 零锁调用零 factory
    factory_calls_c = {"n": 0}

    def _spy_factory_c():
        factory_calls_c["n"] += 1
        raise AssertionError("off path must not touch factory")

    monkeypatch.setattr(skill_service_module, "get_identity_locks", _spy_factory_c)
    repo_c = _FakeSkillRepo()
    svc_c = SkillService(repo_c)   # off
    await _install(svc_c, actor_id="admin-1")
    assert factory_calls_c["n"] == 0


def svc_last_skill_id(repo: _FakeSkillRepo) -> str:
    return next(iter(repo._items))
