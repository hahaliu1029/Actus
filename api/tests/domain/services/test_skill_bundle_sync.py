from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.application.services.sandbox_accessors import EagerSandboxAccessor
from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType
from app.domain.models.tool_result import ToolResult
from app.domain.services.tools.skill_bundle_sync import SkillBundleSyncManager

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeSandbox:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.upload_paths: list[str] = []
        self.fail_upload_paths: set[str] = set()

    async def upload_file(self, file_data, filepath: str, filename: str | None = None) -> ToolResult:
        if filepath in self.fail_upload_paths:
            return ToolResult(success=False, message="upload failed")

        raw = file_data.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")

        self.files[filepath] = raw
        self.upload_paths.append(filepath)
        return ToolResult(success=True, data={"filepath": filepath, "filename": filename})

    async def write_file(
        self,
        filepath: str,
        content: str,
        append: bool = False,
        leading_newline: bool = False,
        trailing_newline: bool = False,
        sudo: bool = False,
    ) -> ToolResult:
        if leading_newline:
            content = "\n" + content
        if trailing_newline:
            content = content + "\n"
        if append and filepath in self.files:
            self.files[filepath] = self.files[filepath] + content.encode("utf-8")
        else:
            self.files[filepath] = content.encode("utf-8")
        return ToolResult(success=True, data={"filepath": filepath})

    async def check_file_exists(self, filepath: str) -> ToolResult:
        return ToolResult(success=True, data={"exists": filepath in self.files})

    async def read_file(
        self,
        filepath: str,
        start_line: int | None = None,
        end_line: int | None = None,
        sudo: bool = False,
        max_length: int = 10000,
    ) -> ToolResult:
        if filepath not in self.files:
            return ToolResult(success=False, message="not found")
        content = self.files[filepath].decode("utf-8")
        return ToolResult(success=True, data={"filepath": filepath, "content": content[:max_length]})



def _build_native_skill(skill_id: str, *, version: str, bundle_file_count: int = 2) -> Skill:
    return Skill(
        id=skill_id,
        slug=skill_id,
        name=skill_id,
        description="demo",
        source_type=SkillSourceType.LOCAL,
        source_ref=f"local:{skill_id}",
        runtime_type=SkillRuntimeType.NATIVE,
        manifest={
            "runtime_type": "native",
            "tools": [],
            "bundle_file_count": bundle_file_count,
            "last_sync_at": version,
        },
        enabled=True,
    )



def _write_bundle(skills_root: Path, skill_id: str) -> None:
    bundle_dir = skills_root / skill_id / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "SKILL.md").write_text("# Demo", encoding="utf-8")
    (bundle_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "scripts" / "run.py").write_text("print('ok')", encoding="utf-8")


async def test_initial_sync_uploads_bundle_and_writes_marker(tmp_path: Path) -> None:
    sandbox = _FakeSandbox()
    skill = _build_native_skill("pptx--1234", version="v1")
    _write_bundle(tmp_path, skill.id)

    manager = SkillBundleSyncManager(
        sandbox_accessor=EagerSandboxAccessor(sandbox),
        skills_root_dir=tmp_path,
        sandbox_skill_root="/home/ubuntu/workspace/.skills",
    )

    await manager.prepare_startup_sync(skill_pool=[skill], initial_selected=[skill])
    await manager.await_initial_sync()

    sandbox_dir, error = await manager.ensure_ready_for_invoke(skill.id)

    assert error is None
    assert sandbox_dir == f"/home/ubuntu/workspace/.skills/{skill.id}"
    assert f"{sandbox_dir}/SKILL.md" in sandbox.upload_paths
    assert f"{sandbox_dir}/scripts/run.py" in sandbox.upload_paths
    assert f"{sandbox_dir}/.actus-sync.json" in sandbox.files


async def test_same_version_marker_skips_reupload(tmp_path: Path) -> None:
    sandbox = _FakeSandbox()
    skill = _build_native_skill("pptx--1234", version="v1")
    _write_bundle(tmp_path, skill.id)

    manager_first = SkillBundleSyncManager(
        sandbox_accessor=EagerSandboxAccessor(sandbox),
        skills_root_dir=tmp_path,
        sandbox_skill_root="/home/ubuntu/workspace/.skills",
    )
    await manager_first.prepare_startup_sync(skill_pool=[skill], initial_selected=[skill])
    await manager_first.await_initial_sync()
    first_upload_count = len(sandbox.upload_paths)

    manager_second = SkillBundleSyncManager(
        sandbox_accessor=EagerSandboxAccessor(sandbox),
        skills_root_dir=tmp_path,
        sandbox_skill_root="/home/ubuntu/workspace/.skills",
    )
    await manager_second.prepare_startup_sync(skill_pool=[skill], initial_selected=[skill])
    await manager_second.await_initial_sync()

    assert len(sandbox.upload_paths) == first_upload_count


async def test_concurrent_ensure_ready_syncs_once(tmp_path: Path) -> None:
    sandbox = _FakeSandbox()
    skill = _build_native_skill("pptx--1234", version="v1")
    _write_bundle(tmp_path, skill.id)

    manager = SkillBundleSyncManager(
        sandbox_accessor=EagerSandboxAccessor(sandbox),
        skills_root_dir=tmp_path,
        sandbox_skill_root="/home/ubuntu/workspace/.skills",
    )
    await manager.prepare_startup_sync(skill_pool=[skill], initial_selected=[])

    await asyncio.gather(
        manager.ensure_ready_for_invoke(skill.id),
        manager.ensure_ready_for_invoke(skill.id),
    )

    assert sandbox.upload_paths.count(f"/home/ubuntu/workspace/.skills/{skill.id}/SKILL.md") == 1
    assert sandbox.upload_paths.count(f"/home/ubuntu/workspace/.skills/{skill.id}/scripts/run.py") == 1


async def test_sync_failure_is_reported_to_invoke_path(tmp_path: Path) -> None:
    sandbox = _FakeSandbox()
    skill = _build_native_skill("pptx--1234", version="v1")
    _write_bundle(tmp_path, skill.id)

    sandbox.fail_upload_paths.add(f"/home/ubuntu/workspace/.skills/{skill.id}/scripts/run.py")

    manager = SkillBundleSyncManager(
        sandbox_accessor=EagerSandboxAccessor(sandbox),
        skills_root_dir=tmp_path,
        sandbox_skill_root="/home/ubuntu/workspace/.skills",
    )
    await manager.prepare_startup_sync(skill_pool=[skill], initial_selected=[skill])
    await manager.await_initial_sync()

    sandbox_dir, error = await manager.ensure_ready_for_invoke(skill.id)

    assert sandbox_dir is None
    assert error
    assert "上传文件失败" in error


class TestFileListingCache:
    """Tests for post-sync file listing cache."""

    def test_get_file_listing_returns_none_before_sync(self):
        """Before sync, file listing should be None."""
        sandbox = _FakeSandbox()
        manager = SkillBundleSyncManager(
            sandbox_accessor=EagerSandboxAccessor(sandbox), skills_root_dir="/tmp/skills",
            sandbox_skill_root="/home/ubuntu/workspace/.skills",
        )
        assert manager.get_file_listing("nonexistent-id") is None

    def test_get_file_listing_all_returns_empty_before_sync(self):
        """Before sync, get_file_listing_all should return empty dict."""
        sandbox = _FakeSandbox()
        manager = SkillBundleSyncManager(
            sandbox_accessor=EagerSandboxAccessor(sandbox), skills_root_dir="/tmp/skills",
            sandbox_skill_root="/home/ubuntu/workspace/.skills",
        )
        assert manager.get_file_listing_all() == {}

    async def test_file_listing_populated_after_sync(self, tmp_path):
        """After successful sync, file listing should contain relative paths."""
        skills_root = tmp_path / "skills"
        _write_bundle(skills_root, "test-skill")

        sandbox = _FakeSandbox()
        manager = SkillBundleSyncManager(
            sandbox_accessor=EagerSandboxAccessor(sandbox), skills_root_dir=skills_root,
            sandbox_skill_root="/home/ubuntu/workspace/.skills",
        )
        skill = _build_native_skill("test-skill", version="v1")
        await manager.prepare_startup_sync([skill], [skill])
        await manager.await_initial_sync()

        listing = manager.get_file_listing("test-skill")
        assert listing is not None
        assert "SKILL.md" in listing
        assert "scripts/run.py" in listing

    async def test_file_listing_populated_on_version_match(self, tmp_path):
        """Cache should also be populated when sync skips upload (marker matches)."""
        skills_root = tmp_path / "skills"
        _write_bundle(skills_root, "cached-skill")

        sandbox = _FakeSandbox()
        manager = SkillBundleSyncManager(
            sandbox_accessor=EagerSandboxAccessor(sandbox), skills_root_dir=skills_root,
            sandbox_skill_root="/home/ubuntu/workspace/.skills",
        )
        skill = _build_native_skill("cached-skill", version="v1")

        # First sync: uploads files + writes marker
        await manager.prepare_startup_sync([skill], [skill])
        await manager.await_initial_sync()

        # Second manager: should hit version-match early return
        manager2 = SkillBundleSyncManager(
            sandbox_accessor=EagerSandboxAccessor(sandbox), skills_root_dir=skills_root,
            sandbox_skill_root="/home/ubuntu/workspace/.skills",
        )
        await manager2.prepare_startup_sync([skill], [skill])
        await manager2.await_initial_sync()

        listing2 = manager2.get_file_listing("cached-skill")
        assert listing2 is not None
        assert "SKILL.md" in listing2

    async def test_file_listing_empty_on_sync_failure(self, tmp_path):
        """When sync fails, file listing should remain None."""
        skills_root = tmp_path / "skills"
        _write_bundle(skills_root, "fail-skill")

        sandbox = _FakeSandbox()
        sandbox.fail_upload_paths.add(
            "/home/ubuntu/workspace/.skills/fail-skill/SKILL.md"
        )
        manager = SkillBundleSyncManager(
            sandbox_accessor=EagerSandboxAccessor(sandbox), skills_root_dir=skills_root,
            sandbox_skill_root="/home/ubuntu/workspace/.skills",
        )
        skill = _build_native_skill("fail-skill", version="v1")
        await manager.prepare_startup_sync([skill], [skill])
        await manager.await_initial_sync()

        assert manager.get_file_listing("fail-skill") is None


# ===========================================================================
# T12: G4b/G5 治理接线 + SyncOutcome 五态穷尽
# ===========================================================================


class _FakeAdmissionPort:
    """双面 fake：check_many（G4b/G5 前置状态门）+ verify_observation（G5 artifact）。"""

    def __init__(self, *, gate_reason=None, verify_admitted=True, mode="enforce"):
        self.mode = mode
        self._gate_reason = gate_reason          # None → admitted；str → blocked reason
        self._verify_admitted = verify_admitted
        self.check_calls: list = []
        self.verify_calls: list = []

    async def check_many(self, kind, ext_ids, config_fingerprints=None):
        from app.domain.external.extension_admission import AdmissionDecision
        self.check_calls.append((kind, tuple(ext_ids)))
        admitted = self._gate_reason is None
        reason = "ok" if admitted else self._gate_reason
        return {e: AdmissionDecision(admitted, reason, 1) for e in ext_ids}

    async def verify_observation(self, kind, ext_id, obs):
        from app.domain.external.extension_admission import AdmissionDecision
        self.verify_calls.append((kind, ext_id, obs))
        return AdmissionDecision(
            self._verify_admitted,
            "ok" if self._verify_admitted else "pin_mismatch",
            1,
        )


def _mgr(sandbox, tmp_path, **kw) -> SkillBundleSyncManager:
    return SkillBundleSyncManager(
        sandbox_accessor=EagerSandboxAccessor(sandbox),
        skills_root_dir=tmp_path,
        sandbox_skill_root="/home/ubuntu/workspace/.skills",
        **kw,
    )


async def test_governance_five_outcomes_exhaustive(tmp_path, monkeypatch) -> None:
    """_sync_bundle 每条路径穷尽覆盖 get_args(SyncOutcome) 全五值。"""
    from typing import get_args

    import app.domain.services.tools.skill_bundle_sync as sbs
    from app.domain.models.extension_governance import SyncOutcome

    observed: set[str] = set()

    # no_bundle：bundle_file_count=0（gate 之前的 short-circuit）
    nb = _build_native_skill("nb--1", version="v1", bundle_file_count=0)
    observed.add((await _mgr(_FakeSandbox(), tmp_path)._sync_bundle(nb)).outcome)

    # uploaded（fresh）→ already_current（marker 命中），共用同一 sandbox+manager
    up = _build_native_skill("up--1", version="v1")
    _write_bundle(tmp_path, up.id)
    up_mgr = _mgr(_FakeSandbox(), tmp_path, admission_port=_FakeAdmissionPort())
    observed.add((await up_mgr._sync_bundle(up)).outcome)     # uploaded
    observed.add((await up_mgr._sync_bundle(up)).outcome)     # already_current

    # r3_rejected_old_bundle：强制 R3 scan block（port=None，G5 verify 直通）
    monkeypatch.setattr(sbs, "get_install_decision", lambda *a, **k: "block")
    r3 = _build_native_skill("r3--1", version="v1")
    _write_bundle(tmp_path, r3.id)
    observed.add((await _mgr(_FakeSandbox(), tmp_path)._sync_bundle(r3)).outcome)

    # governance_rejected：gate 阻断
    gov = _build_native_skill("gov--1", version="v1")
    _write_bundle(tmp_path, gov.id)
    gov_mgr = _mgr(_FakeSandbox(), tmp_path,
                   admission_port=_FakeAdmissionPort(gate_reason="quarantined"))
    observed.add((await gov_mgr._sync_bundle(gov)).outcome)

    assert observed == set(get_args(SyncOutcome))


async def test_governance_rejected_gate_zero_sandbox_writes(tmp_path) -> None:
    sandbox = _FakeSandbox()
    skill = _build_native_skill("gov--z", version="v1")
    _write_bundle(tmp_path, skill.id)
    port = _FakeAdmissionPort(gate_reason="quarantined")
    result = await _mgr(sandbox, tmp_path, admission_port=port)._sync_bundle(skill)

    assert result.outcome == "governance_rejected"
    assert result.synced_dir is None
    assert result.error == "quarantined"
    assert sandbox.upload_paths == []    # 零上传
    assert sandbox.files == {}           # 零 marker 写


async def test_governance_rejected_artifact_verify_zero_writes(tmp_path) -> None:
    # gate 放行、artifact verify 失败 → governance_rejected（拒写保留旧盘）
    sandbox = _FakeSandbox()
    skill = _build_native_skill("gov--v", version="v1")
    _write_bundle(tmp_path, skill.id)
    port = _FakeAdmissionPort(gate_reason=None, verify_admitted=False)
    result = await _mgr(sandbox, tmp_path, admission_port=port)._sync_bundle(skill)

    assert result.outcome == "governance_rejected"
    assert sandbox.upload_paths == []


async def test_ensure_ready_maps_governance_rejected_to_unavailable(tmp_path) -> None:
    sandbox = _FakeSandbox()
    skill = _build_native_skill("gov--e", version="v1")
    _write_bundle(tmp_path, skill.id)
    port = _FakeAdmissionPort(gate_reason="quarantined")
    manager = _mgr(sandbox, tmp_path, admission_port=port)

    sandbox_dir, error = await manager.ensure_ready_for_invoke(skill.id, skill=skill)

    assert sandbox_dir is None
    assert error is not None
    assert "extension_unavailable" in error and "quarantined" in error
    state = manager._sync_states[skill.id]
    assert state.outcome == "governance_rejected"   # 簿记含 outcome 终值
    assert state.status == "success"                # 非 except 路径


async def test_ensure_ready_uploaded_dir_nonnull_and_outcome(tmp_path) -> None:
    sandbox = _FakeSandbox()
    skill = _build_native_skill("ok--1", version="v1")
    _write_bundle(tmp_path, skill.id)
    manager = _mgr(sandbox, tmp_path, admission_port=_FakeAdmissionPort())

    sandbox_dir, error = await manager.ensure_ready_for_invoke(skill.id, skill=skill)

    assert error is None
    assert sandbox_dir == f"/home/ubuntu/workspace/.skills/{skill.id}"
    assert manager._sync_states[skill.id].outcome == "uploaded"


async def test_off_mode_byte_identical_outcomes(tmp_path) -> None:
    # admission_port=None：gate/verify no-op；outcome 仅落既有行为路径，ensure_ready 不变
    sandbox = _FakeSandbox()
    skill = _build_native_skill("off--1", version="v1")
    _write_bundle(tmp_path, skill.id)
    manager = _mgr(sandbox, tmp_path)   # 无 admission_port

    r1 = await manager._sync_bundle(skill)
    assert r1.outcome == "uploaded" and r1.synced_dir.endswith(skill.id)
    r2 = await manager._sync_bundle(skill)
    assert r2.outcome == "already_current"

    sandbox_dir, error = await manager.ensure_ready_for_invoke(skill.id, skill=skill)
    assert error is None and sandbox_dir.endswith(skill.id)


async def test_sync_failure_outcome_terminal_none(tmp_path) -> None:
    # 硬失败（upload raise）走 status=failed 通道，outcome 收敛为 None（非五态治理 outcome）
    sandbox = _FakeSandbox()
    skill = _build_native_skill("failx--1", version="v1")
    _write_bundle(tmp_path, skill.id)
    sandbox.fail_upload_paths.add(
        f"/home/ubuntu/workspace/.skills/{skill.id}/scripts/run.py"
    )
    manager = _mgr(sandbox, tmp_path)
    await manager.prepare_startup_sync([skill], [skill])
    await manager.await_initial_sync()

    state = manager._sync_states[skill.id]
    assert state.status == "failed"
    assert state.outcome is None


# ===========================================================================
# SPM Task 16: native skill bundle deferred sync mode (provision hook ②)
# ===========================================================================


class _CountingAccessor:
    """SandboxAccessor whose ``get()`` is FORBIDDEN inside deferred sync — in
    production it would re-enter the inflight ``provisioner._provision_once``
    and DEADLOCK. We count calls so tests can assert the anti-deadlock contract:
    the deferred path must operate on the bound CONCRETE handle, never
    ``accessor.get()``. Eager mode legitimately uses ``get()`` (byte-identical).
    """

    def __init__(self, handle: _FakeSandbox) -> None:
        self._handle = handle
        self.gets = 0

    async def get(self) -> _FakeSandbox:
        self.gets += 1
        return self._handle

    def peek(self) -> _FakeSandbox | None:
        return self._handle

    async def release_owned(self) -> None:
        return None


class _TaskSpawnCounter:
    """Wraps ``asyncio.create_task`` and counts only the manager's own sync
    tasks (filtered by coroutine ``__qualname__``) so foreign event-loop tasks
    do not pollute the count. Delegates every call to the real factory."""

    def __init__(self, real) -> None:
        self._real = real
        self.count = 0

    def __call__(self, coro, *args, **kwargs):
        qualname = getattr(coro, "__qualname__", "") or ""
        if "SkillBundleSyncManager" in qualname:
            self.count += 1
        return self._real(coro, *args, **kwargs)


class _SyncEnv:
    def __init__(self, tmp_path, handle, accessor, pool, counter) -> None:
        self.tmp_path = tmp_path
        self.handle = handle
        self.accessor = accessor
        self.pool = pool
        self._counter = counter
        self.mgr: SkillBundleSyncManager | None = None

    def make_manager(self, *, deferred: bool) -> SkillBundleSyncManager:
        self.mgr = SkillBundleSyncManager(
            sandbox_accessor=self.accessor,
            skills_root_dir=self.tmp_path,
            sandbox_skill_root="/home/ubuntu/workspace/.skills",
            deferred=deferred,
        )
        return self.mgr

    def spawned_tasks(self) -> int:
        return self._counter.count


@pytest.fixture
def sync_env(tmp_path: Path, monkeypatch) -> _SyncEnv:
    import app.domain.services.tools.skill_bundle_sync as sbs

    handle = _FakeSandbox()
    accessor = _CountingAccessor(handle)
    pool: list[Skill] = []
    for sid in ("alpha--1", "beta--1"):
        skill = _build_native_skill(sid, version="v1")
        _write_bundle(tmp_path, sid)
        pool.append(skill)

    counter = _TaskSpawnCounter(sbs.asyncio.create_task)
    monkeypatch.setattr(sbs.asyncio, "create_task", counter)
    return _SyncEnv(tmp_path, handle, accessor, pool, counter)


class TestDeferredMode:
    async def test_deferred_prepare_creates_no_tasks(self, sync_env: _SyncEnv) -> None:
        mgr = sync_env.make_manager(deferred=True)
        await mgr.prepare_startup_sync(
            skill_pool=sync_env.pool, initial_selected=sync_env.pool[:1]
        )
        # deferred prepare stores intent ONLY — zero asyncio.create_task
        assert sync_env.spawned_tasks() == 0
        # await_initial_sync returns immediately (does not hang / does not sync)
        await mgr.await_initial_sync()
        assert sync_env.spawned_tasks() == 0
        assert sync_env.accessor.gets == 0
        # start_background_sync is a no-op until the sandbox is provisioned
        mgr.start_background_sync()
        assert sync_env.spawned_tasks() == 0
        # nothing was uploaded to the sandbox yet
        assert sync_env.handle.upload_paths == []

    async def test_start_deferred_sync_runs_stored_intent_idempotently(
        self, sync_env: _SyncEnv
    ) -> None:
        mgr = sync_env.make_manager(deferred=True)
        await mgr.prepare_startup_sync(
            skill_pool=sync_env.pool, initial_selected=sync_env.pool[:1]
        )
        # provision hook ②: concrete handle in (anti-deadlock contract)
        await mgr.start_deferred_sync(sync_env.handle)
        n = sync_env.spawned_tasks()
        assert n > 0
        # the sync path used the bound concrete handle, NEVER accessor.get()
        assert sync_env.accessor.gets == 0
        # real work happened against the concrete handle
        assert sync_env.handle.upload_paths
        # second call is a no-op (idempotent) — spawns no new tasks
        await mgr.start_deferred_sync(sync_env.handle)
        assert sync_env.spawned_tasks() == n
        assert sync_env.accessor.gets == 0
        await mgr.cleanup()

    async def test_eager_mode_unchanged(self, sync_env: _SyncEnv) -> None:
        mgr = sync_env.make_manager(deferred=False)
        await mgr.prepare_startup_sync(
            skill_pool=sync_env.pool, initial_selected=sync_env.pool[:1]
        )
        # eager prepare eagerly creates the foreground sync task(s)
        assert sync_env.spawned_tasks() > 0
        await mgr.await_initial_sync()
        await mgr.cleanup()

    # ── SPM Task 17 fix #1(b): deferred+unbound _acquire_sandbox raises loud ──

    async def test_acquire_sandbox_deferred_unbound_raises(
        self, sync_env: _SyncEnv
    ) -> None:
        """A manager-spawned sync task is only ever created AFTER
        start_deferred_sync bound a handle. Reaching _acquire_sandbox with
        deferred=True and no bound handle is the exact pre-fix deadlock ordering;
        it must raise loudly (RuntimeError) instead of silently re-entering the
        inflight provision via accessor.get()."""
        mgr = sync_env.make_manager(deferred=True)
        # never called start_deferred_sync → _bound_handle is None
        with pytest.raises(RuntimeError, match="deferred skill sync"):
            await mgr._acquire_sandbox()
        assert sync_env.accessor.gets == 0  # never fell through to accessor.get()

    async def test_always_mode_acquire_sandbox_uses_accessor(
        self, sync_env: _SyncEnv
    ) -> None:
        """always (non-deferred) mode keeps the byte-identical accessor.get()
        path (INV-SPM-2) — fix #1(b) only guards the deferred branch."""
        mgr = sync_env.make_manager(deferred=False)
        handle = await mgr._acquire_sandbox()
        assert handle is sync_env.handle
        assert sync_env.accessor.gets == 1


class TestOnDemandProvisionNoDeadlock:
    """SPM Task 17 fix #1: the on_demand provision flow (get → hook ② →
    start_deferred_sync) must COMPLETE — hook ② binds the concrete handle so the
    deferred sync uses it, never accessor.get() (which re-enters the SAME inflight
    provision → circular deadlock)."""

    async def test_provision_get_completes_within_timeout(
        self, sync_env: _SyncEnv
    ) -> None:
        from app.application.services.sandbox_provisioner import SandboxProvisioner
        from app.domain.errors.sandbox_lifecycle import SessionUnboundError

        mgr = sync_env.make_manager(deferred=True)
        await mgr.prepare_startup_sync(
            skill_pool=sync_env.pool, initial_selected=sync_env.pool[:1]
        )

        class _Lifecycle:
            async def acquire(self, sid):
                raise SessionUnboundError(sid)  # → bind_new (no ensure_sandbox)

            async def bind_new(self, sid, *, user_id=None):
                return sync_env.handle

        provisioner = SandboxProvisioner(
            session_id="s-1",
            user_id="owner-1",
            lifecycle=_Lifecycle(),
            # hook ② mirrors AgentService: concrete handle → start_deferred_sync
            hooks=[("skill_sync", lambda h: mgr.start_deferred_sync(h))],
            timeout_seconds=5,
        )
        # Would HANG (wait_for timeout) if the deferred sync re-entered the
        # inflight via accessor.get(); completes because it uses the bound handle.
        handle = await asyncio.wait_for(provisioner.get(), timeout=5)
        assert handle is sync_env.handle
        assert sync_env.accessor.gets == 0  # bound handle used, no re-entry
        assert sync_env.handle.upload_paths  # deferred sync really ran
        await mgr.cleanup()
