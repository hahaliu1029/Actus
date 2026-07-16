"""Skill bundle synchronization manager for sandbox runtime."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from app.domain.external.sandbox import SandboxAccessor, SandboxHandle
from app.domain.models.extension_governance import SyncOutcome
from app.domain.models.skill import Skill, SkillRuntimeType
from app.domain.services.extension_admission_gates import (
    skill_invoke_gate,
    verify_skill_artifact,
)
from app.domain.services.skills_guard import SkillsGuard, ScanReport
from app.domain.services.trust_matrix import scan_skill_source, get_install_decision

logger = logging.getLogger(__name__)

SYNC_STATUS = Literal["pending", "running", "success", "failed"]
SYNC_MARKER_FILENAME = ".actus-sync.json"
DEFAULT_BACKGROUND_CONCURRENCY = 4


@dataclass
class SkillSyncState:
    status: SYNC_STATUS = "pending"
    version: str = ""
    sandbox_dir: str = ""
    error: str | None = None
    task: asyncio.Task[str | None] | None = None
    outcome: SyncOutcome | None = None   # D1a §4.2: 治理同步 outcome（追加键，旧读者零感知）


@dataclass(frozen=True)
class BundleSyncResult:
    """D1a §4.2 _sync_bundle typed 结果：SyncOutcome 五态全路径穷尽赋值。"""

    outcome: SyncOutcome
    synced_dir: str | None
    error: str | None


class SkillBundleSyncManager:
    """Sync skill bundle files from API filesystem into sandbox filesystem."""

    def __init__(
        self,
        sandbox_accessor: SandboxAccessor,
        skills_root_dir: str | Path,
        sandbox_skill_root: str,
        background_concurrency: int = DEFAULT_BACKGROUND_CONCURRENCY,
        admission_port: Any = None,  # D1a §4.1: ExtensionAdmissionPort | None（off=None → 旧路径零调用）
        deferred: bool = False,  # SPM Task 16: on_demand 档懒同步（default False = always 字节等价）
    ) -> None:
        self._sandbox_accessor = sandbox_accessor
        self._skills_root_dir = Path(skills_root_dir)
        self.sandbox_skill_root = str(sandbox_skill_root).rstrip("/")
        self._background_concurrency = max(1, int(background_concurrency or 1))
        self._admission_port = admission_port  # D1a §4.1: G 逻辑在 T10-T12 消费
        self._skill_pool: dict[str, Skill] = {}
        self._sync_states: dict[str, SkillSyncState] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._initial_tasks: list[asyncio.Task[str | None]] = []
        self._background_task: asyncio.Task[None] | None = None
        self._background_skills: list[Skill] = []
        self._file_listings: dict[str, list[str]] = {}
        # SPM Task 16: deferred (on_demand) mode. When True, prepare_startup_sync
        # only STORES the intent (zero create_task); the real task creation +
        # background start happen in start_deferred_sync(handle), wired as
        # provision hook ② (Task 17). All fields inert on the always path.
        self._deferred = bool(deferred)
        self._bound_handle: SandboxHandle | None = None  # concrete handle (anti-deadlock)
        self._deferred_started = False  # start_deferred_sync idempotency guard
        self._stored_skill_pool: list[Skill] = []
        self._stored_initial_selected: list[Skill] = []

    async def prepare_startup_sync(
        self,
        skill_pool: list[Skill],
        initial_selected: list[Skill],
    ) -> None:
        """Prepare startup synchronization tasks.

        Initial selected skills are synchronized in foreground (blocking).
        Remaining syncable skills are prepared for background sync.

        SPM Task 16: in deferred (on_demand) mode this STORES the intent only —
        zero ``asyncio.create_task`` — because the sandbox is not yet provisioned.
        The real task creation runs in ``start_deferred_sync(handle)`` (provision
        hook ②). Always mode is byte-equivalent to before (INV-SPM-2).
        """
        self._skill_pool = {skill.id: skill for skill in skill_pool}

        if self._deferred:
            self._stored_skill_pool = list(skill_pool)
            self._stored_initial_selected = list(initial_selected)
            return

        self._create_startup_tasks(skill_pool, initial_selected)

    def _create_startup_tasks(
        self,
        skill_pool: list[Skill],
        initial_selected: list[Skill],
    ) -> None:
        """Foreground/background split of the startup sync (eager body).

        Shared by the always path (``prepare_startup_sync``) and the deferred
        path (``start_deferred_sync``). Assumes ``self._skill_pool`` is already
        populated by the caller.
        """
        selected_ids = {skill.id for skill in initial_selected}

        self._initial_tasks = []
        self._background_skills = []

        for skill in skill_pool:
            if not self._needs_sync(skill):
                continue
            if skill.id in selected_ids:
                task = self._ensure_sync_task(skill)
                self._initial_tasks.append(task)
            else:
                self._background_skills.append(skill)

    async def await_initial_sync(self) -> None:
        """Wait startup foreground sync tasks to complete."""
        if not self._initial_tasks:
            return
        results = await asyncio.gather(*self._initial_tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.warning("前台Skill bundle同步任务异常: %s", str(result))

    def start_background_sync(self) -> None:
        """Start background synchronization for remaining skills.

        SPM Task 16: in deferred mode, the startup-sequence call (runner :4785)
        arrives BEFORE the sandbox is provisioned, so it is a no-op — the real
        background start happens inside ``start_deferred_sync`` (which always
        starts the sweep, mirroring the always sequence). Always mode is
        byte-equivalent to before.
        """
        if self._deferred and not self._deferred_started:
            # No-op until the sandbox is provisioned: start_deferred_sync always
            # starts the background sweep at the end (mirrors the always
            # sequence), so no pending flag is needed here.
            return
        self._start_background_sync_impl()

    def _start_background_sync_impl(self) -> None:
        if self._background_task or not self._background_skills:
            return
        self._background_task = asyncio.create_task(self._run_background_sync())

    async def start_deferred_sync(self, handle: SandboxHandle) -> None:
        """SPM Task 16 (provision hook ②): run the stored startup sync using a
        CONCRETE handle.

        CRUX — anti-deadlock: this runs INSIDE ``provisioner._provision_once()``
        while the binding is not yet ACTIVE. Any path that did
        ``await self._sandbox_accessor.get()`` here would re-enter (join) the
        same inflight provision task and DEADLOCK. We therefore bind the concrete
        ``handle`` first; every sandbox upload/marker op then prefers it (see
        ``_acquire_sandbox``). Background tasks that run after the binding is
        ready would also fast-return from the accessor, but they too keep using
        the bound handle here.

        Idempotent: a second call is a no-op. Never raises lifecycle errors
        (``SessionSuspendedError`` / ``SessionFinalizedError``); a cooperative
        ``CancelledError`` still propagates. Sync failures follow the manager's
        existing log/degrade style (``await_initial_sync`` gathers with
        ``return_exceptions``; ``_sync_skill`` swallows and marks failed), so the
        provision hook wrapper never sees a ``SessionSuspendedError`` /
        ``SessionFinalizedError`` leak out of here.
        """
        if self._deferred_started:
            return
        self._deferred_started = True
        # bind_ready_handle: record the concrete handle so sandbox ops use it
        # directly instead of joining the inflight provision (see docstring).
        self._bound_handle = handle
        # Now replay the prepare that deferred mode only stored, then really
        # start the background sweep (mirrors always: prepare → await → background).
        self._create_startup_tasks(
            self._stored_skill_pool, self._stored_initial_selected
        )
        await self.await_initial_sync()
        self._start_background_sync_impl()

    def get_file_listing(self, skill_id: str) -> list[str] | None:
        """Return cached file listing for a skill, or None if not cached.

        Returns a defensive copy to prevent external mutation.
        """
        listing = self._file_listings.get(skill_id)
        return list(listing) if listing is not None else None

    def get_file_listing_all(self) -> dict[str, list[str]]:
        """Return all cached file listings. Keys are skill IDs, values are relative path lists.

        Returns a deep copy to prevent external mutation of cached data.
        """
        return {k: list(v) for k, v in self._file_listings.items()}

    async def ensure_ready_for_invoke(
        self,
        skill_id: str,
        *,
        skill: Skill | None = None,
    ) -> tuple[str | None, str | None]:
        """Ensure a skill bundle is synchronized before native invoke."""
        resolved_skill = skill or self._skill_pool.get(skill_id)
        if resolved_skill is None:
            return None, f"Skill[{skill_id}]未在当前会话同步池中"

        if not self._needs_sync(resolved_skill):
            return None, None

        self._skill_pool[resolved_skill.id] = resolved_skill
        task = self._ensure_sync_task(resolved_skill)
        await task

        state = self._sync_states.get(resolved_skill.id)
        if not state:
            return None, f"Skill[{resolved_skill.id}]同步状态缺失"
        if state.status == "failed":
            return None, state.error or f"Skill[{resolved_skill.id}]同步失败"
        if state.status != "success":
            return None, f"Skill[{resolved_skill.id}]同步未完成"
        # D1a §4.2 G4b/G5：治理拒绝映射为 extension_unavailable（非硬同步失败；
        # off-mode port=None 永不产出此 outcome → 分支零命中，(None,None) 语义不变）
        if state.outcome == "governance_rejected":
            return None, f"extension_unavailable: {state.error}"
        return state.sandbox_dir or None, None

    async def cleanup(self) -> None:
        """Cancel and cleanup pending synchronization tasks."""
        tasks: list[asyncio.Task] = []
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            tasks.append(self._background_task)

        for state in self._sync_states.values():
            if state.task and not state.task.done():
                state.task.cancel()
                tasks.append(state.task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._background_task = None
        self._initial_tasks = []
        self._background_skills = []

    async def _acquire_sandbox(self) -> SandboxHandle:
        """Return the sandbox handle to operate on.

        SPM Task 16: deferred (on_demand) mode binds a CONCRETE handle in
        ``start_deferred_sync`` (which runs INSIDE ``provisioner._provision_once``
        before the binding is ACTIVE). Calling ``self._sandbox_accessor.get()``
        there would re-enter the same inflight provision task and DEADLOCK, so we
        prefer the bound handle. Always mode leaves ``_bound_handle=None`` → the
        exact legacy ``accessor.get()`` call (INV-SPM-2 byte equivalence).
        """
        if self._bound_handle is not None:
            return self._bound_handle
        # SPM Task 17 fix #1(b) — belt-and-braces anti-deadlock guard. A
        # manager-spawned sync task is only ever created AFTER
        # ``start_deferred_sync`` bound a concrete handle (deferred mode) or is
        # never deferred at all (always mode). Reaching here with
        # ``deferred and _bound_handle is None`` means a sync op ran before
        # provision bound the handle — the exact ordering that deadlocked
        # (re-entering the inflight provision via ``accessor.get()``). Fail LOUD
        # instead of silently deadlocking; a future ordering regression surfaces
        # as an error, not a hang. Always mode keeps the byte-identical
        # ``accessor.get()`` path (INV-SPM-2).
        if self._deferred:
            raise RuntimeError(
                "deferred skill sync used before provision bound a handle"
            )
        return await self._sandbox_accessor.get()

    def _ensure_sync_task(self, skill: Skill) -> asyncio.Task[str | None]:
        state = self._sync_states.get(skill.id)
        if state and state.task:
            return state.task

        version = self._version_of(skill)
        state = SkillSyncState(status="pending", version=version)
        task = asyncio.create_task(self._sync_skill(skill))
        state.task = task
        self._sync_states[skill.id] = state
        return task

    async def _sync_skill(self, skill: Skill) -> str | None:
        lock = self._locks.setdefault(skill.id, asyncio.Lock())
        state = self._sync_states.setdefault(
            skill.id, SkillSyncState(version=self._version_of(skill))
        )

        async with lock:
            state.status = "running"
            state.error = None
            state.outcome = None   # D1a: 终态收敛——重置，防上一轮 outcome 残留
            try:
                result = await self._sync_bundle(skill)
                state.outcome = result.outcome
                state.status = "success"
                # 簿记只放 result.synced_dir，绝不放 result 对象（旧读者零感知）
                state.sandbox_dir = result.synced_dir or ""
                state.error = result.error
                # Cache file listing from bundle directory (covers both
                # fresh upload and version-match early return paths)
                self._cache_file_listing(skill)
                return result.synced_dir
            except Exception as e:  # noqa: BLE001
                state.status = "failed"
                state.error = str(e)
                # outcome 保持上面重置的 None：硬失败走 status 通道，非五态治理 outcome
                logger.warning("Skill bundle同步失败(skill=%s): %s", skill.id, str(e))
                return None

    def _cache_file_listing(self, skill: Skill) -> None:
        """Enumerate bundle directory and cache the relative file paths."""
        bundle_dir = self._skills_root_dir / skill.id / "bundle"
        if not bundle_dir.exists() or not bundle_dir.is_dir():
            return
        rel_paths = sorted(
            p.relative_to(bundle_dir).as_posix()
            for p in bundle_dir.rglob("*")
            if p.is_file()
        )
        if rel_paths:
            self._file_listings[skill.id] = rel_paths

    async def _run_background_sync(self) -> None:
        semaphore = asyncio.Semaphore(self._background_concurrency)

        async def _worker(skill: Skill) -> None:
            async with semaphore:
                task = self._ensure_sync_task(skill)
                await task

        try:
            await asyncio.gather(
                *[_worker(skill) for skill in self._background_skills],
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            raise
        finally:
            self._background_task = None

    async def _sync_bundle(self, skill: Skill) -> BundleSyncResult:
        bundle_count = self._bundle_file_count(skill)
        if bundle_count <= 0:
            return BundleSyncResult("no_bundle", None, None)

        sandbox_skill_dir = f"{self.sandbox_skill_root}/{skill.id}"
        marker_path = f"{sandbox_skill_dir}/{SYNC_MARKER_FILENAME}"
        version = self._version_of(skill)

        # D1a §4.2 G4b/G5 前置状态门（R2#3）：置于 marker 快路径之前，否则已同步
        # 版本会绕过治理判定。admission_port=None → None（off 直通，零 I/O）。
        _gate_reason = await skill_invoke_gate(self._admission_port, skill.id)
        if _gate_reason is not None:
            return BundleSyncResult("governance_rejected", None, _gate_reason)

        marker_version = await self._read_marker_version(marker_path)
        if marker_version and marker_version == version:
            # marker 命中：机会性 artifact verify（复用已存 content_hash，零额外 I/O）
            stored = (skill.scan_report or {}).get("content_hash")
            if stored and not await verify_skill_artifact(
                self._admission_port, skill.id, stored
            ):
                return BundleSyncResult("governance_rejected", None, "artifact_mismatch")
            return BundleSyncResult("already_current", sandbox_skill_dir, None)

        bundle_dir = self._skills_root_dir / skill.id / "bundle"
        if not bundle_dir.exists() or not bundle_dir.is_dir():
            raise RuntimeError(f"Skill[{skill.id}] bundle目录不存在: {bundle_dir}")

        # --- R3: Security scan gate ---
        # Scan skill root dir (includes manifest.json + SKILL.md + bundle/)
        # for consistency with install-time and startup scan paths.
        _skill_root_dir = self._skills_root_dir / skill.id
        new_hash = SkillsGuard.compute_content_hash(_skill_root_dir)
        old_hash = (skill.scan_report or {}).get("content_hash")

        # D1a §4.2 G5 artifact verify：hash 算出即 verify（payload=已算 hash，零重复
        # I/O，R9#3）。verify 失败 → 拒写、保留旧盘版本继续服务。off → True 直通。
        if not await verify_skill_artifact(self._admission_port, skill.id, new_hash):
            return BundleSyncResult("governance_rejected", None, "artifact_mismatch")

        if new_hash != old_hash:
            # Check if this hash was force-approved at install time
            _force_hash = getattr(skill, "force_approved_hash", None)
            if _force_hash and _force_hash == new_hash:
                logger.info(
                    "Bundle sync: skill %s hash matches force_approved_hash, skipping gate",
                    skill.id,
                )
            else:
                report = scan_skill_source(skill.runtime_type, _skill_root_dir)
                decision = get_install_decision(skill.trust_origin, report.verdict)

                if decision == "block":
                    logger.error(
                        "Bundle sync blocked for skill %s: verdict=%s, findings=%s",
                        skill.id,
                        report.verdict,
                        [f.pattern_id for f in report.findings[:5]],
                    )
                    self._write_last_rejected_sync(skill, report)
                    # R3 拒新 bundle：保留旧盘版本继续服务（synced_dir=None → 调用方
                    # 回退 default_exec_dir；error=None 保持 off-mode ensure_ready (None,None)）
                    return BundleSyncResult("r3_rejected_old_bundle", None, None)

                if decision == "warn":
                    logger.warning(
                        "Bundle sync warning for skill %s: verdict=%s",
                        skill.id,
                        report.verdict,
                    )

                # allow or warn: update scan_report on the skill object AND persist to disk
                # so subsequent sessions see the new risk level immediately.
                skill.scan_report = report.to_dict()
                self._persist_scan_report(skill, report)
        # --- End R3 scan gate ---

        files = sorted(path for path in bundle_dir.rglob("*") if path.is_file())
        if not files:
            raise RuntimeError(f"Skill[{skill.id}] bundle为空，无法同步")

        # PR-1b (SPM Task 9): pull the concrete handle lazily at use time.
        # SPM Task 16: prefer the bound handle in deferred mode (anti-deadlock).
        sandbox = await self._acquire_sandbox()
        for source_path in files:
            rel_path = source_path.relative_to(bundle_dir).as_posix()
            target_path = f"{sandbox_skill_dir}/{rel_path}"
            with source_path.open("rb") as fp:
                result = await sandbox.upload_file(
                    file_data=fp,
                    filepath=target_path,
                    filename=source_path.name,
                )
            if not result.success:
                raise RuntimeError(
                    f"上传文件失败: {rel_path} ({result.message or 'unknown error'})"
                )

        marker = json.dumps(
            {
                "skill_id": skill.id,
                "version": version,
                "synced_at": datetime.now().isoformat(),
                "bundle_file_count": len(files),
            },
            ensure_ascii=False,
        )
        marker_result = await sandbox.write_file(
            filepath=marker_path,
            content=marker,
        )
        if not marker_result.success:
            raise RuntimeError(
                f"写入同步标记失败: {marker_result.message or 'unknown error'}"
            )
        return BundleSyncResult("uploaded", sandbox_skill_dir, None)

    def _write_last_rejected_sync(self, skill: Skill, report: ScanReport) -> None:
        """Write rejected sync info to meta.json without changing scan_report."""
        meta_path = self._skills_root_dir / skill.id / "meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, FileNotFoundError):
            return
        meta["last_rejected_sync"] = {
            "verdict": report.verdict,
            "findings": [
                {
                    "pattern_id": f.pattern_id,
                    "category": f.category,
                    "severity": f.severity,
                    "file": f.file,
                    "line": f.line,
                    "match": f.match,
                }
                for f in report.findings
            ],
            "content_hash": report.content_hash,
            "rejected_at": report.scanned_at.isoformat(),
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _persist_scan_report(self, skill: Skill, report: ScanReport) -> None:
        """Persist updated scan_report to meta.json after allow/warn rescan."""
        meta_path = self._skills_root_dir / skill.id / "meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, FileNotFoundError):
            return
        meta["scan_report"] = report.to_dict()
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    async def _read_marker_version(self, marker_path: str) -> str:
        # PR-1b (SPM Task 9): pull the concrete handle lazily at use time.
        # SPM Task 16: prefer the bound handle in deferred mode (anti-deadlock).
        sandbox = await self._acquire_sandbox()
        exists_result = await sandbox.check_file_exists(marker_path)
        if not exists_result.success:
            return ""
        data = exists_result.data if isinstance(exists_result.data, dict) else {}
        if not data.get("exists"):
            return ""

        read_result = await sandbox.read_file(filepath=marker_path, max_length=4096)
        if not read_result.success:
            return ""
        read_data = read_result.data if isinstance(read_result.data, dict) else {}
        content = str(read_data.get("content") or "").strip()
        if not content:
            return ""
        try:
            payload = json.loads(content)
        except Exception:
            return ""
        return str(payload.get("version") or "")

    @staticmethod
    def _needs_sync(skill: Skill) -> bool:
        return (
            skill.runtime_type == SkillRuntimeType.NATIVE
            and SkillBundleSyncManager._bundle_file_count(skill) > 0
        )

    @staticmethod
    def _bundle_file_count(skill: Skill) -> int:
        manifest = skill.manifest if isinstance(skill.manifest, dict) else {}
        raw = manifest.get("bundle_file_count")
        try:
            return int(raw or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _version_of(skill: Skill) -> str:
        manifest = skill.manifest if isinstance(skill.manifest, dict) else {}
        return str(manifest.get("last_sync_at") or "static")
