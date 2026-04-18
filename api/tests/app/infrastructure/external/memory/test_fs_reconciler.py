"""FsReconciler unit tests (M1 PR-5B).

Coverage:

``scan_pending_fs_sync`` — 启动时/lifespan 快路径
- 读 ``find_pending_fs_sync`` → 调 writer.write → 成功 ``mark_fs_synced(True)``
- writer 失败不标 synced，下一轮继续重试
- 空列表直接返回，不调 writer
- 传入 ``user_id`` 走 per-user 过滤

``walk_user_directory`` — per-user 懒路径
- 孤儿文件（fs 有 DB 无） → 移到 ``.orphans/{ts}/``
- 孤儿 DB 行（DB 有 + fs_synced=true + file 无） → 调 writer.write 重建
- user 目录下任意 symlink（目录或文件）→ 移到 ``.orphans/``
- ``.orphans/`` 本身被跳过，不会把自己再移进去
- 进程内第二次同 user_id 调用是 no-op（``_walked_users`` 缓存）
- ``force=True`` 绕过缓存重跑

``reconcile_all_users`` — CLI 全量
- 取 DB distinct user_ids + fs listdir 并集，逐个 walk
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from contextlib import asynccontextmanager

import pytest

from app.domain.models.memory_chunk import MemoryChunk
from app.infrastructure.external.memory.fs_memory_writer import FsMemoryWriter
from app.infrastructure.external.memory.fs_reconciler import FsReconciler
from app.infrastructure.external.memory.frontmatter import (
    build_memory_frontmatter,
    serialize_memory_file,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


_USER = "11111111-1111-4111-8111-111111111111"
_USER_B = "22222222-2222-4222-8222-222222222222"


def _chunk(
    *,
    id_: str = "01HXYZ-ABCDEF",
    user_id: str = _USER,
    category: str | None = "user",
    content: str = "pref: dark mode",
    fs_synced: bool = False,
    metadata: dict | None = None,
) -> MemoryChunk:
    now = datetime.now(timezone.utc)
    return MemoryChunk(
        id=id_,
        user_id=user_id,
        session_id=None,
        content=content,
        content_hash="h" + id_,
        embedding=None,
        source="manual",
        metadata=metadata or {},
        created_at=now,
        updated_at=now,
        category=category,
        auto_promoted_at=None,
        fs_synced=fs_synced,
        pinned=False,
    )


def _fake_session_factory(repo: AsyncMock):
    """Build a fake session_factory whose context manager yields a session
    object the reconciler passes to repo_factory. We don't care about the
    session's identity — repo_factory ignores its arg and returns the
    preconfigured AsyncMock repo."""
    session = MagicMock(name="session")
    session.commit = AsyncMock()
    session.rollback = AsyncMock()

    @asynccontextmanager
    async def factory():
        yield session

    return factory, session


def _build_reconciler(
    repo: AsyncMock,
    writer: FsMemoryWriter,
    memory_root: Path,
) -> FsReconciler:
    session_factory, _ = _fake_session_factory(repo)

    def repo_factory(_session):
        return repo

    return FsReconciler(
        session_factory=session_factory,
        repo_factory=repo_factory,
        file_store=writer,
        memory_root=memory_root,
    )


# ──────────────────────────────────────────────────────────────────────
# scan_pending_fs_sync
# ──────────────────────────────────────────────────────────────────────


class TestScanPendingFsSync:
    async def test_writer_success_marks_synced(self, tmp_path: Path) -> None:
        writer = FsMemoryWriter(tmp_path, max_retries=1, base_backoff_seconds=0.0)
        repo = AsyncMock()
        pending = _chunk(fs_synced=False)
        repo.find_pending_fs_sync.return_value = [pending]
        repo.mark_fs_synced.return_value = True

        reconciler = _build_reconciler(repo, writer, tmp_path)
        summary = await reconciler.scan_pending_fs_sync()

        assert summary["attempted"] == 1
        assert summary["succeeded"] == 1
        assert summary["failed"] == 0
        repo.mark_fs_synced.assert_awaited_once()
        called_kwargs = repo.mark_fs_synced.await_args.kwargs
        assert called_kwargs["chunk_id"] == pending.id
        assert called_kwargs["synced"] is True

        target = tmp_path / pending.user_id / pending.category / f"{pending.id}.md"
        assert target.exists()

    async def test_writer_failure_keeps_pending(self, tmp_path: Path) -> None:
        repo = AsyncMock()
        repo.find_pending_fs_sync.return_value = [_chunk(fs_synced=False)]

        writer = MagicMock()
        writer.write = AsyncMock(side_effect=OSError("disk full"))
        writer.delete = AsyncMock()
        writer.move_category = AsyncMock()

        reconciler = _build_reconciler(repo, writer, tmp_path)
        summary = await reconciler.scan_pending_fs_sync()

        assert summary["failed"] == 1
        repo.mark_fs_synced.assert_not_awaited()

    async def test_empty_pending_list_noop(self, tmp_path: Path) -> None:
        repo = AsyncMock()
        repo.find_pending_fs_sync.return_value = []
        writer = MagicMock()
        writer.write = AsyncMock()

        reconciler = _build_reconciler(repo, writer, tmp_path)
        summary = await reconciler.scan_pending_fs_sync()

        assert summary["attempted"] == 0
        writer.write.assert_not_awaited()

    async def test_passes_user_id_filter_to_repo(self, tmp_path: Path) -> None:
        repo = AsyncMock()
        repo.find_pending_fs_sync.return_value = []
        writer = MagicMock()

        reconciler = _build_reconciler(repo, writer, tmp_path)
        await reconciler.scan_pending_fs_sync(user_id=_USER)

        repo.find_pending_fs_sync.assert_awaited_once()
        assert repo.find_pending_fs_sync.await_args.kwargs["user_id"] == _USER


# ──────────────────────────────────────────────────────────────────────
# walk_user_directory
# ──────────────────────────────────────────────────────────────────────


def _write_memory_file(root: Path, user_id: str, memory_id: str, category: str, chunk: MemoryChunk) -> Path:
    """Helper: materialize a canonical memory file on disk bypassing writer
    (测试用，让我们人工构造各种冲突场景)。"""
    target_dir = root / user_id / category
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{memory_id}.md"
    fm = build_memory_frontmatter(chunk)
    target.write_text(
        serialize_memory_file(fm, chunk.content),
        encoding="utf-8",
    )
    return target


class TestWalkUserDirectoryOrphanFiles:
    async def test_orphan_file_moved_to_orphans(self, tmp_path: Path) -> None:
        """fs 有 id=A 的文件，DB 查 id=A 返回 None → 孤儿文件进 .orphans。"""
        orphan_chunk = _chunk(id_="orphan-A")
        fs_path = _write_memory_file(
            tmp_path, _USER, "orphan-A", "user", orphan_chunk
        )
        assert fs_path.exists()

        repo = AsyncMock()
        repo.get_by_id.return_value = None  # 孤儿
        # 第二个 pass（DB → fs）需要 list_by_user + count_by_user 也给一个空集
        repo.list_by_user.return_value = []
        repo.count_by_user.return_value = 0

        writer = FsMemoryWriter(tmp_path, max_retries=1, base_backoff_seconds=0.0)
        reconciler = _build_reconciler(repo, writer, tmp_path)
        result = await reconciler.walk_user_directory(_USER)

        assert not fs_path.exists(), "原孤儿文件应被移走"
        orphans_dir = tmp_path / _USER / ".orphans"
        assert orphans_dir.exists()
        moved = list(orphans_dir.rglob("orphan-A.md"))
        assert len(moved) == 1
        assert result["orphan_files"] == 1

    async def test_skips_orphans_subdirectory_itself(self, tmp_path: Path) -> None:
        """``.orphans`` 目录本身不参与 orphan 扫描 —— 不然会无限搬自己。"""
        orphans_dir = tmp_path / _USER / ".orphans" / "2026-04-17T00-00-00Z"
        orphans_dir.mkdir(parents=True)
        stale = orphans_dir / "old.md"
        stale.write_text("garbage", encoding="utf-8")

        repo = AsyncMock()
        repo.get_by_id.return_value = None
        repo.list_by_user.return_value = []
        repo.count_by_user.return_value = 0

        writer = FsMemoryWriter(tmp_path, max_retries=1, base_backoff_seconds=0.0)
        reconciler = _build_reconciler(repo, writer, tmp_path)
        result = await reconciler.walk_user_directory(_USER)

        assert stale.exists(), ".orphans 内文件应被完全忽略"
        assert result["orphan_files"] == 0

    async def test_path_mismatch_after_failed_move_category(
        self, tmp_path: Path
    ) -> None:
        """DB 里 chunk.category='rule'，但磁盘上的文件在 ``/user/<id>.md`` —— 说明
        上次 ``move_category`` 写完新路径后删旧路径失败。reconciler 必须把旧路径文件
        搬进 ``.orphans``，否则新旧两份永久残留，sandbox 读到的内容会分叉。"""
        stale_chunk = _chunk(id_="moved-id", category="user")  # 文件内 frontmatter 仍写 user
        stale_file = _write_memory_file(
            tmp_path, _USER, "moved-id", "user", stale_chunk
        )
        # DB 里 category 已经迁到 rule（其他路径也写好了 rule/<id>.md，
        # 模拟 move_category step 3 成功、step 4 删旧失败的遗留）。
        db_chunk = _chunk(id_="moved-id", category="rule")

        repo = AsyncMock()
        repo.get_by_id.return_value = db_chunk
        repo.list_by_user.return_value = []
        repo.count_by_user.return_value = 0

        writer = FsMemoryWriter(tmp_path, max_retries=1, base_backoff_seconds=0.0)
        reconciler = _build_reconciler(repo, writer, tmp_path)
        result = await reconciler.walk_user_directory(_USER)

        assert not stale_file.exists(), "旧路径文件应被移走（category 不匹配）"
        orphans_dir = tmp_path / _USER / ".orphans"
        assert orphans_dir.exists()
        moved = list(orphans_dir.rglob("moved-id.md"))
        assert len(moved) == 1
        assert result["orphan_files"] == 1

    async def test_nested_nonstandard_path_quarantined(
        self, tmp_path: Path
    ) -> None:
        """canonical 布局是 ``{user}/{category}/{id}.md`` 三层；更深层级的文件
        不符合规范，一律当孤儿。"""
        nested = tmp_path / _USER / "user" / "subdir" / "weird.md"
        nested.parent.mkdir(parents=True)
        fm = build_memory_frontmatter(_chunk(id_="weird", category="user"))
        nested.write_text(
            serialize_memory_file(fm, "content"), encoding="utf-8"
        )

        repo = AsyncMock()
        # 就算 DB 里 id 存在，path 层级超过预期仍然孤儿化
        repo.get_by_id.return_value = _chunk(id_="weird", category="user")
        repo.list_by_user.return_value = []
        repo.count_by_user.return_value = 0

        writer = FsMemoryWriter(tmp_path, max_retries=1, base_backoff_seconds=0.0)
        reconciler = _build_reconciler(repo, writer, tmp_path)
        result = await reconciler.walk_user_directory(_USER)

        assert not nested.exists()
        assert result["orphan_files"] == 1

    async def test_filename_mismatch_quarantined(self, tmp_path: Path) -> None:
        """frontmatter id 和 DB row 都存在，但 basename 不是 ``{id}.md`` 时，
        仍应视为非 canonical 路径并搬进 .orphans。"""
        chunk = _chunk(id_="real-id", category="user")
        wrong_dir = tmp_path / _USER / "user"
        wrong_dir.mkdir(parents=True)
        wrong_name = wrong_dir / "wrong-name.md"
        wrong_name.write_text(
            serialize_memory_file(build_memory_frontmatter(chunk), chunk.content),
            encoding="utf-8",
        )

        repo = AsyncMock()
        repo.get_by_id.return_value = _chunk(id_="real-id", category="user")
        repo.list_by_user.return_value = []
        repo.count_by_user.return_value = 0

        writer = FsMemoryWriter(tmp_path, max_retries=1, base_backoff_seconds=0.0)
        reconciler = _build_reconciler(repo, writer, tmp_path)
        result = await reconciler.walk_user_directory(_USER)

        assert not wrong_name.exists()
        moved = list((tmp_path / _USER / ".orphans").rglob("wrong-name.md"))
        assert len(moved) == 1
        assert result["orphan_files"] == 1

    async def test_symlink_in_user_dir_quarantined(self, tmp_path: Path) -> None:
        """symlink（design L655 Case C: 恶意预置到 user_id 下指向 /etc/passwd）必须搬
        走到 .orphans，防止 sandbox 透过 bind-mount 读到任意路径。"""
        target_real = tmp_path / "external_secret.md"
        target_real.write_text("---\nid: x\n---\n\nsecret", encoding="utf-8")
        user_dir = tmp_path / _USER / "user"
        user_dir.mkdir(parents=True)
        sym = user_dir / "evil.md"
        try:
            os.symlink(target_real, sym)
        except (OSError, NotImplementedError):
            pytest.skip("filesystem 不支持 symlink")

        repo = AsyncMock()
        repo.get_by_id.return_value = None
        repo.list_by_user.return_value = []
        repo.count_by_user.return_value = 0

        writer = FsMemoryWriter(tmp_path, max_retries=1, base_backoff_seconds=0.0)
        reconciler = _build_reconciler(repo, writer, tmp_path)
        result = await reconciler.walk_user_directory(_USER)

        assert not sym.exists() or sym.is_symlink() is False, "symlink 应被搬走"
        orphans = list((tmp_path / _USER / ".orphans").rglob("evil*"))
        assert len(orphans) == 1
        assert result["orphan_symlinks"] >= 1
        assert target_real.exists(), "symlink 的 target 不能被动到"


class TestWalkUserDirectoryOrphanDBRows:
    async def test_orphan_db_row_rebuilt_to_fs(self, tmp_path: Path) -> None:
        """DB 有 id=A (fs_synced=true) 但磁盘不存在 → 从 DB 重建文件。"""
        chunk = _chunk(id_="rebuild-me", fs_synced=True, content="body text")

        repo = AsyncMock()
        repo.get_by_id.return_value = None  # fs-side 没有文件，orphan pass 空
        repo.list_by_user.side_effect = [[chunk], []]  # page 0 有一条, page 1 空
        repo.count_by_user.return_value = 1

        writer = FsMemoryWriter(tmp_path, max_retries=1, base_backoff_seconds=0.0)
        reconciler = _build_reconciler(repo, writer, tmp_path)
        result = await reconciler.walk_user_directory(_USER)

        rebuilt = tmp_path / _USER / "user" / "rebuild-me.md"
        assert rebuilt.exists()
        assert "body text" in rebuilt.read_text(encoding="utf-8")
        assert result["rebuilt_files"] == 1

    async def test_skips_legacy_rows_without_category(self, tmp_path: Path) -> None:
        """legacy 行 category=NULL (M1 前写入) 不重建——没有 category 不知往哪写。"""
        legacy = _chunk(id_="legacy", category=None, fs_synced=True)

        repo = AsyncMock()
        repo.get_by_id.return_value = None
        repo.list_by_user.side_effect = [[legacy], []]
        repo.count_by_user.return_value = 1

        writer = FsMemoryWriter(tmp_path, max_retries=1, base_backoff_seconds=0.0)
        reconciler = _build_reconciler(repo, writer, tmp_path)
        result = await reconciler.walk_user_directory(_USER)

        assert result["rebuilt_files"] == 0

    async def test_rebuild_write_failure_flips_fs_synced_false_and_commits(
        self, tmp_path: Path
    ) -> None:
        """写盘失败 → ``mark_fs_synced(False)`` + commit，下轮 scan_pending 接着捡。
        回归：之前 commit 只在 ``rebuilt_files>0`` 时触发，会导致翻 flag 静默回滚。"""
        chunk = _chunk(id_="rebuild-me", fs_synced=True, content="body")

        writer = MagicMock()
        writer.write = AsyncMock(side_effect=OSError("disk full"))

        repo = AsyncMock()
        repo.get_by_id.return_value = None
        repo.list_by_user.side_effect = [[chunk], []]
        repo.count_by_user.return_value = 1

        # 注入自定义 session factory 以观察 commit 行为
        session = MagicMock(name="session")
        session.commit = AsyncMock()
        session.rollback = AsyncMock()

        from contextlib import asynccontextmanager as acm

        @acm
        async def session_factory():
            yield session

        reconciler = FsReconciler(
            session_factory=session_factory,
            repo_factory=lambda _s: repo,
            file_store=writer,
            memory_root=tmp_path,
        )
        result = await reconciler.walk_user_directory(_USER)

        repo.mark_fs_synced.assert_awaited_once()
        assert repo.mark_fs_synced.await_args.kwargs["synced"] is False
        session.commit.assert_awaited_once()
        assert result["rebuilt_files"] == 0

    async def test_skips_fs_synced_false_rows(self, tmp_path: Path) -> None:
        """``fs_synced=false`` 的行由 ``scan_pending_fs_sync`` 快路径负责，
        reconciler 的慢路径不重复处理，避免和快路径抢写。"""
        pending = _chunk(id_="pending", fs_synced=False)

        repo = AsyncMock()
        repo.get_by_id.return_value = None
        repo.list_by_user.side_effect = [[pending], []]
        repo.count_by_user.return_value = 1

        writer = FsMemoryWriter(tmp_path, max_retries=1, base_backoff_seconds=0.0)
        reconciler = _build_reconciler(repo, writer, tmp_path)
        result = await reconciler.walk_user_directory(_USER)

        assert result["rebuilt_files"] == 0


class TestWalkUserDirectoryIdempotence:
    async def test_second_call_short_circuits(self, tmp_path: Path) -> None:
        repo = AsyncMock()
        repo.get_by_id.return_value = None
        repo.list_by_user.return_value = []
        repo.count_by_user.return_value = 0

        writer = FsMemoryWriter(tmp_path, max_retries=1, base_backoff_seconds=0.0)
        reconciler = _build_reconciler(repo, writer, tmp_path)
        first = await reconciler.walk_user_directory(_USER)
        second = await reconciler.walk_user_directory(_USER)

        assert first["walked"] is True
        assert second["walked"] is False
        # 第二次不应该再查 DB
        assert repo.list_by_user.await_count == 1

    async def test_force_bypasses_cache(self, tmp_path: Path) -> None:
        repo = AsyncMock()
        repo.get_by_id.return_value = None
        repo.list_by_user.return_value = []
        repo.count_by_user.return_value = 0

        writer = FsMemoryWriter(tmp_path, max_retries=1, base_backoff_seconds=0.0)
        reconciler = _build_reconciler(repo, writer, tmp_path)
        await reconciler.walk_user_directory(_USER)
        result = await reconciler.walk_user_directory(_USER, force=True)

        assert result["walked"] is True
        assert repo.list_by_user.await_count == 2


class TestReconcileAllUsers:
    async def test_also_runs_pending_scan(self, tmp_path: Path) -> None:
        """默认 CLI (``reconcile_all_users``) 必须覆盖 ``fs_synced=false`` backlog，
        不能只走 fs-walk —— 不然 ops 按帮助文案跑默认命令后 pending 一条没碰。
        回归：P2 #2 之前 reconcile_all_users 只对每个 user 调 walk_user_directory
        (force=True)，而 walk 的 DB→fs 路径显式 skip fs_synced=false。"""
        pending_chunk = _chunk(id_="pending-1", fs_synced=False)
        # 成功写盘后的 chunk（fs_synced=True）用于第二阶段 walk 的 get_by_id
        synced_chunk = _chunk(id_="pending-1", fs_synced=True)
        repo = AsyncMock()
        repo.distinct_user_ids.return_value = [_USER]
        repo.find_pending_fs_sync.return_value = [pending_chunk]
        repo.mark_fs_synced.return_value = True
        repo.get_by_id.return_value = synced_chunk
        repo.list_by_user.return_value = []
        repo.count_by_user.return_value = 0

        writer = FsMemoryWriter(tmp_path, max_retries=1, base_backoff_seconds=0.0)
        reconciler = _build_reconciler(repo, writer, tmp_path)
        summary = await reconciler.reconcile_all_users()

        # 关键断言：pending scan 被调用（find_pending_fs_sync at least once without user_id filter）
        assert repo.find_pending_fs_sync.await_count >= 1
        assert summary["pending"]["attempted"] == 1
        assert summary["pending"]["succeeded"] == 1
        # 文件已写盘并由 walk 认为合法（canonical 路径 + DB 有对应 chunk）
        target = tmp_path / _USER / "user" / "pending-1.md"
        assert target.exists()

    async def test_iterates_db_and_fs_union(self, tmp_path: Path) -> None:
        """DB 有 user A；fs 上有 user A 和 user B 的目录（B 被 nuclear delete 过，
        只剩 fs 残骸）。reconcile_all_users 应同时处理两个。"""
        (tmp_path / _USER).mkdir()
        (tmp_path / _USER_B).mkdir()

        repo = AsyncMock()
        repo.distinct_user_ids.return_value = [_USER]
        repo.get_by_id.return_value = None
        repo.list_by_user.return_value = []
        repo.count_by_user.return_value = 0

        writer = FsMemoryWriter(tmp_path, max_retries=1, base_backoff_seconds=0.0)
        reconciler = _build_reconciler(repo, writer, tmp_path)
        summary = await reconciler.reconcile_all_users()

        assert summary["users_walked"] == 2
        assert set(summary["user_ids"]) == {_USER, _USER_B}
