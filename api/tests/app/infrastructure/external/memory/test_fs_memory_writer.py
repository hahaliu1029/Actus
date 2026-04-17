"""FsMemoryWriter 单元测试（PR-5A）。

覆盖：
- write / delete / move_category 的 happy path
- Path-traversal 拒绝（user_id / memory_id / category 含 `/`、`..`、`\\x00`、空）
- Symlink 拒绝（user 目录 / category 目录 / 目标文件）
- Atomic write（tmp file + rename；中途失败清理 tmp）
- overwrite=False + 已存在 → FileExistsError
- 指数退避重试（仅 OSError；PermissionError / SecurityError 不重试）
- 落盘内容包含 frontmatter + body
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from app.application.errors.exceptions import SecurityError
from app.infrastructure.external.memory.fs_memory_writer import FsMemoryWriter

pytestmark = pytest.mark.anyio


# Realistic UUID v4 that upstream validation would accept; FsMemoryWriter itself
# only checks for traversal chars, but tests should look like production usage.
_USER = "11111111-1111-4111-8111-111111111111"
_MEM = "01HXYZABCDEF123456789"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """tmp memory root. Not pre-created user dirs — writer mkdir -p on demand."""
    return tmp_path


@pytest.fixture
def writer(root: Path) -> FsMemoryWriter:
    # ``base_backoff_seconds=0`` keeps retry tests fast; production uses 0.1.
    return FsMemoryWriter(root, max_retries=3, base_backoff_seconds=0.0)


def _fm(**overrides):
    base = {
        "id": _MEM,
        "title": "t",
        "category": "user",
        "source": "manual",
        "created_at": "2026-04-17T09:48:57Z",
        "updated_at": "2026-04-17T09:48:57Z",
        "tags": [],
        "pinned": False,
    }
    base.update(overrides)
    return base


class TestWriteHappyPath:
    async def test_creates_file_under_user_category_dir(
        self, writer: FsMemoryWriter, root: Path
    ):
        await writer.write(_USER, _MEM, "user", "body", _fm(), overwrite=False)
        path = root / _USER / "user" / f"{_MEM}.md"
        assert path.exists()
        assert path.is_file()

    async def test_file_contains_frontmatter_and_body(
        self, writer: FsMemoryWriter, root: Path
    ):
        await writer.write(_USER, _MEM, "user", "hello world", _fm(), overwrite=False)
        path = root / _USER / "user" / f"{_MEM}.md"
        content = path.read_text(encoding="utf-8")
        assert content.startswith("---\n")
        assert "id: " in content
        assert "hello world" in content

    async def test_overwrite_false_rejects_existing(
        self, writer: FsMemoryWriter
    ):
        await writer.write(_USER, _MEM, "user", "a", _fm(), overwrite=False)
        with pytest.raises(FileExistsError):
            await writer.write(_USER, _MEM, "user", "b", _fm(), overwrite=False)

    async def test_overwrite_true_replaces_content(
        self, writer: FsMemoryWriter, root: Path
    ):
        await writer.write(_USER, _MEM, "user", "first", _fm(), overwrite=False)
        await writer.write(_USER, _MEM, "user", "second", _fm(), overwrite=True)
        path = root / _USER / "user" / f"{_MEM}.md"
        content = path.read_text(encoding="utf-8")
        assert "second" in content
        assert "first" not in content

    async def test_tmp_file_not_left_behind(
        self, writer: FsMemoryWriter, root: Path
    ):
        """Success path removes the tmp file via os.replace."""
        await writer.write(_USER, _MEM, "user", "ok", _fm(), overwrite=False)
        category_dir = root / _USER / "user"
        tmp_files = [p for p in category_dir.iterdir() if p.name.startswith(".")]
        assert tmp_files == []

    async def test_mkdir_creates_intermediate_dirs(
        self, writer: FsMemoryWriter, root: Path
    ):
        """``root/user_id/category`` doesn't exist pre-write — writer mkdir -p."""
        assert not (root / _USER).exists()
        await writer.write(_USER, _MEM, "rule", "a", _fm(category="rule"), overwrite=False)
        assert (root / _USER / "rule").is_dir()


class TestDeleteHappyPath:
    async def test_delete_removes_file(
        self, writer: FsMemoryWriter, root: Path
    ):
        await writer.write(_USER, _MEM, "user", "body", _fm(), overwrite=False)
        await writer.delete(_USER, _MEM, "user")
        assert not (root / _USER / "user" / f"{_MEM}.md").exists()

    async def test_delete_missing_is_noop(self, writer: FsMemoryWriter):
        # File doesn't exist → idempotent, no raise
        await writer.delete(_USER, "nonexistent", "user")


class TestMoveCategory:
    async def test_moves_file_between_categories_and_rewrites_frontmatter(
        self, writer: FsMemoryWriter, root: Path
    ):
        """Move 必须让磁盘 YAML 里的 category 跟路径一致——service 交来的
        new_frontmatter['category'] 已更新到 to_category，writer 只 serialize
        + 落盘，不继承旧文件内容。"""
        await writer.write(_USER, _MEM, "user", "body", _fm(category="user"), overwrite=False)
        await writer.move_category(
            _USER, _MEM, "user", "rule",
            content="body",
            new_frontmatter=_fm(category="rule"),
        )

        assert not (root / _USER / "user" / f"{_MEM}.md").exists()
        new_path = root / _USER / "rule" / f"{_MEM}.md"
        assert new_path.exists()
        # 新文件 YAML 里的 category 必须是 rule——不能残留旧 user
        new_content = new_path.read_text(encoding="utf-8")
        assert "category: rule" in new_content
        assert "category: user" not in new_content

    async def test_missing_source_still_writes_dest(
        self, writer: FsMemoryWriter, root: Path
    ):
        """Source 不存在 → write-to-dest 成功 + delete 幂等 no-op。

        这是刻意的：service 层构造 move 时已经知道 id+content+frontmatter；
        即使 source 文件因前一次半失败丢了，move 仍能建立正确的新路径视图
        （reconciler 之后只需清理 source 端孤儿）。比 FileNotFoundError 的
        "直接报错给 API 500" 鲁棒得多。"""
        await writer.move_category(
            _USER, "mid-crash-id", "user", "rule",
            content="recovered body",
            new_frontmatter=_fm(id="mid-crash-id", category="rule"),
        )
        assert (root / _USER / "rule" / "mid-crash-id.md").exists()


class TestPathTraversalDefense:
    @pytest.mark.parametrize("bad_user", ["../evil", "..", "/etc", "a/b", "a\\b"])
    async def test_bad_user_id(
        self, writer: FsMemoryWriter, bad_user: str
    ):
        with pytest.raises(SecurityError):
            await writer.write(bad_user, _MEM, "user", "x", _fm(), overwrite=False)

    @pytest.mark.parametrize(
        "bad_id", ["../../etc/passwd", "a/b", "a\\b", "a..b"],
    )
    async def test_bad_memory_id(
        self, writer: FsMemoryWriter, bad_id: str
    ):
        with pytest.raises(SecurityError):
            await writer.write(_USER, bad_id, "user", "x", _fm(), overwrite=False)

    @pytest.mark.parametrize("bad_cat", ["../root", "a/b", "..", "a..b"])
    async def test_bad_category(
        self, writer: FsMemoryWriter, bad_cat: str
    ):
        with pytest.raises(SecurityError):
            await writer.write(_USER, _MEM, bad_cat, "x", _fm(), overwrite=False)

    async def test_null_byte_rejected(self, writer: FsMemoryWriter):
        with pytest.raises(SecurityError):
            await writer.write(_USER, "mem\x00bad", "user", "x", _fm(), overwrite=False)

    async def test_empty_segment_rejected(self, writer: FsMemoryWriter):
        with pytest.raises(SecurityError):
            await writer.write("", _MEM, "user", "x", _fm(), overwrite=False)
        with pytest.raises(SecurityError):
            await writer.write(_USER, "", "user", "x", _fm(), overwrite=False)
        with pytest.raises(SecurityError):
            await writer.write(_USER, _MEM, "", "x", _fm(), overwrite=False)


class TestSymlinkDefense:
    async def test_rejects_symlink_user_dir(
        self, writer: FsMemoryWriter, root: Path, tmp_path: Path
    ):
        """(root)/(user_id) 是 symlink → reject，不管指向哪。"""
        outside = tmp_path / "outside_target"
        outside.mkdir()
        user_link = root / _USER
        user_link.symlink_to(outside)

        with pytest.raises(SecurityError):
            await writer.write(_USER, _MEM, "user", "x", _fm(), overwrite=False)

    async def test_rejects_symlink_category_dir(
        self, writer: FsMemoryWriter, root: Path, tmp_path: Path
    ):
        outside = tmp_path / "outside_cat"
        outside.mkdir()
        (root / _USER).mkdir()
        cat_link = root / _USER / "user"
        cat_link.symlink_to(outside)

        with pytest.raises(SecurityError):
            await writer.write(_USER, _MEM, "user", "x", _fm(), overwrite=False)

    async def test_rejects_symlink_target_file(
        self, writer: FsMemoryWriter, root: Path, tmp_path: Path
    ):
        """If some prior hand-edit left a symlink at the target path → reject."""
        (root / _USER / "user").mkdir(parents=True)
        target = tmp_path / "stolen"
        target.touch()
        (root / _USER / "user" / f"{_MEM}.md").symlink_to(target)

        with pytest.raises(SecurityError):
            await writer.write(_USER, _MEM, "user", "x", _fm(), overwrite=False)


class TestRetryBehavior:
    async def test_transient_oserror_retries_up_to_max(
        self, writer: FsMemoryWriter, root: Path
    ):
        """Writer should retry OSError up to max_retries (configured 3 here)."""
        call_log = []

        # Patch the underlying atomic write to raise 2 times then succeed
        real_write = writer._write_atomic
        call_count = {"n": 0}

        def flaky(*args, **kwargs):
            call_log.append("call")
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise OSError("transient")
            return real_write(*args, **kwargs)

        with patch.object(writer, "_write_atomic", side_effect=flaky):
            await writer.write(_USER, _MEM, "user", "b", _fm(), overwrite=False)

        assert len(call_log) == 3

    async def test_exhausted_retries_raise_last_error(
        self, writer: FsMemoryWriter
    ):
        """All attempts fail → raise the last OSError."""
        def always_fail(*args, **kwargs):
            raise OSError("disk full")

        with patch.object(writer, "_write_atomic", side_effect=always_fail):
            with pytest.raises(OSError, match="disk full"):
                await writer.write(_USER, _MEM, "user", "b", _fm(), overwrite=False)

    async def test_security_error_does_not_retry(
        self, writer: FsMemoryWriter
    ):
        """SecurityError is raised from ``_resolve_target`` BEFORE retry
        wrapper — verifies retry stack doesn't accidentally catch it."""
        # Use an invalid user_id that triggers _reject_segment
        attempts = {"n": 0}

        def _fake_atomic(*args, **kwargs):
            attempts["n"] += 1

        # Should raise SecurityError without ever reaching _write_atomic
        with patch.object(writer, "_write_atomic", side_effect=_fake_atomic):
            with pytest.raises(SecurityError):
                await writer.write("../bad", _MEM, "user", "x", _fm(), overwrite=False)
        assert attempts["n"] == 0

    async def test_permission_error_not_retried(
        self, writer: FsMemoryWriter
    ):
        """PermissionError is a config/chown issue — retry is futile, raise once."""
        attempts = {"n": 0}

        def perm_denied(*args, **kwargs):
            attempts["n"] += 1
            raise PermissionError("chmod please")

        with patch.object(writer, "_write_atomic", side_effect=perm_denied):
            with pytest.raises(PermissionError):
                await writer.write(_USER, _MEM, "user", "x", _fm(), overwrite=False)
        assert attempts["n"] == 1

    async def test_backoff_sleeps_between_attempts(
        self, root: Path
    ):
        """Exponential backoff: base * 2^attempt. With base=0.01s and 3
        attempts, total sleep ≈ 0.01 + 0.02 = 0.03s (on 3rd attempt no sleep).
        Use a coarse threshold so the test isn't flaky on busy CI."""
        w = FsMemoryWriter(root, max_retries=3, base_backoff_seconds=0.01)
        call_count = {"n": 0}

        def always_oserror(*args, **kwargs):
            call_count["n"] += 1
            raise OSError("e")

        start = time.monotonic()
        with patch.object(w, "_write_atomic", side_effect=always_oserror):
            with pytest.raises(OSError):
                await w.write(_USER, _MEM, "user", "x", _fm(), overwrite=False)
        elapsed = time.monotonic() - start
        # 0.01 + 0.02 = 0.03 baseline; allow 2× headroom for scheduler noise
        assert call_count["n"] == 3
        assert 0.02 < elapsed < 0.5


class TestMemoryRootResolution:
    async def test_resolves_tilde_path(self, tmp_path: Path, monkeypatch):
        """``memory_root`` 用 ``~/...`` 形式传入时 expanduser 生效。"""
        monkeypatch.setenv("HOME", str(tmp_path))
        w = FsMemoryWriter("~/.actus/memory")
        # Root will resolve to tmp_path/.actus/memory — directory doesn't exist
        # yet, writer just warns. First write should mkdir -p.
        assert str(w._root).startswith(str(tmp_path.resolve()))

    async def test_relative_paths_resolved_to_absolute(
        self, tmp_path: Path, monkeypatch
    ):
        """相对路径经由 Path.resolve() 得到绝对路径——部署容器里也是绝对
        路径（/app/data/memory），这个检查防御 config 意外传相对值。"""
        monkeypatch.chdir(tmp_path)
        w = FsMemoryWriter("relative/memory")
        assert w._root.is_absolute()
