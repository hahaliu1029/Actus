"""FsMemoryWriter — filesystem backend for ``FileMemoryStore`` protocol.

Path layout (design §P2):

    ${MEMORY_ROOT_CONTAINER}/{user_id}/{category}/{memory_id}.md

Responsibilities:

* Path-traversal defense: ``(root / user_id / category / f"{memory_id}.md").resolve()``
  must stay under ``root.resolve()``; otherwise ``SecurityError``.
* Symlink rejection: target path or any ancestor **under ``root``** must not be a
  symlink. Ancestors above ``root`` (e.g. ``/home/liu``) are out of scope — we
  can't control the deployer's homedir, and the M0 spike treats the host mount
  as a trusted root.
* id/user_id/category hygiene: reject ``/``, ``..``, null byte, empty string.
* Atomic write: ``tempfile.mkstemp`` + fsync + ``os.replace`` in the target dir.
* Retryable errors (``OSError`` that is NOT ``PermissionError``/``IsADirectoryError``
  /``NotADirectoryError``): exponential backoff, ``max_retries`` attempts total.
* Fs ops run on a thread pool via ``asyncio.to_thread``; FsMemoryWriter is
  always called from async contexts (service layer).

What PR-5A does NOT include:

* Reconciler scans for ``fs_synced=false`` rows or orphan files — PR-5B.
* Host-side symlink removal during reconcile walks — PR-5B.
* Sandbox mount wiring — PR-6A.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from app.application.errors.exceptions import SecurityError
from app.infrastructure.external.memory.frontmatter import serialize_memory_file

logger = logging.getLogger(__name__)


# Characters / substrings forbidden in user_id / memory_id / category segments.
# ``..`` is path traversal, ``/`` would split the segment, ``\x00`` is a
# POSIX path terminator that surprises many tools. Empty string is also
# rejected because it collapses to the parent dir under ``Path`` semantics.
_FORBIDDEN_SUBSTRINGS = ("/", "\\", "..", "\x00")


def _reject_segment(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise SecurityError(f"{field} 不能为空")
    for bad in _FORBIDDEN_SUBSTRINGS:
        if bad in value:
            raise SecurityError(f"{field} 含非法字符 {bad!r}: {value!r}")


class FsMemoryWriter:
    """Filesystem implementation of :class:`FileMemoryStore`.

    Consumes ``memory_root`` (string or Path) at construction — the same root
    must match ``config.memory_root_container`` so that api-container writes
    and sandbox-container reads see the same underlying host directory
    (design §P2).
    """

    def __init__(
        self,
        memory_root: str | os.PathLike[str],
        *,
        max_retries: int = 5,
        base_backoff_seconds: float = 0.1,
    ) -> None:
        root = Path(os.fspath(memory_root)).expanduser()
        # ``resolve()`` both canonicalizes the root and surfaces a missing
        # parent early. We deliberately do NOT ``mkdir`` the root here —
        # that's the deployer's responsibility (design L81, chown-by-deployer
        # model); mkdir-ing would mask config mistakes where memory_root
        # points at a typo.
        self._root = root.resolve()
        if not self._root.exists():
            # Don't hard-fail: ``FsMemoryWriter`` can be constructed before
            # the deployer's ``mkdir -p`` runs in CI/tests. We just skip the
            # pre-resolution sanity check. ``_resolve_target`` still enforces
            # the is_relative_to invariant at each write.
            logger.warning(
                "FsMemoryWriter memory_root %s 不存在，首次 write 时将由 mkdir -p 创建",
                self._root,
            )
        self._max_retries = max(1, max_retries)
        self._base_backoff_seconds = base_backoff_seconds

    # ---- FileMemoryStore protocol ------------------------------------- #

    async def write(
        self,
        user_id: str,
        memory_id: str,
        category: str,
        content: str,
        frontmatter: dict,
        *,
        overwrite: bool = False,
    ) -> None:
        target = self._resolve_target(user_id, memory_id, category)
        serialized = serialize_memory_file(frontmatter, content)
        await self._run_with_retry(
            self._write_atomic,
            target,
            serialized,
            overwrite,
        )

    async def delete(
        self,
        user_id: str,
        memory_id: str,
        category: str,
    ) -> None:
        target = self._resolve_target(user_id, memory_id, category)
        await self._run_with_retry(self._delete_if_exists, target)

    async def move_category(
        self,
        user_id: str,
        memory_id: str,
        from_category: str,
        to_category: str,
        content: str,
        new_frontmatter: dict,
    ) -> None:
        # 语义：service 层已经把 ``new_frontmatter['category']`` 更新到 to_category、
        # ``updated_at`` 也已刷新。writer 不解析旧文件，直接用 service 交来的
        # content + new_frontmatter 重新 serialize，原子写入新路径、再清旧路径。
        # 这样避免了 "旧 bytes 搬家" 路径里 YAML 里 category 残留旧值的漂移。
        #
        # 如果新路径已存在：用 ``overwrite=True``，因为理论上 service 层只有
        # category 变更时才调 move，同 id 新路径不应该存在；但如果存在，就是
        # reconciler 或前一次半成功的残留，直接覆盖重写是对的。
        dest = self._resolve_target(user_id, memory_id, to_category)
        # Compute source path only to verify + delete; don't need its contents.
        source = self._resolve_target(user_id, memory_id, from_category)
        serialized = serialize_memory_file(new_frontmatter, content)
        await self._run_with_retry(self._write_atomic, dest, serialized, True)
        # Best-effort delete of old path; reconciler mops up if this partially
        # fails (design L451, §Update Flow / Category Change).
        await self._run_with_retry(self._delete_if_exists, source)

    # ---- Internals (run on thread pool via asyncio.to_thread) --------- #

    def _resolve_target(
        self, user_id: str, memory_id: str, category: str
    ) -> Path:
        """Validate segments + resolve full path under root.

        Raises :class:`SecurityError` on any traversal or symlink finding.
        """
        _reject_segment(user_id, field="user_id")
        _reject_segment(memory_id, field="memory_id")
        _reject_segment(category, field="category")

        root = self._root.resolve()

        # Path = root / user_id / category / f"{id}.md". Segments are already
        # safe (traversal chars rejected above), so the only attack surface
        # is a symlink that was placed there out-of-band (hand-edit, previous
        # buggy version, malicious user with fs access). We check the two
        # ancestor dirs + the file itself. ``is_symlink()`` reports on the
        # path itself, not its target — a non-existent path returns False,
        # which is the right answer (we'll mkdir it below).
        user_dir = self._root / user_id
        category_dir = user_dir / category
        unresolved = category_dir / f"{memory_id}.md"

        if user_dir.is_symlink():
            raise SecurityError(
                f"user 目录 {user_dir} 是 symlink，拒绝写入"
            )
        if category_dir.is_symlink():
            raise SecurityError(
                f"category 目录 {category_dir} 是 symlink，拒绝写入"
            )
        if unresolved.is_symlink():
            raise SecurityError(f"目标 {unresolved} 是 symlink，拒绝写入")

        # Post-resolution traversal check. Even if the above ancestors aren't
        # symlinks, this catches ``memory_id`` smuggling some path trick that
        # slipped past ``_reject_segment`` (defense in depth). In practice the
        # segment filter already covers it.
        candidate = unresolved.resolve()
        if not candidate.is_relative_to(root):
            raise SecurityError(
                f"路径穿越：{candidate} 不在 memory_root {root} 之下"
            )
        return candidate

    def _write_atomic(
        self, target: Path, content: str, overwrite: bool
    ) -> None:
        """Tmp-write + fsync + os.replace. Runs on threadpool."""
        target.parent.mkdir(parents=True, exist_ok=True)
        if not overwrite and target.exists():
            raise FileExistsError(f"{target} 已存在且 overwrite=False")

        encoded = content.encode("utf-8")
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(encoded)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, target)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "FsMemoryWriter 清理 tmp 文件 %s 失败（忽略）", tmp_path
                )
            raise

    def _delete_if_exists(self, target: Path) -> None:
        """Idempotent unlink. Missing file is not an error (design §Delete Flow)."""
        try:
            target.unlink()
        except FileNotFoundError:
            return

    # ---- Retry wrapper ----------------------------------------------- #

    async def _run_with_retry(self, fn, *args) -> Any:
        """Invoke ``fn(*args)`` on the threadpool with exponential backoff.

        Retryable: generic ``OSError`` that isn't ``PermissionError`` /
        ``IsADirectoryError`` / ``NotADirectoryError`` / ``FileExistsError``
        (all of those are programmer/caller errors, retrying won't help).
        Non-retryable: ``SecurityError`` (already raised by ``_resolve_target``
        before this wrapper runs in all intended paths, but belt-and-braces).
        """
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return await asyncio.to_thread(fn, *args)
            except (PermissionError, IsADirectoryError, NotADirectoryError, FileExistsError):
                raise
            except SecurityError:
                raise
            except OSError as exc:
                last_exc = exc
                if attempt == self._max_retries - 1:
                    break
                delay = self._base_backoff_seconds * (2 ** attempt)
                logger.warning(
                    "FsMemoryWriter fs op 失败 attempt=%d/%d err=%s 等待 %.2fs 重试",
                    attempt + 1,
                    self._max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
        assert last_exc is not None  # noqa: S101 — loop guarantees this
        raise last_exc
