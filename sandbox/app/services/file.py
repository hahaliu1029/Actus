import asyncio
import errno
import glob
import logging
import os.path
import re
import stat
import tempfile
from typing import Optional

from fastapi import UploadFile

from app.interfaces.errors.exceptions import (
    AppException,
    BadRequestException,
    NotFoundException,
)
from app.core.workspace import (
    deny_service_tree_write,
    is_within_workspace,
    resolve_in_workspace,
)
from app.models.file import (
    FileCheckResult,
    FileDeleteResult,
    FileFindResult,
    FileKind,
    FileReadResult,
    FileReplaceResult,
    FileSearchResult,
    FileUploadResult,
    FileWriteResult,
)

logger = logging.getLogger(__name__)


def _read_umask_once() -> int:
    """Read the process umask without leaving it changed.

    Snapshotted once at import (uvicorn runs ``--workers 1`` under
    supervisord) to avoid the process-global ``os.umask()`` read-modify-read
    race if it were done per-write.
    """
    old = os.umask(0)
    os.umask(old)
    return old


_UMASK = _read_umask_once()
# Parity with the old ``open(path, "wb")`` new-file mode (~0o644 under the
# default 0o022 umask).
_NEW_FILE_MODE = 0o666 & ~_UMASK


def _write_all(fd: int, data: bytes) -> int:
    """Write all of ``data`` to ``fd`` via the raw ``os.write`` loop.

    Handles short writes (the kernel may write fewer bytes than requested) and
    uses no buffered layer, so a subsequent ``os.fsync`` covers every byte.
    Returns the number of bytes written.
    """
    mv = memoryview(data)
    off = 0
    while off < len(mv):
        off += os.write(fd, mv[off:])
    return len(mv)


def _direct_write_through(target: str, source_chunks) -> int:
    """Legacy non-atomic write-through — the old ``open(target, "wb")`` path.

    Used ONLY for an EXISTING special file (FIFO / socket / device, incl.
    ``/dev/null``) where an atomic replace would DESTROY the node (D12).
    Atomicity is meaningless for a stream/device, so today's pass-through
    semantics are preserved. Returns the number of bytes written.
    """
    n = 0
    with open(target, "wb") as f:
        for chunk in source_chunks:
            f.write(chunk)
            n += len(chunk)
    return n


def _classify(st) -> "FileKind":
    """TOTAL inode classifier: every st_mode → exactly one FileKind."""
    m = st.st_mode
    if stat.S_ISREG(m):
        return "regular"
    if stat.S_ISDIR(m):
        return "directory"
    if stat.S_ISLNK(m):
        return "symlink"  # lstat → the link itself, never its referent
    if stat.S_ISFIFO(m):
        return "fifo"
    if stat.S_ISSOCK(m):
        return "socket"
    if stat.S_ISBLK(m):
        return "block"
    if stat.S_ISCHR(m):
        return "char"
    return "other"  # platform/unknown types (S_ISDOOR/PORT/WHT, …)


def _is_special(st_mode) -> bool:
    """The exact set 2a rejects + 2b refuses: FIFO/socket/block/char."""
    return (
        stat.S_ISFIFO(st_mode)
        or stat.S_ISSOCK(st_mode)
        or stat.S_ISBLK(st_mode)
        or stat.S_ISCHR(st_mode)
    )


def _apply_target_mode(fd: int, target: str) -> None:
    """Set the temp fd's mode to match the overwrite target (D10).

    - missing target -> new-file mode (parity with old ``open(path, "wb")``)
    - existing regular file -> preserve permission bits ONLY, stripping
      setuid/setgid/sticky (owner is not preserved and the sandbox runs as
      root, so copying a 04xxx bit would mint a setuid-root file)
    - symlink -> new-file mode (special files never reach here; D12 handles
      them before this is called)
    """
    try:
        st = os.lstat(target)  # lstat: do NOT follow a symlink target
    except FileNotFoundError:
        os.fchmod(fd, _NEW_FILE_MODE)
        return
    if stat.S_ISREG(st.st_mode):
        os.fchmod(fd, stat.S_IMODE(st.st_mode) & 0o777)
    else:
        os.fchmod(fd, _NEW_FILE_MODE)


def _atomic_write_bytes(
    target: str, source_chunks, *, refuse_special: bool = False
) -> int:
    """Atomically overwrite ``target`` with the concatenation of
    ``source_chunks`` (a bytes iterable). Returns the number of bytes written.

    Recipe: same-dir ``tempfile.mkstemp`` -> ``_write_all`` -> ``os.fsync`` ->
    ``os.replace`` (POSIX-atomic visibility — a concurrent reader sees
    old-complete or new-complete, never truncated). On any exception BEFORE
    ``os.replace`` the original target is left intact and the temp is
    best-effort unlinked (a rare unlink failure logs a warning and may leave a
    benign ``.actus-tmp-*`` orphan — visible, never the final path) without
    masking the original error.

    Final-component symlinks are REPLACED in place (D9, not followed).
    Existing special files (FIFO/socket/device) fall back to a non-atomic
    direct write-through (D12) so ``os.replace`` does not clobber the node
    (single-writer assumption per the S1 spec; a concurrent swap of ``target``
    between the ``lstat`` dispatch and ``os.replace`` is out of scope).

    When ``refuse_special=True`` (the coordinator apply/seed path) a direct
    special target is REFUSED with ``OSError(EINVAL)`` instead of the D12
    write-through — this restores the atomic-or-raise contract and closes the
    TOCTOU FIFO-hang. The agent ``write_file`` path keeps ``refuse_special=False``
    → D12 unchanged.
    """
    parent = os.path.dirname(target)
    # Bare-path rejection FIRST: os.makedirs("") raises, exactly as today, so
    # even a bare special-file target fails before the D12 fallback.
    os.makedirs(parent, exist_ok=True)
    try:
        st = os.lstat(target)
        if not (stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode)):
            if refuse_special and _is_special(st.st_mode):
                raise OSError(
                    errno.EINVAL,
                    f"refusing to write special file "
                    f"(atomic write required): {target!r}",
                )
            return _direct_write_through(target, source_chunks)
    except FileNotFoundError:
        pass  # missing -> atomic create below
    fd = tmp = None
    try:
        # mkstemp in the TARGET's own dir -> same filesystem -> os.replace is
        # an intra-FS swap, never EXDEV.
        fd, tmp = tempfile.mkstemp(dir=parent, prefix=".actus-tmp-")
        _apply_target_mode(fd, target)
        n = 0
        for chunk in source_chunks:
            n += _write_all(fd, chunk)
        os.fsync(fd)
        _fd, fd = fd, None  # null BEFORE close -> cleanup never double-closes
        os.close(_fd)
        os.replace(tmp, target)  # ATOMIC visibility; replaces a symlink too
        # parent-dir fsync deliberately omitted (durability, not atomicity)
        return n
    except BaseException:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass  # best-effort; never mask the original
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                # best-effort cleanup; never mask the original. A rare unlink
                # failure leaves a benign ``.actus-tmp-*`` orphan (visible,
                # never the final path) — surface it but do not raise. The
                # logging call itself is guarded so a misbehaving log
                # handler/formatter can never replace the in-flight error.
                try:
                    logger.warning("原子写清理临时文件失败，可能残留: %s", tmp)
                except Exception:
                    pass
        raise


class FileService:
    """文件沙箱服务"""

    def __init__(self) -> None:
        pass

    @classmethod
    async def read_file(
        cls,
        filepath: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        sudo: bool = False,
        max_length: Optional[int] = 10000,
    ) -> FileReadResult:
        """根据传递的文件路径+起始行号+权限+最大长度读取文件内容"""
        # 不支持文本预览的二进制文件扩展名
        BINARY_EXTENSIONS = {
            ".pdf", ".pptx", ".ppt", ".docx", ".doc", ".xlsx", ".xls",
            ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
            ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg",
            ".mp3", ".mp4", ".avi", ".mov", ".mkv", ".wav", ".flac",
            ".exe", ".dll", ".so", ".dylib", ".bin", ".dat",
            ".woff", ".woff2", ".ttf", ".otf", ".eot",
            ".sqlite", ".db",
        }

        try:
            target = resolve_in_workspace(filepath)
            # 1.检测在当前权限下能否获取该文件
            if not os.path.exists(target) and not sudo:
                logger.error(f"要读取的文件不存在或无权限: {filepath}")
                raise NotFoundException(f"要读取的文件不存在或无权限: {filepath}")

            # 1.5 检测二进制文件，返回友好提示而不是崩溃
            ext = os.path.splitext(filepath)[1].lower()
            if ext in BINARY_EXTENSIONS:
                file_size = os.path.getsize(target) if os.path.exists(target) else 0
                size_str = (
                    f"{file_size / 1024 / 1024:.1f} MB" if file_size > 1024 * 1024
                    else f"{file_size / 1024:.1f} KB" if file_size > 1024
                    else f"{file_size} B"
                )
                return FileReadResult(
                    filepath=filepath,
                    content=f"[二进制文件] {os.path.basename(filepath)} ({size_str})\n"
                    f"该文件为 {ext} 格式，不支持文本预览。\n"
                    f"可通过下载功能获取原始文件。",
                )

            # 2.ubuntu系统下统一使用utf-8编码
            encoding = "utf-8"

            # 3.判断是否为sudo，如果是sudo系统则使用命令行的形式读取文件
            if sudo:
                # 4.使用sudo cat命令读取文件内容
                command = f"sudo cat '{target}'"
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                # 5.读取子进程的输出，并等待子进程结束
                stdout, stderr = await process.communicate()

                # 6.判断子进程的状态是否正常结束
                if process.returncode != 0:
                    raise BadRequestException(f"阅读文件失败: {stderr.decode()}")

                # 7.读取输出内容
                content = stdout.decode(encoding, errors="replace")
            else:
                # 8.创建一个内部读取函数
                def async_read_file() -> str:
                    try:
                        with open(target, "r", encoding=encoding) as f:
                            return f.read()
                    except Exception as async_read_file_exception:
                        raise AppException(
                            msg=f"读取文件失败: {str(async_read_file_exception)}"
                        )

                # 9.使用asyncio创建线程读取文件
                content = await asyncio.to_thread(async_read_file)

            # 10.判断是否传递了读取范围
            if start_line is not None or end_line is not None:
                # 11.将内容切割成行，并且提取指定范围行号的数据
                lines = content.splitlines()
                start = start_line if start_line is not None else 0
                end = end_line if end_line is not None else len(lines)
                content = "\n".join(lines[start:end])

            # 12.裁切下数据长度
            if max_length is not None and 0 < max_length < len(content):
                content = content[:max_length] + "(truncated)"

            return FileReadResult(filepath=filepath, content=content)
        except Exception as e:
            # 13.判断异常类型执行不同操作
            if isinstance(e, BadRequestException) or isinstance(e, AppException):
                raise
            raise AppException(f"文件读取失败: {str(e)}")

    @classmethod
    async def write_file(
        cls,
        filepath: str,
        content: str,
        append: bool = False,
        leading_newline: bool = False,
        trailing_newline: bool = False,
        sudo: bool = False,
    ) -> FileWriteResult:
        """根据传递的文件路径+内容向指定文件写入内容"""
        try:
            # 1.组装实际写入的内容
            if leading_newline:
                content = "\n" + content
            if trailing_newline:
                content = content + "\n"

            target = resolve_in_workspace(filepath)

            # 2.判断是否是sudo权限，如果是则使用命令行的形式先写入一个缓存文件，然后将缓存文件覆盖原始文件
            if sudo:
                # sudo follows the final redirect target -> follows_final_symlink
                deny_service_tree_write(target, follows_final_symlink=True)
                # 3.使用命令的方式先向临时文件写入数据，计算追加模式
                mode = ">>" if append else ">"

                # 4.创建一个临时文件
                temp_file = f"/tmp/file_write_{os.getpid()}.tmp"

                # 5.创建一个内部函数使用asyncio创建新线程写入数据
                def async_write_temp_file() -> int:
                    with open(temp_file, "w", encoding="utf-8") as f:
                        f.write(content)
                    return len(content.encode("utf-8"))

                # 6.使用asyncio创建子线程并写入
                bytes_written = await asyncio.to_thread(async_write_temp_file)

                # 7.使用命令行将临时文件写入到目标哦文件中
                # NOTE (N6): the sudo shell-injection (unquoted target) is the
                # SEPARATE already-tracked ticket — out of scope here. Guard B
                # above stops an HONEST absolute /sandbox sudo write.
                command = f'sudo bash -c "cat {temp_file} {mode} {target}"'
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                # 8.等待子进程执行完毕
                stdout, stderr = await process.communicate()

                # 9.检测子进程是否正常执行
                if process.returncode != 0:
                    raise BadRequestException(f"文件内容写入失败: {stderr.decode()}")

                # 10.清除下临时文件
                os.unlink(temp_file)
            else:
                # 11.非 sudo 写入：覆盖走原子 helper，追加保持旧的非原子语义
                if append:
                    # append FOLLOWS a final symlink -> follows_final_symlink=True
                    deny_service_tree_write(target, follows_final_symlink=True)
                    # append 不可原子化（rename 无法追加）——保持最佳努力写入，
                    # bytes_written 仍是 f.write 的字符数（追加是 non-goal）。
                    os.makedirs(os.path.dirname(target), exist_ok=True)

                    def async_append_file() -> int:
                        with open(target, "a", encoding="utf-8") as f:
                            return f.write(content)

                    bytes_written = await asyncio.to_thread(async_append_file)
                else:
                    # 12.覆盖写：mkstemp + fsync + os.replace 原子落盘（S1）。
                    #    atomic overwrite REPLACES the final component in place
                    #    -> follows_final_symlink=False。返回字节数
                    #    （len(encode)），与 sudo 分支对齐——非 ASCII 内容下与旧的
                    #    f.write 字符数不同（H4，已文档化）。
                    deny_service_tree_write(target, follows_final_symlink=False)
                    bytes_written = await asyncio.to_thread(
                        _atomic_write_bytes,
                        target,
                        [content.encode("utf-8")],
                    )

            return FileWriteResult(
                filepath=filepath,  # echo-original
                bytes_written=bytes_written,
            )
        except Exception as e:
            # 14.根据不同的错误执行不同的操作
            logger.error(f"文件内容写入失败: {str(e)}")
            if isinstance(e, BadRequestException):
                raise
            raise AppException(f"文件内容写入失败: {str(e)}")

    async def replace_in_file(
        self,
        filepath: str,
        old_str: str,
        new_str: str,
        sudo: bool = False,
    ) -> FileReplaceResult:
        """根据传递的数据替换文件内指定的内容"""
        # 1.调用服务获取对应的文件内容
        file_read_result = await self.read_file(
            filepath=filepath, sudo=sudo, max_length=None
        )
        content = file_read_result.content

        # 2.计算old_str出现的次数，只有出现次数>0才需要替换
        replaced_count = content.count(old_str)
        if replaced_count == 0:
            return FileReplaceResult(filepath=filepath, replaced_count=replaced_count)

        # 3.替换旧内容
        new_content = content.replace(old_str, new_str)

        # 4.将替换后的新内容写入到文件中
        await self.write_file(
            filepath=filepath,
            content=new_content,
            sudo=sudo,
        )

        return FileReplaceResult(filepath=filepath, replaced_count=replaced_count)

    async def search_in_file(
        self,
        filepath: str,
        regex: str,
        sudo: bool = False,
    ) -> FileSearchResult:
        """根据传递的文件路径+匹配规则查询文件内符合的内容"""
        # 1.调用服务获取对应的文件内容
        file_read_result = await self.read_file(
            filepath=filepath, sudo=sudo, max_length=None
        )
        content = file_read_result.content

        # 2.将读取的内容拆分成每一行
        lines = content.splitlines()
        matches = []
        line_numbers = []

        # 3.将外部传递的regex转换为正则
        try:
            pattern = re.compile(regex)
        except Exception as e:
            raise BadRequestException(f"传递正则表达式[{regex}]出错: {str(e)}")

        # 4.创建一个异步函数，使用子线程方式执行避免长时间io阻塞
        def async_matches():
            nonlocal matches, line_numbers
            for idx, line in enumerate(lines):
                if pattern.match(line):
                    matches.append(line)
                    line_numbers.append(idx)

        # 5.使用asyncio创建子线程并调用
        await asyncio.to_thread(async_matches)

        return FileSearchResult(
            filepath=filepath,
            matches=matches,
            line_numbers=line_numbers,
        )

    @classmethod
    async def find_files(cls, dir_path: str, glob_pattern: str) -> FileFindResult:
        """根据传递的文件夹路径+glob规则查询文件列表"""
        original_dir = dir_path

        # (1) glob_pattern hygiene (uniform): an absolute or ..-bearing pattern
        #     would let os.path.join's second arg win and escape the anchored
        #     dir. Reject it (4xx).
        if os.path.isabs(glob_pattern) or ".." in glob_pattern.split("/"):
            raise BadRequestException(
                f"非法的 glob 模式(绝对路径或包含 ..): {glob_pattern!r}"
            )

        absolute_dir = os.path.isabs(dir_path)
        if absolute_dir:
            # ABSOLUTE dir_path -> pass-through, NOT filtered (G6).
            resolved_dir = dir_path
        else:
            # RELATIVE (incl ""/".") -> confine, resolving the final component.
            resolved_dir = resolve_in_workspace(dir_path, follow_final=True)

        if not os.path.exists(resolved_dir):
            raise NotFoundException(f"当前文件夹不存在: {original_dir}")

        def async_glob():
            search_pattern = os.path.join(resolved_dir, glob_pattern)
            return glob.glob(search_pattern, recursive=True)

        files = await asyncio.to_thread(async_glob)

        if absolute_dir:
            # absolute-in -> absolute-out, unfiltered.
            return FileFindResult(dir_path=original_dir, files=files)

        # RELATIVE: (3) drop any result that escapes via an inner symlink, then
        # re-base to the ORIGINAL dir (relative-in -> relative-out, D5).
        confined = [f for f in files if is_within_workspace(f)]
        rebased = [
            os.path.normpath(os.path.join(original_dir, os.path.relpath(f, resolved_dir)))
            for f in confined
        ]
        return FileFindResult(dir_path=original_dir, files=rebased)

    @classmethod
    async def upload_file(
        cls, file: UploadFile, filepath: str, *, refuse_special: bool = False
    ) -> FileUploadResult:
        """根据传递的文件源+路径将文件上传至沙箱。

        ``refuse_special=True``（仅 coordinator apply/seed 路径）会让底层原子写
        在目标是直接 special 文件（FIFO/socket/block/char）时 raise 而非走 D12
        非原子写穿透——关闭 TOCTOU FIFO 挂起。默认 False，agent 上传不受影响。
        """
        try:
            target = resolve_in_workspace(filepath)
            deny_service_tree_write(target, follows_final_symlink=False)

            # 1.分块读取上传内容，每次最多 8K
            chunk_size = 1024 * 8

            def _source_chunks():
                while True:
                    chunk = file.file.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

            # 2.通过共享的原子写 helper 落盘（mkstemp + fsync + os.replace）——
            #   崩溃/中途异常不会留下截断文件（S1）。
            file_size = await asyncio.to_thread(
                _atomic_write_bytes,
                target,
                _source_chunks(),
                refuse_special=refuse_special,
            )

            return FileUploadResult(
                filepath=filepath,  # echo-original
                file_size=file_size,
                success=True,
            )
        except Exception as e:
            logger.error(f"上传文件到沙箱出错: {str(e)}")
            if isinstance(e, BadRequestException):
                raise  # OutsideWorkspaceError / ServiceTreeWriteDenied -> 4xx
            raise AppException(f"上传文件到沙箱出错: {str(e)}")

    @classmethod
    async def ensure_file(cls, filepath: str) -> None:
        """传递filepath用于确保当前文件存在"""
        target = resolve_in_workspace(filepath)
        if not os.path.exists(target):
            raise NotFoundException(f"该文件不存在: {filepath}")

    @classmethod
    async def check_file_exists(cls, filepath: str) -> FileCheckResult:
        """根据传递的路径判断文件是否存在 + inode 类型分类（lstat）。

        ``exists`` 保留 ``os.path.exists`` 语义（跟随符号链接、破损链接→False、
        吞掉所有 OSError）。``kind`` 由 ``os.lstat`` 独立计算，且必须永不令 RPC 失败：
        非 ENOENT 的 OSError（EACCES/ELOOP/ENOTDIR…）退化为 ``other``（fail-open，
        非 special → 2a 照常放行），与 S1b 前行为一致。
        """
        target = resolve_in_workspace(filepath)
        exists = os.path.exists(target)
        try:
            kind = _classify(os.lstat(target))
        except FileNotFoundError:
            kind = "missing"
        except OSError:
            kind = "other"
        return FileCheckResult(filepath=filepath, exists=exists, kind=kind)  # echo-original filepath

    async def delete_file(self, filepath: str) -> FileDeleteResult:
        """根据传递的路径删除指定文件（幂等）。

        丢弃旧的 ensure_file check-then-act（TOCTOU + 非幂等）：删除一个
        已不存在的文件视为成功（terminal-absent，ENOENT→success）。其它
        OSError（EACCES/EISDIR/EROFS 等）包成 AppException 上抛——保留具体
        诊断信息，与 write_file/upload_file 一致（delete endpoint 不再额外
        包装，raw OSError 会退化成全局 500 泛化文案，见 Deviation D-2）。
        RPC success=True 表示"终态：文件不存在"（S1 §3.4）。
        """
        target = resolve_in_workspace(filepath)
        deny_service_tree_write(target, follows_final_symlink=False)

        def _rm() -> None:
            try:
                os.remove(target)
            except FileNotFoundError:
                pass  # 已不存在 == 成功（幂等）
            except OSError as e:
                # 非 ENOENT 错误包成 AppException，保留具体诊断信息
                raise AppException(f"删除文件{filepath}失败: {e}")

        await asyncio.to_thread(_rm)
        return FileDeleteResult(filepath=filepath, deleted=True)  # echo-original
