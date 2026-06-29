"""M3-B: Memory bind-mount kernel 级对抗测试。

设计文档 §698-724 / M3 P0 第 2 条。验证 ``/workspace/.memory`` read-only
bind mount 在 kernel 层真的阻挡三组攻击路径，独立于 Actus sandbox 镜像
本身——用 alpine:3.20 启一次性容器，配置与 ``DockerSandbox._build_memory_mount``
产出的 ``Mount(type='bind', read_only=True)`` 等价。

三组 case（`design §719-724`）：

- **Case A（agent 工具走 file_write sudo）**：容器内 ``touch`` 即便 root 运行
  也应返 ``Read-only file system`` 非零退出。**断言**：kernel-enforced ro
  bind mount 与容器 uid / sudo 权限无关。
- **Case B（agent 工具走 shell_execute sudo）**：``sh -c "echo x > ..."`` 同样
  失败。sudo 在 alpine 里不存在，但容器默认 root 身份已是最高权限——kernel
  ro 在 uid=0 下依然兜底。
- **Case C（host 端预置 symlink）**：host 侧在 ``user_dir`` 下放一个指向容器
  内敏感路径（如 ``/etc/hostname``）的 symlink，容器 ``cat`` 读这个 symlink
  会透过 bind mount 解析到容器 namespace 下的 target。这**是设计里标记的
  真实风险**，证明仅靠 ``:ro`` 拦不住；**正式防御是 FsReconciler
  ``walk_user_directory`` 把 symlink 移到 `.orphans/`**（见
  ``test_fs_reconciler.py::test_symlink_in_user_dir_quarantined``，已覆盖）。
  本 case 走完整流程：pre-plant → 确认读通 → 跑 reconciler → 再读 404。

**运行方式：**

```
cd api && uv run pytest tests/integration/sandbox/ -m sandbox
```

默认 CI 排除（见 ``pytest.ini`` addopts）。本地跑需：(1) docker daemon
可连；(2) 本地 cached ``alpine:3.20`` 镜像（``docker image ls`` 检查）。
不满足会 ``pytest.skip``，不会让默认套装挂。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.sandbox._docker_helpers import (
    ALPINE_IMAGE,
    _docker_client_or_skip,
    _require_image,
)

pytestmark = [pytest.mark.sandbox, pytest.mark.anyio]

# 容器内挂载点——与 ``settings.sandbox_memory_mount_target`` 默认值保持一致。
_MOUNT_TARGET = "/workspace/.memory"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _docker_or_skip():
    """Thin wrapper preserving the original behavior (client + alpine:3.20 require)
    on top of the shared helpers (C5d-1 refactor). Behavior-identical: same
    multi-socket client discovery, same REQUIRE=1 fail-not-skip, same image gate."""
    client = _docker_client_or_skip()
    _require_image(client, ALPINE_IMAGE)
    return client


def _run_alpine_with_ro_mount(
    client, host_dir: Path, cmd: str
) -> tuple[int, bytes]:
    """以 ro bind mount 启动一次性 alpine，运行 ``sh -c cmd``。返回 (exit_code, output_bytes)。

    与 DockerSandbox 生产路径的关键等价性：
    - ``Mount(type='bind', target=<MOUNT_TARGET>, source=host_dir, read_only=True)``
    - ``remove=True`` 容器退出即清理，不留悬挂
    - 默认 root 运行——重点就是 "即便 uid=0 也写不进"。
    """
    from docker.types import Mount

    container = client.containers.run(
        ALPINE_IMAGE,
        command=["sh", "-c", cmd],
        mounts=[
            Mount(
                target=_MOUNT_TARGET,
                source=str(host_dir),
                type="bind",
                read_only=True,
            )
        ],
        detach=True,
        remove=False,  # 保留以便读取 exit_code + logs；手动 remove
    )
    try:
        result = container.wait(timeout=30)
        exit_code = int(result.get("StatusCode", -1))
        logs = container.logs(stdout=True, stderr=True)
        return exit_code, logs
    finally:
        try:
            container.remove(force=True)
        except Exception:
            pass


class TestMemoryMountKernelInvariants:
    """Case A + B：kernel ro 拒写（与 uid / sudo 无关）。"""

    async def test_case_a_touch_under_mount_returns_erofs(
        self, tmp_path: Path
    ) -> None:
        """``touch /workspace/.memory/evil`` → EROFS，即便容器默认 root 身份。

        对应 design §719：agent 工具 ``file_write(sudo=True)`` 会在 sandbox 服务
        内部走 ``write(2)``；kernel 在 ro bind mount 上直接 EROFS，绕不过。
        """
        client = _docker_or_skip()
        exit_code, logs = _run_alpine_with_ro_mount(
            client,
            tmp_path,
            f"touch {_MOUNT_TARGET}/evil.md",
        )
        assert exit_code != 0, f"触发 EROFS 时容器必须非零退出，实际 exit={exit_code}, logs={logs!r}"
        assert b"Read-only file system" in logs, f"logs 缺少 EROFS 字样: {logs!r}"

    async def test_case_b_shell_redirect_under_mount_fails(
        self, tmp_path: Path
    ) -> None:
        """``sh -c "echo x > ..."`` → EROFS。

        对应 design §720：agent 走 ``shell_execute`` + 容器内 bash redirection，
        kernel 层仍然拒绝。alpine 没有 sudo，但容器默认 root 已覆盖 "最大权限
        仍写不进" 的断言语义。
        """
        client = _docker_or_skip()
        exit_code, logs = _run_alpine_with_ro_mount(
            client,
            tmp_path,
            f'sh -c "echo adversarial > {_MOUNT_TARGET}/evil.md"',
        )
        assert exit_code != 0, f"shell 重定向写 ro mount 必须失败: exit={exit_code}, logs={logs!r}"
        assert b"Read-only file system" in logs, f"logs 缺少 EROFS 字样: {logs!r}"


class TestMemoryMountSymlinkKernelReality:
    """Case C kernel 侧现实：单纯 ``:ro`` bind mount **不拦**读 symlink，仅作现象
    记录。**真正的防御**是客户端工具层 + host 端 reconciler 双层，见：

    - 客户端守护（codex fix P0）：``tests/app/domain/services/tools/
      test_memory_mount_scope_guard.py`` — agent 调用 ``file_read`` 时，若
      path 在 memory mount scope 内且 target 是 symlink → ``Denied``，
      **不发 sandbox HTTP**
    - reconciler 侧：``tests/app/infrastructure/external/memory/
      test_fs_reconciler.py::test_symlink_in_user_dir_quarantined`` — host
      端 symlink 扫描时被搬到 ``.orphans/``

    本测试不再作为"mitigation 有效"的 oracle，而是文档化 kernel 层面行为，
    防止误把 :ro 当作读侧防御。
    """

    async def test_ro_bind_mount_does_not_block_symlink_read(
        self, tmp_path: Path
    ) -> None:
        """记录 kernel 行为：host 预置 symlink → 容器 ``cat`` 透读 target 成功。

        这是**已知风险**，不是"预期安全行为"。客户端守护 + reconciler 负责
        真实拦截；本测试存在的唯一价值是防止有人把 :ro 误当作读侧防御——
        若某天 kernel 行为变了（symlink 不再被 follow），这测试会 fail，
        那时应当去掉此 test + 更新 design §704。
        """
        client = _docker_or_skip()

        user_dir = tmp_path / "user-evil"
        user_dir.mkdir()
        symlink = user_dir / "evil.md"
        try:
            os.symlink("/etc/hostname", symlink)
        except (OSError, NotImplementedError):
            pytest.skip("当前 fs 不支持 symlink")

        exit_code, logs = _run_alpine_with_ro_mount(
            client,
            user_dir,
            f"cat {_MOUNT_TARGET}/evil.md",
        )
        # 记录现象，非"expected secure behavior"。
        assert exit_code == 0, (
            f":ro bind mount 当前不拦 symlink 读——如果这里失败说明 kernel 行为"
            f"变了，需要更新 design §704 + 考虑去掉本测试: exit={exit_code}"
        )
        assert logs.strip(), "symlink target 读取后应有内容"

    async def test_reconciler_quarantines_host_planted_symlink(
        self, tmp_path: Path
    ) -> None:
        """验证第二层防御：host 端 reconciler 把 symlink 搬到 ``.orphans/``。

        与上方 kernel 测试独立：kernel 是"读时会 follow"的现象记录；本测试
        是"用户开新 session 时 reconciler 主动清理"的正常路径。两者正交。
        """
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock

        from app.infrastructure.external.memory.fs_reconciler import FsReconciler

        user_dir = tmp_path / "user-evil"
        user_dir.mkdir()
        symlink = user_dir / "evil.md"
        try:
            os.symlink("/etc/hostname", symlink)
        except (OSError, NotImplementedError):
            pytest.skip("当前 fs 不支持 symlink")

        repo = AsyncMock()
        repo.get_by_id.return_value = None
        repo.list_by_user.return_value = []
        repo.count_by_user.return_value = 0

        @asynccontextmanager
        async def fake_session_factory():
            yield AsyncMock()

        reconciler = FsReconciler(
            session_factory=fake_session_factory,
            repo_factory=lambda _s: repo,
            file_store=AsyncMock(),
            memory_root=tmp_path,
        )
        result = await reconciler.walk_user_directory("user-evil")

        assert result["orphan_symlinks"] >= 1
        assert not symlink.is_symlink(), "symlink 必须已从 user_dir 移走"
        assert (user_dir / ".orphans").exists(), "symlink 应被迁到 .orphans/"
