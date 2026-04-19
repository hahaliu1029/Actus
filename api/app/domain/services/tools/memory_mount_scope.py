"""MemoryMountScope — 客户端侧 symlink 守护的参数载体（codex fix P0）。

Design doc §724 Case C (iii)：sandbox ``:ro`` bind mount 挡不住 host 端预植
symlink 的读透穿。正式的读侧防御是"agent 工具拒绝跟随 memory mount 内的
symlink"，而不是依赖 ``FsReconciler.walk_user_directory`` 的异步收尾——
reconciler 是 fire-and-forget（见 ``session_service.py::_spawn_fs_reconciler_walk``），
session 创建 → sandbox 启动 → agent 可读之间存在真实窗口。

本模块提供 ``_make_file_tools`` 需要的最小参数集：host 端 memory 目录路径
（api 容器视角）+ user_id + sandbox 内的挂载目标路径。工具 wrapper 在收到
``file_read`` / ``file_str_replace`` / ``file_find_in_content`` 请求、且 filepath
落在 ``sandbox_target`` 下时，映射回 api 容器侧的实际路径做 ``lstat``：
- 是 symlink → 客户端直接抛 SecurityError，**不发 sandbox HTTP 请求**
- 否则 → 正常透传

为什么能做到：docker-compose.yml api service 已经把 ``MEMORY_ROOT_HOST`` bind
到 ``MEMORY_ROOT_CONTAINER``（PR-6A），api 容器内的 ``container_root / user_id``
看到的字节和 sandbox 看到的字节**完全同源**（host bind mount 在两处露出）。
所以客户端检查等价于 sandbox 内检查，而且免去了改 sandbox 镜像的成本。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryMountScope:
    """客户端侧 memory mount symlink 守护的配置。

    Attributes:
        container_root: api 容器视角的 MEMORY_ROOT_CONTAINER。例如
            ``/app/data/memory``。加上 ``user_id`` 后即 host 端 bind source
            在 api 容器内的挂载点，``lstat`` 这里等价于 sandbox 内 ``lstat``。
        user_id: 当前 session 的用户 id，白名单 UUID v4；scope 只在非空
            + 合法时创建（DockerSandbox._build_memory_mount 已有一致校验）。
        sandbox_target: sandbox 内固定挂载路径，例如 ``/workspace/.memory``。
            工具收到的 filepath 若以此为前缀，说明 agent 在读 memory mount 范围。

    不可变：frozen dataclass 避免 scope 被 tool 调用链意外改写。
    """

    container_root: Path
    user_id: str
    sandbox_target: str

    def map_to_host_path(self, sandbox_path: str) -> Path | None:
        """把 sandbox 内 filepath 映射到 api 容器侧的实际路径。

        返回 None 表示 path 不在 memory mount 范围内——调用方应跳过 symlink
        检查、正常透传工具请求（非 memory scope 的路径不归此守护管）。

        严格前缀匹配：``sandbox_target`` 必须是 path 的直接前缀 + 后续是
        ``/`` 或等于 target 本身（避免 ``/workspace/.memory_evil`` 误配）。
        """
        normalized = sandbox_path.rstrip("/")
        target = self.sandbox_target.rstrip("/")
        if normalized == target:
            return self.container_root / self.user_id
        prefix = target + "/"
        if not normalized.startswith(prefix):
            return None
        relative = normalized[len(prefix) :]
        # 额外防御：relative 部分不得含 ``..``，避免攻击者伪造
        # ``/workspace/.memory/../../etc/passwd`` 绕过。
        if ".." in Path(relative).parts:
            return None
        return self.container_root / self.user_id / relative

    def any_ancestor_is_symlink(self, sandbox_path: str) -> bool:
        """检测 path 在 memory scope 内的任一祖先（含 user_id 根）是否为 symlink。

        **仅查叶子不够**（codex fix P1 旁路）：若攻击者把 ``{user_id}/user`` 做成
        symlink 指向 ``/etc``，``lstat(/.../user/secret.txt)`` 只检查 ``secret.txt``
        是否为 symlink——它不是，但父目录是，整条路径仍透穿到 symlink target。

        从 ``container_root / user_id``（含自身）开始逐段 ``lstat``，任一命中
        即返 True。path 不在 scope 内 → False（跳过 guard，非守护职责）。

        异常保守语义：``OSError`` 视为"可疑"，返 True 让守护链 refuse——
        比起让 sandbox 去读一个状态不明的路径，保守拒绝更安全。
        """
        host_leaf = self.map_to_host_path(sandbox_path)
        if host_leaf is None:
            return False

        root = self.container_root / self.user_id
        try:
            relative = host_leaf.relative_to(root)
        except ValueError:
            # map_to_host_path 已保证 relative_to(root) 成立；走到这里
            # 说明参数不一致，保守拒绝。
            return True

        cursor = root
        # 从 root 自身开始检查——attacker 可能把 user_id 目录本身做成 symlink。
        # 用 list 头插 root，然后追加每个 relative part，逐段走。
        segments: list[Path] = [cursor]
        for part in relative.parts:
            cursor = cursor / part
            segments.append(cursor)

        for seg in segments:
            try:
                if seg.is_symlink():
                    return True
            except OSError as exc:
                # 不可 lstat 的路径（权限 / 文件系统损坏）→ 保守拒绝，避免
                # 降级为"透传给 sandbox" 暴露攻击面。
                logger.warning(
                    "memory_mount_scope guard: lstat 异常，保守拒绝 path=%s err=%s",
                    seg,
                    exc,
                )
                return True
        return False


def build_memory_mount_scope_from_settings(
    user_id: str | None,
    settings: Any,
) -> MemoryMountScope | None:
    """Shared factory — `PlannerReActFlow` 和 `AgentTaskRunner` 共用。

    codex fix P0：memory mount scope 的构造条件散在两处很容易漂移；之前只
    接 planner_react，漏了 agent_task_runner `_build_lc_tools_full` 这条
    step graph 实际绑定路径。用一个 factory 统一，确保任何新 tool-binding
    位点都用同一条件。

    条件：user_id 非空 **AND** ``settings.sandbox_memory_mount_enabled`` 为 True。
    条件不满足返 None——``_make_file_tools`` scope=None 时保持旧行为
    （适用于未启用 memory mount 的 deployment）。

    ``settings`` 用 ``Any`` 签名：production 里是 ``core.config.Settings``，
    测试里是 ``SimpleNamespace`` / dict-like，避免循环 import + 保持可测。
    """
    if not user_id:
        return None
    if not getattr(settings, "sandbox_memory_mount_enabled", False):
        return None
    return MemoryMountScope(
        container_root=Path(settings.memory_root_container).expanduser(),
        user_id=user_id,
        sandbox_target=str(settings.sandbox_memory_mount_target),
    )
