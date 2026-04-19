from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class FileMemoryStore(Protocol):
    """Memory 文件系统侧的写入协议。

    路径规范：``${MEMORY_ROOT_CONTAINER}/{user_id}/{category}/{memory_id}.md``。
    Domain 层仅依赖此接口；PR-5A 由 ``FsMemoryWriter`` 提供真实实现。
    """

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
        """写入 / 覆盖单条 memory 文件。

        - ``overwrite=False`` 且目标已存在时实现方应抛错。
        - 实现必须保证 frontmatter + content 原子落盘（tmp + rename）。
        """
        ...

    async def delete(
        self,
        user_id: str,
        memory_id: str,
        category: str,
    ) -> None:
        """删除单条 memory 文件。目标不存在时幂等返回，不抛错。"""
        ...

    async def move_category(
        self,
        user_id: str,
        memory_id: str,
        from_category: str,
        to_category: str,
        content: str,
        new_frontmatter: dict,
    ) -> None:
        """category 变更：先写新路径、再删旧路径。

        ``content`` + ``new_frontmatter`` 由 service 层提供——writer 侧不再
        "读旧文件 + 搬 bytes"，那样会把旧 ``category: X`` 残留写到新路径下
        的 YAML 里，路径和 frontmatter 自相矛盾。由 service 构造新 frontmatter
        (category 已更新) 和 content，writer 只负责原子落盘 + 清旧。
        """
        ...

    async def read(
        self,
        user_id: str,
        memory_id: str,
        category: str,
    ) -> tuple[dict, str]:
        """读取 memory 文件并解析 frontmatter 与 body。

        返回 ``(frontmatter_dict, body_str)``。

        语义（reindex endpoint 的底层）：
        - 文件不存在 → raise ``FileNotFoundError``（service 层 map 成 409）
        - frontmatter YAML 解析失败 → raise ``ValueError``（→ 400）
        - 路径穿越 / symlink → raise ``SecurityError``（→ 403，与 write 路径同源防御）

        读侧的 symlink 防御必须复用 write 侧的 ``_resolve_target`` 校验——
        不允许 host 预植 symlink 通过 reindex 路径被 service 拿到 bytes 以外
        的 target 内容（与 MemoryMountScope 客户端守卫对偶但独立）。
        """
        ...


class NoopFileMemoryStore(FileMemoryStore):
    """DB-only 模式 / 单测用的空实现。所有方法都是 no-op。

    PR-0 ~ PR-5A 的中间态 `MemoryManagementService` 在 `file_store=None` 时
    降级为 DB-only；显式注入本类则让调用点不必写 `if file_store is not None:`。

    显式继承 ``FileMemoryStore`` 以便 mypy 在类定义时校验签名一致。
    """

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
        return None

    async def delete(
        self,
        user_id: str,
        memory_id: str,
        category: str,
    ) -> None:
        return None

    async def move_category(
        self,
        user_id: str,
        memory_id: str,
        from_category: str,
        to_category: str,
        content: str,
        new_frontmatter: dict,
    ) -> None:
        return None

    async def read(
        self,
        user_id: str,
        memory_id: str,
        category: str,
    ) -> tuple[dict, str]:
        raise NotImplementedError(
            "NoopFileMemoryStore 不支持 read —— DB-only 模式下没有 fs 侧数据可读。"
            " reindex endpoint 需要 file_store 注入 FsMemoryWriter 才能工作。"
        )
