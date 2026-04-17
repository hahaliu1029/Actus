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
    ) -> None:
        """category 变更：先写新路径、再删旧路径。"""
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
    ) -> None:
        return None
