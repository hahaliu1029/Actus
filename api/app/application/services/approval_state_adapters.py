"""R5 CS4 ApprovalStateReader 的 Protocol 适配器。

Domain ``ApprovalStateReader`` 用 ``ApprovalGrantQuery`` Protocol 抽象 grant
数据源。本模块把生产路径的 SQLAlchemy UoW 包装成该 Protocol，供
``service_dependencies.get_approval_state_reader`` 注入。

PE-4d1：旧 tool_approval_rules 表的 legacy rule 适配器已删除。

adapter 放 application 层（而非 infrastructure）：
- 职责是"桥接 domain protocol ↔ infra repo"，属于编排粒度
- 和 ``ApprovalStateWriter`` 同层，方便 DI 注入点统一
- application 层允许 import SQLAlchemy / infrastructure（CLAUDE.md 约束）
"""

from __future__ import annotations

from typing import Callable, Optional

from app.domain.models.approval_grant import ApprovalGrant
from app.domain.repositories.uow import IUnitOfWork

UoWFactory = Callable[[], IUnitOfWork]


class UowApprovalGrantQuery:
    """把 ``IUnitOfWork.approval_grants.find_active_grants`` 包装成
    ``ApprovalGrantQuery`` Protocol。

    每次 ``find_active_grants`` 调用都打开新 UoW（进 ``__aenter__`` → 调 repo
    → 退出时 commit/rollback，这里是只读所以只触发一次 SELECT）。
    Phase 1 Reader 的调用频率按 tool_call 级，独立事务边界清晰，不会压垮连接池。
    """

    def __init__(self, uow_factory: UoWFactory) -> None:
        self._uow_factory = uow_factory

    async def find_active_grants(
        self,
        user_id: str,
        session_id: Optional[str],
        tool_name: str,
    ) -> list[ApprovalGrant]:
        async with self._uow_factory() as uow:
            return await uow.approval_grants.find_active_grants(
                user_id=user_id,
                session_id=session_id,
                tool_name=tool_name,
            )
