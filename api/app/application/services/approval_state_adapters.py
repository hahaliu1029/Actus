"""R5 CS4 ApprovalStateReader 的 Protocol 适配器。

Domain ``ApprovalStateReader`` 用 ``ApprovalGrantQuery`` 和 ``LegacyRuleQuery``
两个 Protocol 抽象 grant 数据源和 legacy rule 数据源。本模块把生产路径的
SQLAlchemy UoW / session_factory 包装成这两个 Protocol，供
``service_dependencies.get_approval_state_reader`` 注入。

adapter 放 application 层（而非 infrastructure）：
- 职责是"桥接 domain protocol ↔ infra repo"，属于编排粒度
- 和 ``ApprovalStateWriter`` 同层，方便 DI 注入点统一
- application 层允许 import SQLAlchemy / infrastructure（CLAUDE.md 约束）
"""

from __future__ import annotations

from typing import Callable, Optional

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.models.approval_grant import ApprovalGrant
from app.domain.repositories.uow import IUnitOfWork
from app.domain.services.approval_state_reader import CheckResult

UoWFactory = Callable[[], IUnitOfWork]
SessionFactory = async_sessionmaker[AsyncSession]


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


class SessionLegacyRuleQuery:
    """把 ``tool_approval_rules`` 表包装成 ``LegacyRuleQuery`` Protocol。

    过渡期适配器：Reader 在 grants 表 miss 后，按 AppConfig
    ``legacy_rule_fallback`` 开关查旧表。本 adapter 复用
    ``DBToolApprovalRuleRepository`` 做数据读取，在 adapter 内部应用
    ``always_deny > always_allow`` 优先级并把结果翻译成 ``CheckResult``。

    **不读 Redis session cache**：session scope 已由 grants 表的
    ``session_id + arg_digest`` 匹配接管，legacy Redis 键不再 honor。
    """

    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory

    async def check(
        self,
        user_id: str,
        tool_name: str,
        primary_arg: str,
        dir_arg: Optional[str],
    ) -> CheckResult:
        # import 放方法内，避免 adapter 文件 import 链在启动期拉入 DB repo
        from app.infrastructure.repositories.db_tool_approval_rule_repository import (
            DBToolApprovalRuleRepository,
        )

        async with self._session_factory() as session:
            repo = DBToolApprovalRuleRepository(session)
            rules = await repo.find_by_user_and_tool(user_id, tool_name)

        for rule in rules:
            if rule.rule == "always_deny" and rule.matches(primary_arg, dir_arg):
                return "deny"
        for rule in rules:
            if rule.rule == "always_allow" and rule.matches(primary_arg, dir_arg):
                return "allow"
        return "no_match"
