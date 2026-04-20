"""R5 CS4 单一 Writer：grants + audit log 原子写入。

契约：
- ``write(decision) -> (decision_id, newly_created: bool)``
- ``newly_created=True`` 表示本调用赢得了 UNIQUE claim，caller 可安全触发
  ``task.resume()``；``False`` 表示 confirmation_id 或 SmartApprove partial
  UNIQUE 已被别人赢走，已存在的 grant 直接幂等返回
- ``(scope, effect)`` 不匹配的冲突抛 ``ValueError``（deny→approve 这类语义矛盾）
- ``delete_grant(decision_id)`` 在 ``task.resume()`` kickoff 失败时同事务删
  grant + audit log，保证下一次 ``/resume`` 重新成为 winning claim

这是 CS4 的**唯一**写入路径（AST guard ``test_cs4_single_writer`` 锁住）。
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

from sqlalchemy.exc import IntegrityError

from app.domain.models.approval_grant import ApprovalDecision, ApprovalGrant
from app.domain.repositories.uow import IUnitOfWork

logger = logging.getLogger(__name__)

UoWFactory = Callable[[], IUnitOfWork]


class ApprovalStateWriter:
    """CS4 单一 Writer。

    Phase 1 scope: ``write`` + ``delete_grant``。Revoke / policy_id 等是 Phase 2
    PermissionEngine 的职责，这里不涉及。
    """

    def __init__(self, uow_factory: UoWFactory) -> None:
        self._uow_factory = uow_factory

    async def write(self, decision: ApprovalDecision) -> Tuple[str, bool]:
        """原子写入 grant + audit log。返回 ``(decision_id, newly_created)``.

        UNIQUE 冲突分叉：
        1. ``confirmation_id`` 非 None → 查主 UNIQUE
           - 同 ``(scope, effect)`` → 返 existing（幂等）
           - 不同 ``(scope, effect)`` → raise ``ValueError``（语义冲突）
        2. ``confirmation_id`` 为 None (SmartApprove) → 查 partial UNIQUE
           - 命中 → 返 existing（去重）
           - 未命中 → 重新 raise 原 IntegrityError（意外情况）
        """
        async with self._uow_factory() as uow:
            try:
                decision_id = await uow.approval_grants.create(decision)
                await uow.tool_approval_log.create(
                    user_id=decision.user_id,
                    session_id=decision.session_id or "",
                    tool_name=decision.tool_name,
                    tool_args={},
                    risk_level=decision.risk_level,
                    action=decision.effect,
                    scope=decision.scope,
                    approved_by=(
                        "user"
                        if decision.source_type == "user_click"
                        else decision.source_type
                    ),
                    decision_id=decision_id,
                )
                return decision_id, True
            except IntegrityError:
                # UNIQUE 冲突：回滚当前事务后走幂等回读
                await uow.rollback()
                existing: Optional[ApprovalGrant]
                if decision.confirmation_id is not None:
                    existing = await uow.approval_grants.find_by_confirmation_id(
                        decision.confirmation_id
                    )
                else:
                    existing = await uow.approval_grants.find_smart_approve_dedup(
                        user_id=decision.user_id,
                        session_id=decision.session_id,
                        tool_name=decision.tool_name,
                        arg_digest=decision.arg_digest,
                        effect=decision.effect,
                    )
                if existing is None:
                    # 意外 IntegrityError：不吞掉，交给上层排查
                    raise
                if existing.scope != decision.scope or existing.effect != decision.effect:
                    raise ValueError(
                        "claim collision: existing "
                        f"scope={existing.scope}/effect={existing.effect} "
                        f"vs new scope={decision.scope}/effect={decision.effect}"
                    )
                return existing.decision_id, False

    async def delete_grant(self, decision_id: str) -> None:
        """回滚路径：同事务删 grant + audit log 行（R5b kickoff 失败场景）。"""
        async with self._uow_factory() as uow:
            await uow.tool_approval_log.delete_by_decision_id(decision_id)
            await uow.approval_grants.delete(decision_id)
