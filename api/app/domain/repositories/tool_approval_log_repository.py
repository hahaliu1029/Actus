from abc import ABC, abstractmethod
from typing import Any, Optional


class ToolApprovalLogRepository(ABC):
    """审计日志仓储接口（只写，R5 起支持按 decision_id 回滚删除）"""

    @abstractmethod
    async def create(
        self,
        user_id: str,
        session_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        risk_level: str,
        action: str,
        scope: str,
        approved_by: str,
        decision_id: Optional[str] = None,
    ) -> None: ...

    @abstractmethod
    async def delete_by_decision_id(self, decision_id: str) -> None:
        """R5 CS4 writer 回滚路径：删除 grant 的同一 UoW 内删对应 audit 行。

        保持"grant + audit 同生同灭"对称，避免孤儿 audit 行（R5 writer.delete_grant
        在 ``task.resume()`` kickoff 失败时调用，使同 confirmation_id 的下次 resume
        重新 ``newly_created=True``）。
        """
        ...
