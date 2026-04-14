from abc import ABC, abstractmethod
from typing import Any


class ToolApprovalLogRepository(ABC):
    """审计日志仓储接口（只写）"""

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
    ) -> None: ...
