"""Sandbox 生命周期审计日志仓储接口（只写）

PR2 §10.5: K8s 风 terminal-immutable 承诺需要审计轨迹。
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional


class SandboxLifecycleLogRepository(ABC):
    """审计日志仓储接口（只写）"""

    @abstractmethod
    async def create(
        self,
        session_id: str,
        old_state: str,
        new_state: str,
        generation: int,
        sandbox_id: Optional[str] = None,
        reason: Optional[str] = None,
        triggered_by: Optional[str] = None,
    ) -> None: ...
