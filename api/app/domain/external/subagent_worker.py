"""C4 通用 SubagentWorker 执行 port（spec §4）。

mirror domain/external/task.py:TaskRunner(ABC)。具体 runtime 适配器实现
run()。DTO 在 domain/models/，执行 Protocol 在 domain/external/。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.models.subagent_worker import SubagentRunResult, WorkerSpec


class SubagentWorker(ABC):
    """统一子 worker 执行 port。具体 runtime 适配器实现 run()。"""

    @abstractmethod
    async def run(self, spec: WorkerSpec) -> SubagentRunResult:
        raise NotImplementedError
