"""C4.1a 域仓库 ABC —— subagent-run 观测面（spec §3.3 / §6 INV-C4.1-4）。

observation-only：仅 ``record`` 写 + ``list_by_parent_session`` 读，无
cancel/retry/start/delete/permission（false-unification 护栏）。domain 约束：
仅 stdlib + 同层 ``app.domain.*``；禁写框架名子串（AST+子串纯度守门）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.models.subagent_run_record import SubagentRunRecord
from app.domain.models.subagent_worker import SubagentRunResult


class SubagentRunRepository(ABC):
    """subagent-run 持久化观测面。记录「一次 run 完成/失败」的事实，不含执行语义。"""

    @abstractmethod
    async def record(self, result: SubagentRunResult) -> None:
        """best-effort 幂等持久化一条 run 结果。

        - 幂等：对重复 ``child_session_id`` 无害（实现层用 INSERT ... ON CONFLICT
          DO NOTHING；见 spec §4 为何用 ``child_session_id`` 而非 ``source_ref``）。
        - **不返回 id**（调用侧 fire-and-observe）。
        - **不承诺 never-raise**（是异步 I/O）；调用侧负责 try/except swallow
          （spec §6 INV-C4.1-2）。
        """
        ...

    @abstractmethod
    async def list_by_parent_session(
        self, parent_session_id: str
    ) -> list[SubagentRunRecord]:
        """按 ``parent_session_id`` 返回记录，按 ``created_at ASC, id ASC`` 排序
        （同毫秒确定序）。每行从扁平列重建 ``SubagentRunResult`` 再包
        ``SubagentRunRecord``（重建规则见 spec §3.3）。
        """
        ...
