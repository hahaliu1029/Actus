"""C4.1a 域读模型 — subagent_runs 持久化行的读视图（spec §3.2）。

domain 约束：仅依赖 pydantic + stdlib + 同层 domain models。持久化视图 =
C4 结果（嵌入的 ``run``）+ 一行的 ``id`` 与 ``created_at``。ORM 层把 ``run``
的字段扁平化成列以支持 SQL 查询；repo 读回时从列重建 ``SubagentRunResult``
再包成本读模型（见 spec §3.3 / §4）。
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.domain.models.subagent_worker import SubagentRunResult


class SubagentRunRecord(BaseModel):
    """一条 subagent-run 持久化记录的读视图（frozen）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    created_at: datetime
    run: SubagentRunResult
