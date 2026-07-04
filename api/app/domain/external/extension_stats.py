"""B9 扩展统计端口（domain Protocol，P-9 钉子）。

domain 协议——Redis 实现在 infrastructure/external/runtime_stats（R16#3 分层冻结）。
本模块**只声明形状**：`ExtensionStatsData` 聚合 DTO + `ExtensionStatsRecorder` /
`ExtensionStatsReader` 两个结构化 Protocol。**无 Redis / httpx / FastAPI 概念**——
纯 domain 端口，供 application 层（Task 6 RuntimeExtensionService）与 infrastructure
层（Task 19 RedisExtensionStats）以结构化子类型对接，不产生跨层运行期依赖。

- `ExtensionStatsRecorder.record` 同步、绝不 await（生产路径由工具执行热路径调用，
  内部实现自行入队/落盘，禁止阻塞 event loop）。
- `ExtensionStatsReader.read_many` async（GET 聚合器批量读快照）；`delete_key`
  fire-and-forget（reconcile 剔除后入队 DEL，绝不 await）。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class ExtensionStatsData:
    """单个扩展的调用统计聚合值（不可变 DTO）。"""

    call_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_active_at: datetime | None = None
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None


class ExtensionStatsRecorder(Protocol):
    """记录端口（工具执行热路径消费）。同步、绝不 await。"""

    def record(self, tool_name: str, success: bool, latency_ms: float) -> None: ...


class ExtensionStatsReader(Protocol):
    """读取端口（GET 聚合器 / reconcile 消费）。"""

    async def read_many(
        self, keys: list[tuple[str, str]]
    ) -> dict[tuple[str, str], ExtensionStatsData]: ...

    def delete_key(self, kind: str, extension_id: str) -> None: ...
