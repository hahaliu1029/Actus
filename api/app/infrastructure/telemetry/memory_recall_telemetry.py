"""B8 telemetry 通道 2：``actus.memory_recall`` logger → OTel metrics 桥。

通道 1（planner_react._emit_recall_telemetry）在专用 logger 上发一行
JSON，并把结构化 payload 挂在 ``record.recall_event``。本模块的
``logging.Handler`` 消费该 payload，更新两个低基数 instruments：

- counter ``memory_recall_total{mode,result}``
- histogram ``memory_recall_latency_ms{mode,result}``

**挂载目标 = root logger（实现期偏差 #5，controller 裁定）**：
``setup_logging()``（main.py import 期）把 ``actus.memory_recall`` 变成
``_RedactingPropagateOnlyLogger``（redaction.py Q2 self-heal），其
``addHandler`` 是文档化 no-op——named logger 永远不持有本地 handler，
record 只经 propagation 到达 root。因此 handler 挂 root，并在 ``emit``
里按 ``record.name == RECALL_LOGGER_NAME`` 门控，只消费 recall logger
的记录（root 上的其他记录全部忽略）。

挂载点 = FastAPI lifespan（``mount_memory_recall_metrics()``）——
``OtelMeter()`` 在 ``setup_observability`` 前是 no-op proxy，可无条件
构造（先例：service_dependencies.py CoordinatorMetrics）。任何构造失败
吞掉返回 None：metrics 是可选的，logger 通道照常工作。domain 保持
OTel-free：provider 只碰 stdlib logging。
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

RECALL_LOGGER_NAME = "actus.memory_recall"


class MemoryRecallMetricsHandler(logging.Handler):
    def __init__(self, meter: Any) -> None:
        super().__init__(level=logging.INFO)
        self._counter = meter.create_counter(
            "memory_recall_total",
            description="B8 query-time recall calls by mode/result",
        )
        self._latency = meter.create_histogram(
            "memory_recall_latency_ms",
            unit="ms",
            description="B8 query-time recall end-to-end latency",
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # 偏差 #5：handler 挂 root，会收到所有 logger 的记录——
            # 先按 record 身份门控，只消费 recall logger 的 recall_event
            if record.name != RECALL_LOGGER_NAME:
                return
            payload = getattr(record, "recall_event", None)
            if not isinstance(payload, dict):
                return
            attributes = {
                "mode": str(payload.get("mode")),
                "result": str(payload.get("result")),
            }
            self._counter.add(1, attributes=attributes)
            latency = payload.get("latency_ms")
            if isinstance(latency, (int, float)):
                self._latency.record(float(latency), attributes=attributes)
        except Exception:
            # metrics 永不破坏 logging 或召回路径
            pass


def mount_memory_recall_metrics(meter: Any | None = None) -> logging.Handler | None:
    """幂等挂载（root logger）。返回 handler；构造失败返回 None
    （metrics 不可用，logger 通道不受影响）。

    偏差 #5：挂 root 而非 ``actus.memory_recall``——named logger 是
    ``_RedactingPropagateOnlyLogger``，其 ``addHandler`` 为 no-op；
    record 只经 propagation 到 root。幂等性 = 扫描 root.handlers 中
    既有的 ``MemoryRecallMetricsHandler`` 实例。
    """
    root = logging.getLogger()
    for existing in root.handlers:
        if isinstance(existing, MemoryRecallMetricsHandler):
            return existing
    try:
        if meter is None:
            from app.infrastructure.observability import OtelMeter

            meter = OtelMeter()
        handler = MemoryRecallMetricsHandler(meter)
    except Exception:
        logger.debug("memory recall metrics mount skipped", exc_info=True)
        return None
    root.addHandler(handler)
    return handler
