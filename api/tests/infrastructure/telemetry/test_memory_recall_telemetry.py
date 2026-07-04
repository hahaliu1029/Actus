"""B8 PR-4: telemetry 通道 2——logger → OTel metrics 桥（P-8）。

实现期偏差 #5（controller 裁定）：``actus.memory_recall`` logger 是
``_RedactingPropagateOnlyLogger``（redaction.py Q2 self-heal），其
``addHandler`` 是文档化 no-op——record 只能经 propagation 到达 root。
因此 handler 挂 **root** logger，``emit`` 按 ``record.name`` 门控只消费
recall logger 的记录。P-8 通道 2 实质契约不变：recall_event → OTel
metrics、``mount_memory_recall_metrics(meter=None)`` 幂等、handler
fail-safe。
"""
import logging
from unittest.mock import MagicMock

import pytest

from app.infrastructure.telemetry.memory_recall_telemetry import (
    RECALL_LOGGER_NAME,
    MemoryRecallMetricsHandler,
    mount_memory_recall_metrics,
)


def _fake_meter():
    meter = MagicMock()
    meter.create_counter = MagicMock(return_value=MagicMock())
    meter.create_histogram = MagicMock(return_value=MagicMock())
    return meter


@pytest.fixture(autouse=True)
def _clean_logger():
    target = logging.getLogger(RECALL_LOGGER_NAME)
    root = logging.getLogger()
    saved_level = target.level
    # 偏差 #5：handler 挂 root（named logger 的 addHandler 是
    # _RedactingPropagateOnlyLogger no-op）——setup/teardown 都从 root 剥离
    for h in list(root.handlers):
        if isinstance(h, MemoryRecallMetricsHandler):
            root.removeHandler(h)
    # pytest 下 root 有效级别通常是 WARNING——不显式设 INFO 的话 .info()
    # record 会在 isEnabledFor 处被吞，handler 永不触发（生产环境 root=INFO
    # 无此问题，spec §5.8 R5#2）
    target.setLevel(logging.INFO)
    yield
    for h in list(root.handlers):
        if isinstance(h, MemoryRecallMetricsHandler):
            root.removeHandler(h)
    target.setLevel(saved_level)


def _emit(payload: dict):
    logging.getLogger(RECALL_LOGGER_NAME).info(
        "recall event", extra={"recall_event": payload},
    )


class TestHandler:
    def test_counter_and_histogram_updated(self):
        meter = _fake_meter()
        handler = mount_memory_recall_metrics(meter)
        # 偏差 #5：handler 真挂在 root 上（named logger 的 addHandler 为 no-op）
        assert handler in logging.getLogger().handlers
        _emit({"mode": "shadow", "result": "items", "latency_ms": 42.5})
        counter = meter.create_counter.return_value
        histogram = meter.create_histogram.return_value
        counter.add.assert_called_once_with(
            1, attributes={"mode": "shadow", "result": "items"},
        )
        histogram.record.assert_called_once_with(
            42.5, attributes={"mode": "shadow", "result": "items"},
        )

    def test_instrument_names(self):
        meter = _fake_meter()
        mount_memory_recall_metrics(meter)
        assert meter.create_counter.call_args.args[0] == "memory_recall_total"
        assert meter.create_histogram.call_args.args[0] == "memory_recall_latency_ms"

    def test_record_without_recall_event_ignored(self):
        meter = _fake_meter()
        mount_memory_recall_metrics(meter)
        logging.getLogger(RECALL_LOGGER_NAME).info("plain line, no extra")
        meter.create_counter.return_value.add.assert_not_called()

    def test_non_recall_logger_record_ignored(self):
        """偏差 #5 新增：root 上的 handler 会收到所有 logger 的记录——
        emit 必须按 record.name 门控；其他 logger 的记录（即使带
        recall_event attr）不产生 metrics。"""
        meter = _fake_meter()
        mount_memory_recall_metrics(meter)
        # 子 logger 继承 fixture 在父 logger 上设的 INFO 有效级别，
        # record 确实会经 propagation 打到 root handler——只被 name 门控拦下
        other = logging.getLogger(RECALL_LOGGER_NAME + ".child")
        other.info(
            "not a recall event",
            extra={"recall_event": {"mode": "on", "result": "items", "latency_ms": 1.0}},
        )
        meter.create_counter.return_value.add.assert_not_called()
        meter.create_histogram.return_value.record.assert_not_called()

    def test_missing_latency_still_counts(self):
        meter = _fake_meter()
        mount_memory_recall_metrics(meter)
        _emit({"mode": "on", "result": "timeout", "latency_ms": None})
        meter.create_counter.return_value.add.assert_called_once()
        meter.create_histogram.return_value.record.assert_not_called()

    def test_instrument_error_swallowed(self):
        meter = _fake_meter()
        meter.create_counter.return_value.add = MagicMock(side_effect=RuntimeError("otel down"))
        mount_memory_recall_metrics(meter)
        _emit({"mode": "on", "result": "items", "latency_ms": 1.0})  # 不抛


class TestMount:
    def test_mount_is_idempotent(self):
        meter = _fake_meter()
        h1 = mount_memory_recall_metrics(meter)
        h2 = mount_memory_recall_metrics(_fake_meter())
        assert h1 is h2
        root = logging.getLogger()
        assert sum(isinstance(h, MemoryRecallMetricsHandler) for h in root.handlers) == 1

    def test_mount_failure_returns_none(self):
        meter = MagicMock()
        meter.create_counter = MagicMock(side_effect=RuntimeError("no meter"))
        assert mount_memory_recall_metrics(meter) is None
        root = logging.getLogger()
        assert not any(isinstance(h, MemoryRecallMetricsHandler) for h in root.handlers)
