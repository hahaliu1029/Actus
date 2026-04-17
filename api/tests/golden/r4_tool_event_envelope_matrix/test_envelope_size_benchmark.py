"""R-6 envelope 体积上限 benchmark.

硬门槛 (2026-04-16 Gate 1 实测重定基线):

26 fixture 实测结构开销区间: 364 B (legacy_no_result) — 752 B (skill_denied_risk_enforce)
平均: ~613 B; 100 × allow_success 累计: ~53 KB

硬上限 (绝对, 不留"百分比余量"—剩余 headroom 由 worst-case 实测决定):
- 单事件结构开销 (不含 .data payload / result_blocks): ≤ 800 bytes
  当前 worst-case ~752 B, 剩余 ~48 B headroom — 任何 projector 扩字段或 Denied
  分支加 message 长度都可能撞线, 届时要么扩上限要么裁字段.
- 100 events 累计 (结构开销): ≤ 100 KB
  当前 ~53 KB, 剩余 ~47 KB headroom.

对于 data payload 大的 mcp / a2a 调用或 passthrough 含 base64 blocks, 体积
天然不受此阈值约束; 单测逻辑里扣掉 .data / .result_blocks 字节数再断言.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from .test_r4_envelope_golden import FIXTURE_NAMES, _load_fixture
from app.application.services.tool_event_envelope_v1 import project_tool_event_to_envelope_v1
from app.domain.models.event import ToolEvent
from app.interfaces.schemas.event import ToolEventEnvelopeV1


STRUCTURE_OVERHEAD_PER_EVENT_LIMIT = 800
STRUCTURE_OVERHEAD_100_EVENTS_LIMIT = 100_000


def _structural_size(envelope: ToolEventEnvelopeV1) -> int:
    """计算 envelope 的结构开销: 总体积减去 .data / .result_blocks payload."""
    wire_bytes = len(envelope.model_dump_json(by_alias=True).encode("utf-8"))
    fr = envelope.function_result
    payload_bytes = 0
    if fr is not None:
        if fr.data is not None:
            payload_bytes += len(json.dumps(fr.data, ensure_ascii=False).encode("utf-8"))
        if fr.result_blocks:
            payload_bytes += len(json.dumps(fr.result_blocks, ensure_ascii=False).encode("utf-8"))
    return wire_bytes - payload_bytes


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_single_event_structural_overhead(fixture_name: str) -> None:
    """每个 fixture 单事件结构开销 (扣 payload) ≤ 800 bytes."""
    fixture = _load_fixture(fixture_name)
    evt = ToolEvent.model_validate(fixture)
    envelope = project_tool_event_to_envelope_v1(evt)
    overhead = _structural_size(envelope)

    assert overhead <= STRUCTURE_OVERHEAD_PER_EVENT_LIMIT, (
        f"{fixture_name}: structure overhead {overhead} bytes exceeds "
        f"{STRUCTURE_OVERHEAD_PER_EVENT_LIMIT} bytes limit"
    )


def test_100_events_cumulative_structural_overhead() -> None:
    """100 次 AllowSuccess 结构开销累计 ≤ 100 KB."""
    total = 0
    for _ in range(100):
        fixture = _load_fixture("native_allow_success")
        evt = ToolEvent.model_validate(fixture)
        envelope = project_tool_event_to_envelope_v1(evt)
        total += _structural_size(envelope)

    assert total <= STRUCTURE_OVERHEAD_100_EVENTS_LIMIT, (
        f"100 events累计 {total} bytes exceeds {STRUCTURE_OVERHEAD_100_EVENTS_LIMIT} bytes limit"
    )


def test_baseline_calibration_sanity() -> None:
    """Sanity: 确认当前实现产出 ~400-800 bytes range (不超 800)."""
    fixture = _load_fixture("native_allow_success")
    evt = ToolEvent.model_validate(fixture)
    envelope = project_tool_event_to_envelope_v1(evt)
    overhead = _structural_size(envelope)
    assert 400 <= overhead <= 800, (
        f"baseline calibration drift: overhead={overhead} out of 400-800 expected range"
    )


def test_worst_case_fixture_under_limit() -> None:
    """Worst-case 断言: 26 个 fixture 中结构开销最大者仍必须 <= 800 B.

    2026-04-16 实测 worst-case: skill_denied_risk_enforce ~752 B
    (离 800 B 仅剩 ~48 B headroom — 如果未来 projector 扩字段或 Denied
    分支加 message 长度, 这条 sanity 会在累积前就先炸).
    """
    max_overhead = 0
    worst_name = ""
    for name in FIXTURE_NAMES:
        fixture = _load_fixture(name)
        evt = ToolEvent.model_validate(fixture)
        envelope = project_tool_event_to_envelope_v1(evt)
        size = _structural_size(envelope)
        if size > max_overhead:
            max_overhead = size
            worst_name = name

    assert max_overhead <= STRUCTURE_OVERHEAD_PER_EVENT_LIMIT, (
        f"worst-case fixture {worst_name!r} overhead {max_overhead} B "
        f"exceeds {STRUCTURE_OVERHEAD_PER_EVENT_LIMIT} B limit"
    )
