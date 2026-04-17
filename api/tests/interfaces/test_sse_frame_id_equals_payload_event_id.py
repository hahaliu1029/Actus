"""R4 / N2 double-ID contract: SSE frame `id:` field == payload `event_id`.

F6 fix: N2 未落地前，此测试 xfail 作为 executable guard. N2 实现 SSE frame `id:`
field 时，开发者负责去掉 xfail marker 让此测试转绿.

规则 (docs/adr/CS3-tool-event-envelope-v1.md):
- SSE frame 里的 `id:` 字段必须 == 同 frame payload 里的 `event_id` 字段
- 防止断点续传时双 ID 通道错位 (一个事件被前端处理两次, 或漏事件)
"""
from __future__ import annotations

import pytest


@pytest.mark.xfail(
    reason="N2 not yet landed: SSE frame id field not yet emitted by session_routes.py. "
           "When N2 lands this test must pass — remove xfail marker.",
    strict=True,
)
def test_sse_frame_id_equals_payload_event_id() -> None:
    """当 N2 发 SSE frame `id:` 后, 该 id 必须与 payload 里 event_id 相等."""
    # TODO (N2 PR): 通过 TestClient 访问 /sessions/{id}/events SSE endpoint,
    # 解析 frame 的 `id:` 字段, 断言与 frame data 里的 payload.event_id 相等.
    # 示例伪代码:
    #   with client.stream("GET", f"/sessions/{session_id}/events") as response:
    #       for frame in parse_sse_frames(response.iter_text()):
    #           assert frame.id == json.loads(frame.data)["event_id"]
    raise NotImplementedError(
        "N2 implementation pending. Remove xfail marker when SSE frame id is emitted."
    )
