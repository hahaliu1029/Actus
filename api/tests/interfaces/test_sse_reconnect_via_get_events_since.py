"""N2 §5.3 + §7 回归锁: full-body parse + GET /events?since= 控制流 smoke.

**测试到底在测什么 (诚实版)**:
  1. POST /api/sessions/{id}/chat 拉回整份 SSE body
  2. 解析出所有 frame, 取前 3 帧作为 "已见事件" 的 stand-in, 记录第 3 帧 event_id
  3. 用 last_seen_id 调 GET /api/sessions/{id}/events?since=<last_seen_id>
  4. 断言: 返回 events 恰好是 last_seen_id 之后那些 (gap)

**这不是真实的 "mid-stream abort" 端到端**: 当前环境 httpx 0.28.x 的 ASGITransport
会把 ASGI StreamingResponse 全量 buffer 后再一次性交给 aiter_text(), sleep(0) 救
不了. 真实 "reader 断线 writer 继续" 的 backlog 语义归 Task 8 (AgentService 层
mock writer + spy recovery) 专门验证.

本测试保留是因为仍然锁住"接口层控制流"回归 — POST body + GET since params +
response shape — 这部分未被更底层的 Task 8 覆盖. 非 N2 红灯, expected PASS.

Spec: docs/superpowers/specs/2026-04-17-n2-sse-transport-repair-design.md §5.3 + §7
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator

import httpx
import pytest

from app.domain.models.event import BaseEvent, MessageEvent
from app.domain.models.user import User, UserRole, UserStatus
from app.interfaces.dependencies import rate_limit_chat, rate_limit_read
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.endpoints import session_routes
from app.interfaces.service_dependencies import get_agent_service
from app.main import app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _fake_user() -> User:
    return User(
        id="test-user",
        username="tester",
        role=UserRole.USER,
        status=UserStatus.ACTIVE,
    )


async def _noop_rate_limit() -> None:
    return None


class _FakeLease:
    def start_heartbeat(self) -> None:
        pass

    async def release(self) -> None:
        pass


def _make_events() -> list[BaseEvent]:
    """预置 5 帧 events, id 单调."""
    events = []
    for i in range(5):
        m = MessageEvent(role="assistant", message=f"msg-{i}")
        m.id = f"1000-{i}"
        events.append(m)
    return events


class _StatefulAgentService:
    """chat() 产 5 帧 (模拟完整 agent 运行); get_events_since() 按 payload id 切片.

    在本测试场景中, events 列表作为 "已持久化事件" 的 stand-in; 客户端 SSE abort
    后发 GET since=, 返回该列表在 since 之后的部分.
    """

    def __init__(self) -> None:
        self._events = _make_events()

    async def chat(self, **kwargs: Any) -> AsyncGenerator[BaseEvent, None]:
        import asyncio as _asyncio
        for e in self._events:
            yield e
            # Force yield point between frames so httpx consumer can abort mid-stream.
            await _asyncio.sleep(0)

    async def get_events_since(
        self,
        *,
        session_id: str,
        since_event_id: str | None,
        user_id: str,
        is_admin: bool = False,
        since_seq: int | None = None,  # B3-core PR-1 §3.3 — additive kwarg
    ) -> dict[str, Any]:
        if since_event_id is None:
            missed = list(self._events)
        else:
            found_idx = None
            for i, e in enumerate(self._events):
                if e.id == since_event_id:
                    found_idx = i
                    break
            missed = (
                self._events[found_idx + 1:]
                if found_idx is not None
                else list(self._events)
            )
        # B3-core PR-1 §3.3 — derive last_seq from missed events' .seq, or fall back.
        seqs_seen = [
            int(getattr(e, "seq", None))
            for e in missed
            if getattr(e, "seq", None) is not None
        ]
        last_seq = max(seqs_seen) if seqs_seen else (since_seq or 0)
        return {
            "events": missed,
            "session_status": "running",
            "has_more": False,
            "last_seq": last_seq,
            "supervisor_snapshot": None,  # PR-3c/PR-4 populates
        }


def _parse_sse_chunk(buffer: str) -> tuple[list[dict[str, str]], str]:
    """从 running buffer 切出完整 frame, 返回 (frames, leftover_buffer).

    跨 chunk 边界安全 (保留最后一个未完整 block 作为 leftover). CRLF-safe:
    normalize CRLF 到 LF 再切, 避免 frame boundary 因换行格式解析失败.
    """
    normalized = buffer.replace("\r\n", "\n")
    parts = normalized.split("\n\n")
    leftover = parts[-1]  # 最后一段可能是半帧
    frames: list[dict[str, str]] = []
    for block in parts[:-1]:
        frame: dict[str, str] = {}
        for line in block.split("\n"):
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            frame[key.strip()] = value.lstrip(" ").rstrip("\r")
        if frame.get("data"):
            frames.append(frame)
    return frames, leftover


async def test_truncated_read_then_get_events_since_returns_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """读 (buffered) body → 取前 3 帧 → GET since=last_seen → 断言 gap 正确.

    **Note**: 因 httpx ASGITransport full-body buffering, 第一个 aiter_text chunk
    会含 5 帧全部内容; 测试在 "计数达 3" 时 break, 这是控制流 abort 模拟, 不是真实
    TCP-level abort. 见文件 module docstring.
    """
    service = _StatefulAgentService()

    async def _fake_acquire_connection_limit(**kwargs: Any) -> _FakeLease:
        return _FakeLease()

    monkeypatch.setattr(
        session_routes,
        "acquire_connection_limit",
        _fake_acquire_connection_limit,
    )

    app.dependency_overrides[get_current_user] = _fake_user
    app.dependency_overrides[get_agent_service] = lambda: service
    app.dependency_overrides[rate_limit_chat] = _noop_rate_limit
    app.dependency_overrides[rate_limit_read] = _noop_rate_limit

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # 1. 起 SSE
            last_seen_id: str | None = None
            frames_read = 0
            buffer = ""
            async with client.stream(
                "POST",
                "/api/sessions/test-session/chat",
                json={"message": "hi"},
            ) as response:
                assert response.status_code == 200
                async for chunk in response.aiter_text():
                    buffer += chunk
                    frames, buffer = _parse_sse_chunk(buffer)
                    for frame in frames:
                        payload = json.loads(frame["data"])
                        eid = payload.get("event_id")
                        if eid is not None:
                            last_seen_id = eid
                            frames_read += 1
                            if frames_read >= 3:
                                break
                    if frames_read >= 3:
                        break
                # `async with ... stream` 退出 — 控制流层面等价于 abort.
                # 注: 因 ASGITransport 全量 buffer, 底层 TCP 并未被打断.

            assert frames_read >= 3, (
                f"expected to count at least 3 frames in buffered body, got {frames_read}"
            )
            assert last_seen_id == "1000-2", (
                f"expected last_seen_id='1000-2', got {last_seen_id!r}"
            )

            # 2. 模拟 Web recoverSession: GET since=last_seen_id
            resp = await client.get(
                "/api/sessions/test-session/events",
                params={"since": last_seen_id},
            )
            assert resp.status_code == 200
            body = resp.json()
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_agent_service, None)
        app.dependency_overrides.pop(rate_limit_chat, None)
        app.dependency_overrides.pop(rate_limit_read, None)

    returned_ids = [e["data"]["event_id"] for e in body["data"]["events"]]
    assert returned_ids == ["1000-3", "1000-4"], (
        f"gap mismatch: got {returned_ids}, expected ['1000-3', '1000-4']"
    )
