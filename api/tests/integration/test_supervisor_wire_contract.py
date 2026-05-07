"""B3-core PR-0: 6 SSE wire contract anchors (xfail).

Spec v3 §3.2-§3.3 + §8.1 (groups C-Wire-* and C-Redis-*).

These anchors flip from xfail → xpass as PR-1 / PR-4 / PR-2 land their respective
contracts. Until then, every test here is expected to fail (or error on missing
imports / endpoints).

PR boundary mapping per spec v3 §8.2 (round-3 patched + round-4 P1#1 fix):
- C-Wire-1: PR-1 (ExecutionStateChangedEvent type discriminator)
- C-Wire-2: PR-4 (SSE heartbeat connection-level, not in Stream — round-4 audit
            P1#1 corrected mapping; SSE handler integration is PR-4 territory)
- C-Wire-3: PR-4 (auto-degrade emit-to-backlog; round-3 spec patch moved this from PR-1 to PR-4)
- C-Wire-4: PR-1 (since_seq query param precedence)
- C-Redis-1: PR-1 (Stream MAXLEN ~ 2000)
- C-Redis-2: PR-2 (supervisor:hot Hash 300s TTL)
"""

from __future__ import annotations

import json
import uuid

import pytest

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


# -- C-Wire-1: ExecutionStateChangedEvent uses `type` discriminator ---------------------
@pytest.mark.xfail(strict=False, reason="PR-1: new event types not yet shipped")
def test_C_Wire_1_execution_state_changed_event_schema():
    from app.domain.models.event import ExecutionStateChangedEvent, ExecutionStatePayload
    evt = ExecutionStateChangedEvent(payload=ExecutionStatePayload(
        execution_mode="foreground", execution_phase="running", retry_budget_remaining=3,
    ))
    raw = evt.model_dump_json()
    parsed = json.loads(raw)
    assert parsed["type"] == "execution_state_changed"  # NOT "event_type"


# -- C-Wire-2: SSE heartbeat is connection-level, NOT in Stream -------------------------
# Round-4 audit P1 fix: prior body was hollow — `final_len - initial_len == 0`
# would trivially XPASS in a real DB/Redis env (no writer = no growth).
# Also, spec v3 §8.2 round-3 patched table maps C-Wire-2 to PR-4 (not PR-1):
# the heartbeat-not-in-Stream invariant is enforced by PR-4's auto-degrade /
# SSE handler integration, not by PR-1's wire schema.
# Converted to explicit placeholder (matches C-Restart-NEW / C-MultiTab-1 pattern).
@pytest.mark.xfail(strict=False, reason="PR-4: SSE heartbeat-not-in-Stream contract not yet shipped")
async def test_C_Wire_2_heartbeat_not_in_stream(asgi_client, redis_client, sample_session):
    pytest.fail(
        "placeholder — flip when PR-4 ships SSE handler with EventSourceResponse "
        "ping=5s + verifies no ping events leak into task:output:{sid} Stream "
        "(spec v3 §3.3 SSE heartbeat clause). Implementation must consume SSE "
        "for ≥6s (covers 5s ping interval) then assert "
        "`final_len - initial_len < 2`."
    )


# -- C-Wire-3: Auto-degrade emits to durable backlog only (not to disconnected client) ---
@pytest.mark.xfail(strict=False, reason="PR-4: auto-degrade detached task not yet shipped")
async def test_C_Wire_3_intentional_eof_auto_degrade(
    asgi_client, redis_client, sample_session, sample_user_token,
):
    # Round-5 audit P2 fix: prior body scanned Redis without opening/disconnecting
    # SSE — `xrevrange` on an empty stream returns [] and `any(...)` over [] is
    # False, so the assert "fails" only because nothing was emitted (not because
    # the contract was tested).  Converted to explicit placeholder pattern
    # matching C-Wire-2 / C-Notif-Reuse.
    pytest.fail(
        "placeholder — flip when PR-4 ships auto-degrade detached task. "
        "Implementation must:\n"
        "  (1) open SSE chat for FG session\n"
        "  (2) disconnect mid-stream (close client side)\n"
        "  (3) await server-side `_do_auto_degrade` to fire (asyncio.create_task detach)\n"
        "  (4) assert `ExecutionStateChangedEvent(execution_mode='background', "
        "background_reason='auto_degrade')` lands in `task:output:{sid}` Stream\n"
        "  (5) assert disconnected client did NOT receive the event before close\n"
        "Per spec v3 §6.6 + decision 6.7."
    )


# -- C-Wire-4: since_seq query param wins over since=<stream_id> ------------------------
@pytest.mark.xfail(strict=False, reason="PR-1: since_seq param not yet shipped")
async def test_C_Wire_4_since_seq_precedence(asgi_client, sample_session, sample_user_token):
    # Round-5 audit P2 fix: prior body's `all(s > 5 for s in seqs)` is vacuously
    # True on an empty list — endpoint exists in PR-0 but returns no events for
    # a freshly-created session, so seqs=[] and test XPASSes.  Converted to
    # explicit placeholder so PR-1 author writes the real assertion.
    pytest.fail(
        "placeholder — flip when PR-1 ships `since_seq` query param + precedence. "
        "Implementation must:\n"
        "  (1) seed the session with ≥2 events (seq=1, seq=2, seq=3, ..., seq=N)\n"
        "  (2) GET /api/sessions/{sid}/events?since_seq=5&since=stream-id-foo\n"
        "  (3) assert response 200 + `last_seq` field present in body\n"
        "  (4) assert returned events list is non-empty AND all seqs > 5\n"
        "  (5) confirm `since_seq=5` precedence: server logged warning that\n"
        "      `since=stream-id-foo` was ignored when both given\n"
        "Per spec v3 §3.3 + plan PR-1 Task 7."
    )


# -- C-Redis-1: task:output Stream has MAXLEN ~2000 trim --------------------------------
# Round-3 audit P1#4 fix: test through producer path (`agent_service._emit_event`),
# not bare `redis_client.xadd(maxlen=2000)`. The bare-XADD form was a false positive —
# it tested Redis command parameters, not the contract that the production producer
# applies MAXLEN. PR-1 ships `_emit_event` that internally uses MAXLEN.
@pytest.mark.xfail(strict=False, reason="PR-1: agent_service._emit_event with MAXLEN not yet shipped")
async def test_C_Redis_1_stream_has_maxlen_2000(redis_client, agent_service_with_redis, sample_session):
    sid = sample_session.id
    key = f"task:output:{sid}"
    # PR-1 will add agent_service._emit_event — until then, this import fails (xfail)
    from app.domain.models.event import MessageEvent  # any concrete event subtype

    # Push 3000 events through producer; producer must apply MAXLEN ~ 2000
    # Round-2 audit P3 fix: MessageEvent field is `message=` not `content=`
    # (see api/app/domain/models/event.py:102).
    for i in range(3000):
        evt = MessageEvent(message=f"msg {i}")
        await agent_service_with_redis._emit_event(sid, evt)

    length = await redis_client.xlen(key)
    assert length <= 2050, f"stream length {length} exceeds MAXLEN+epsilon (producer didn't apply MAXLEN?)"
    assert length >= 1900, f"stream length {length} suspiciously low — MAXLEN trim too aggressive?"


# -- C-Redis-2: supervisor:hot Hash has 300s TTL refreshed on activity ------------------
# Round-3 audit P1#4 fix: test through `IdleWatchdog.touch_activity` (PR-2 production
# refresh path), not bare `hset` + `expire(300)`. The bare form was a false positive —
# it tested Redis command parameters, not the contract that the watchdog refreshes TTL.
# Round-2 audit P2-A fix: source watchdog from `app.state.idle_watchdog` (per spec v3
# §6.1 round-3 P2-A — lifespan stores singleton on app.state), NOT a private attribute
# on agent_service.
# Round-3 audit P2-NEW-1 caveat: ``app.state.idle_watchdog`` is populated by the
# FastAPI lifespan startup hook, NOT at app construction.  Until PR-2 ships
# lifespan integration AND the test fixture invokes lifespan (via asgi-lifespan
# `LifespanManager` or manual startup call), `app.state.idle_watchdog` will be
# absent — `AttributeError` is a stable xfail signal pointing at PR-2 work.
@pytest.mark.xfail(strict=False, reason="PR-2: IdleWatchdog.touch_activity + lifespan wiring not yet shipped")
async def test_C_Redis_2_hot_hash_300s_ttl(redis_client, app, sample_session):
    from app.domain.services.idle_watchdog import IdleWatchdog  # PR-2

    sid = sample_session.id
    key = f"supervisor:hot:{sid}"

    # PR-2 IdleWatchdog must EXPIRE hot Hash to 300s on touch_activity.
    # Watchdog is wired as `app.state.idle_watchdog` per spec v3 §6.1.
    # Without lifespan startup having run, `app.state.idle_watchdog` raises
    # AttributeError — that's an acceptable xfail until PR-2.
    watchdog: IdleWatchdog = app.state.idle_watchdog
    await watchdog.touch_activity(session_id=sid)

    # Hash exists + TTL is 300s (or close — refresh just happened)
    ttl = await redis_client.ttl(key)
    assert 290 <= ttl <= 300, f"hot Hash TTL = {ttl}s, expected ~300s after touch_activity"
