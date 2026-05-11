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

import inspect
import json
import uuid

import pytest

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


# -- C-Wire-1: ExecutionStateChangedEvent uses `type` discriminator ---------------------
# B3-core PR-1: anchor flipped (xfail → pass) — ExecutionStateChangedEvent shipped in event.py.
def test_C_Wire_1_execution_state_changed_event_schema():
    from app.domain.models.event import ExecutionStateChangedEvent, ExecutionStatePayload
    evt = ExecutionStateChangedEvent(payload=ExecutionStatePayload(
        execution_mode="foreground", execution_phase="running", retry_budget_remaining=3,
    ))
    raw = evt.model_dump_json()
    parsed = json.loads(raw)
    assert parsed["type"] == "execution_state_changed"  # NOT "event_type"


# -- C-Wire-2: SSE heartbeat is connection-level, NOT in Stream -------------------------
async def test_C_Wire_2_heartbeat_not_in_stream(
    redis_client, agent_service_with_redis, sample_session
):
    from app.domain.models.event import MessageEvent
    from app.interfaces.endpoints import session_routes

    sid = sample_session.id
    await agent_service_with_redis._emit_event(
        sid, MessageEvent(role="assistant", message="real producer event")
    )

    rows = await redis_client.xrange(f"task:output:{sid}")
    assert rows, "sanity: producer did not write a real app event"
    persisted = [json.loads(fields["data"]) for _, fields in rows]
    assert {event["type"] for event in persisted} == {"message"}

    source = inspect.getsource(session_routes.chat)
    assert "EventSourceResponse(event_generator()" in source
    assert "ping_message_factory" not in source


# -- C-Wire-3: Auto-degrade emits to durable backlog only (not to disconnected client) ---
async def test_C_Wire_3_intentional_eof_auto_degrade(
    agent_service_with_redis, redis_client, sample_session,
):
    from app.interfaces.endpoints.session_routes import _do_auto_degrade

    sid = sample_session.id
    await agent_service_with_redis._supervisor.script_load_all()

    await _do_auto_degrade(
        sid,
        sample_session.user_id,
        agent_service_with_redis,
        agent_service_with_redis._supervisor,
    )

    rows = await redis_client.xrange(f"task:output:{sid}")
    events = [json.loads(fields["data"]) for _, fields in rows]
    state_events = [event for event in events if event["type"] == "execution_state_changed"]
    assert state_events
    payload = state_events[-1]["payload"]
    assert payload["execution_mode"] == "background"
    assert payload["background_reason"] == "auto_degrade"


# -- C-Wire-4: since_seq query param wins over since=<stream_id> ------------------------
# B3-core PR-1 ships the wire shape (since_seq query param + last_seq response field +
# agent_service.get_events_since since_seq precedence). The integration fixture wires
# only PR-1's DB UoW + Redis recovery path; PR-2's supervisor/repo/Lua surface remains
# out of scope.
async def test_C_Wire_4_since_seq_precedence(
    asgi_client, agent_service_with_redis, sample_session, sample_user_token
):
    """C-Wire-4 (per spec v3 §3.3): seq cursor wins over event_id cursor.

    Seeds session with 8 events stamped via _emit_event (seq 1..8), then
    requests with both `since=<bogus>` and `since_seq=5`. Endpoint must
    return only events with seq > 5 (3 events: seq 6, 7, 8) and include
    `last_seq` in the response body.

    PR-1 has shipped:
      - agent_service._emit_event (T5) — stamps INCR session:seq:{sid}
      - /sessions/{id}/events ?since_seq=N (T8) — endpoint param + response.last_seq
      - get_events_since(since_seq=...) (T7) — service-layer precedence + dict shape
    The fixture intentionally avoids PR-2's ExecutionSupervisor surface; this
    anchor only needs the producer/recovery/route path.
    """
    from app.domain.models.event import MessageEvent

    sid = sample_session.id

    # Seed 8 events through producer; producer stamps event.seq via INCR.
    for i in range(8):
        evt = MessageEvent(role="assistant", message=f"msg-{i}")
        await agent_service_with_redis._emit_event(sid, evt)

    # since=<bogus stream-id> + since_seq=5 → since_seq wins; expect seq 6, 7, 8.
    resp = await asgi_client.get(
        f"/api/sessions/{sid}/events?since_seq=5&since=stream-id-foo",
        headers={"Authorization": f"Bearer {sample_user_token}"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()["data"]
    # last_seq field present + reflects max seq (8) seen
    assert "last_seq" in body
    assert body["last_seq"] == 8

    # Returned events: SSE wire wrapper has shape `{event, data}` with `seq` inside `data`.
    seqs = [
        evt.get("data", {}).get("seq")
        for evt in body["events"]
        if isinstance(evt.get("data"), dict) and evt["data"].get("seq") is not None
    ]
    assert seqs, "since_seq returned no events — filter applied incorrectly?"
    assert all(s > 5 for s in seqs), f"seq filter leaked: {seqs}"
    assert sorted(seqs) == [6, 7, 8], f"unexpected seq tail: {sorted(seqs)}"


# -- C-Redis-1: task:output Stream has MAXLEN ~2000 trim --------------------------------
# Round-3 audit P1#4 fix: test through producer path (`agent_service._emit_event`),
# not bare `redis_client.xadd(maxlen=2000)`. The bare-XADD form was a false positive —
# it tested Redis command parameters, not the contract that the production producer
# applies MAXLEN.
#
# B3-core PR-1 ships:
#   - RedisStreamMessageQueue.put with MAXLEN + first-write EXPIRE (T4)
#   - agent_service._emit_event using output_stream.put (T5, falls back to
#     RedisStreamMessageQueue.put via the in-process registry miss path)
# Unit-level coverage for MAXLEN is in
# api/tests/app/infrastructure/external/message_queue/test_redis_stream_message_queue_maxlen.py
# (5 mock-based tests verifying the call shape).
#
# The integration fixture wires only PR-1's producer path. PR-2's supervisor/repo
# fixture expansion is not required for this Stream MAXLEN contract.
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
