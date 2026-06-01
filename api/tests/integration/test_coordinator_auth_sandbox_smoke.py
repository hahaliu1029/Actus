import pytest

pytestmark = [pytest.mark.integration, pytest.mark.anyio, pytest.mark.coordinator_recovery]


async def test_create_session_requires_auth(coord_async_client):
    """401 without a token (proves the harness can't accidentally bypass auth)."""
    resp = await coord_async_client.post("/api/sessions")
    assert resp.status_code in (401, 403)


async def test_create_and_chat_authenticate_with_real_jwt(
    inject_routing_fake_llm, coord_async_client, coord_jwt_headers,
):
    """[INV-F6.1] create AND chat accept the real JWT for a committed user.
    Uses the routing fake (coord_async_client) so the chat doesn't hit a real
    LLM; a minimal non-parallel plan keeps it off the coordinator path. We assert
    only the auth/connect status (200), not agent success."""
    inject_routing_fake_llm.setup_responses(
        planner_response={"steps": [{"id": "s", "description": "noop"}]},
        child_responses={},
    )
    resp = await coord_async_client.post("/api/sessions", headers=coord_jwt_headers)
    assert resp.status_code == 200
    session_id = resp.json()["data"]["session_id"]
    async with coord_async_client.stream(
        "POST", f"/api/sessions/{session_id}/chat",
        json={"message": "hi"}, headers=coord_jwt_headers,
    ) as r:
        assert r.status_code == 200


async def test_parent_sandbox_adapter_byte_roundtrip(sandbox_real):
    """[INV-F6.2] the adapter byte API round-trips at a G2b subdir-relative path.
    Standalone adapter over sandbox_real — this proves the byte API, not session
    integration (that's the F5 E2E via bind_session_sandbox_adapter)."""
    from app.infrastructure.external.sandbox.parent_sandbox_adapter import (
        ParentSandboxAdapter,
    )
    adapter = ParentSandboxAdapter(sandbox_real)
    await adapter.atomic_write_file("workspace/smoke.txt", b"hello")
    got = await adapter.read_file("workspace/smoke.txt")
    assert got == b"hello"
