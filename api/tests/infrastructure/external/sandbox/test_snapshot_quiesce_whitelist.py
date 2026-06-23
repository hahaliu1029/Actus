"""S2 PR-4 Task 4.11 — quiesce RPC whitelist membership.

``kill_all_shell_sessions`` must be in ``SANDBOX_FORWARDED_METHODS`` so the
``SandboxHandleImpl.__getattr__`` whitelist proxy forwards the call from the
adapter through to ``DockerSandbox`` (§3.2). Without it the handle raises
``AttributeError`` at runtime even though the adapter / protocol / HTTP client
all exist.
"""
from app.infrastructure.external.sandbox.sandbox_handle import (
    SANDBOX_FORWARDED_METHODS,
)


def test_kill_all_shell_sessions_is_forwarded():
    assert "kill_all_shell_sessions" in SANDBOX_FORWARDED_METHODS
