"""S2 PR-1 — /api/file/snapshot-workspace route wiring.

CI-only: sandbox venv. Exercises the route handler directly against a real
tmp workspace (no live HTTP server / Docker needed).
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_route_returns_scan_envelope(tmp_path, monkeypatch):
    from app.core import config as cfg
    from app.interfaces.endpoints.file import snapshot_workspace
    from app.interfaces.schemas.file import SnapshotWorkspaceRequest
    from app.services.file import FileService

    ws = tmp_path / "home"
    ws.mkdir()
    (ws / "workspace").mkdir()
    (ws / "workspace" / "a.py").write_bytes(b"hi")
    monkeypatch.setenv("WORKSPACE_ROOT", str(ws))
    monkeypatch.setenv("SERVICE_INSTALL_DIR", "/sandbox")
    cfg.get_settings.cache_clear()
    try:
        req = SnapshotWorkspaceRequest(
            root=str(ws), max_paths=1000, max_files=1000,
            max_total_bytes=1_000_000, max_seconds=10.0,
        )
        resp = await snapshot_workspace(req, file_service=FileService())
        assert resp.data.truncated is False
        assert "workspace/a.py" in resp.data.entries
    finally:
        cfg.get_settings.cache_clear()
