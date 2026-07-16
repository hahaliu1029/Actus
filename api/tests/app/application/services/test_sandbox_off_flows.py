"""SPM PR-4 Task 32 — service-level ``off`` flow-group tests (spec §9 / INV-SPM-3).

Companion to ``test_sandbox_on_demand_flows.py``: where that module characterizes
the ``on_demand`` provisioner flows, this one characterizes the two user-visible
``off`` flows that must keep working AFTER an ``always → off`` deployment
migration — history files produced during the ``always`` era (carrying MinIO
keys) stay **listable** and **downloadable via MinIO** without touching the
sandbox lifecycle at all (INV-SPM-3 zero-touch).

* list  → ``SessionService.get_session_files`` (pure DB read of ``session.files``)
* MinIO → ``FileService.download_file`` (``file_storage.download_file`` by id) —
  the sandbox-free download route, as opposed to ``SessionService.download_file``
  which routes through ``_acquire_sandbox`` and 409s under off (INV-SPM-7).

Style mirrors ``test_sandbox_on_demand_flows.py``: module-level
``pytestmark = pytest.mark.anyio`` + a local ``anyio_backend`` fixture (the repo
ships pytest-anyio, NOT pytest-asyncio), thin async fakes, a single ``_build_env``
factory. The off deployment constant is modeled by monkeypatching the process
settings singleton to ``sandbox_provision_mode='off'`` (``off`` is a first-class
ALLOWED value since Task 32; the direct attribute set bypasses the field
validator regardless of mode).
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import BinaryIO, Optional

import pytest

from app.application.errors.exceptions import SandboxDisabledError
from app.application.services.file_service import FileService
from app.application.services.session_service import SessionService
from app.domain.models.file import File
from app.domain.models.session import Session, SessionStatus

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ── flow constants ────────────────────────────────────────────────────────────

_OWNER = "owner-1"
_SESSION_ID = "off-sess"
_FILE_ID = "f-hist-1"
_MINIO_KEY = "minio/owner-1/report.pdf"
_FILE_BYTES = b"%PDF-1.7 historical always-era bytes"


def _history_file() -> File:
    """An ``always``-era attachment record: NON-empty MinIO ``key`` + a
    ``/home/ubuntu`` sandbox ``filepath`` (anti-false-green — a blank key/path
    could mask an absent/skipped MinIO record)."""
    return File(
        id=_FILE_ID,
        filename="report.pdf",
        filepath="/home/ubuntu/report.pdf",
        key=_MINIO_KEY,
        extension="pdf",
        mime_type="application/pdf",
        size=len(_FILE_BYTES),
        user_id=_OWNER,
    )


# ── thin async fakes (session + file repos over one UoW; MinIO storage) ───────


class _SpyLifecycle:
    """Sandbox lifecycle spy: every state-touching method increments a counter so
    ``total_calls`` is a single INV-SPM-3 zero-touch signal. Neither off flow may
    call any of these; acquire/bind additionally hard-raise so a regression that
    reached them surfaces loudly rather than silently bumping the counter."""

    def __init__(self) -> None:
        self.acquire_calls = 0
        self.bind_calls = 0
        self.suspend_calls = 0
        self.resume_calls = 0
        self.destroy_calls = 0

    @property
    def total_calls(self) -> int:
        return (
            self.acquire_calls
            + self.bind_calls
            + self.suspend_calls
            + self.resume_calls
            + self.destroy_calls
        )

    async def acquire(self, session_id: str):
        self.acquire_calls += 1
        raise AssertionError("off flow must not acquire the sandbox")

    async def bind_new(self, session_id: str, *, user_id: str | None = None):
        self.bind_calls += 1
        raise AssertionError("off flow must not bind a sandbox")

    async def suspend(self, session_id: str) -> None:
        self.suspend_calls += 1

    async def resume(self, session_id: str):
        self.resume_calls += 1

    async def destroy(self, session_id: str, reason=None) -> None:
        self.destroy_calls += 1


class _FakeSessionRepo:
    def __init__(self, session: Session) -> None:
        self._session = session

    async def get_by_id(self, session_id: str) -> Optional[Session]:
        if self._session and self._session.id == session_id:
            return self._session.model_copy(deep=True)
        return None


class _FakeFileRepo:
    def __init__(self, files: dict[str, File]) -> None:
        self._files = files

    async def get_by_id(self, file_id: str) -> Optional[File]:
        f = self._files.get(file_id)
        return f.model_copy(deep=True) if f is not None else None


class _FakeUoW:
    def __init__(self, session: Session, files: dict[str, File]) -> None:
        self.session = _FakeSessionRepo(session)
        self.file = _FakeFileRepo(files)

    async def __aenter__(self) -> "_FakeUoW":
        return self

    async def __aexit__(self, *a) -> None:
        return None


class _FakeMinioStorage:
    """``FileStorage`` download surface: streams the seeded bytes back paired with
    the ``File`` (mirrors ``MinioFileStorage.download_file`` → ``(BinaryIO, File)``).
    Records each ``file_id`` so the MinIO route is provably exercised."""

    def __init__(self, files: dict[str, File], payload: bytes) -> None:
        self._files = files
        self._payload = payload
        self.download_calls: list[str] = []

    async def download_file(self, file_id: str) -> tuple[BinaryIO, File]:
        self.download_calls.append(file_id)
        return io.BytesIO(self._payload), self._files[file_id].model_copy(deep=True)


@dataclass
class _OffEnv:
    session_service: SessionService
    file_service: FileService
    lifecycle: _SpyLifecycle
    storage: _FakeMinioStorage


def _build_env() -> _OffEnv:
    hist = _history_file()
    session = Session(
        id=_SESSION_ID,
        user_id=_OWNER,
        status=SessionStatus.COMPLETED,
        files=[hist],
    )
    files = {hist.id: hist}
    uow = _FakeUoW(session, files)
    lifecycle = _SpyLifecycle()
    storage = _FakeMinioStorage(files, _FILE_BYTES)
    session_service = SessionService(
        uow_factory=lambda: uow, sandbox_lifecycle_service=lifecycle
    )
    file_service = FileService(uow_factory=lambda: uow, file_storage=storage)
    return _OffEnv(
        session_service=session_service,
        file_service=file_service,
        lifecycle=lifecycle,
        storage=storage,
    )


@pytest.fixture
def off_env(monkeypatch) -> _OffEnv:
    """A fully-wired off deployment: the process settings singleton pinned to
    ``off`` + a session whose ``always``-era files carry MinIO keys."""
    from core.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "sandbox_provision_mode", "off", raising=False)
    return _build_env()


# ── flows (spec §9, off branch) ───────────────────────────────────────────────


class TestOffFlows:
    async def test_always_to_off_history_files_list_ok(self, off_env: _OffEnv) -> None:
        """``always → off`` migration: history files (with MinIO keys) stay listable
        via the pure-DB ``get_session_files`` service method — zero sandbox touch."""
        files = await off_env.session_service.get_session_files(_SESSION_ID, _OWNER)

        assert [f.filename for f in files] == ["report.pdf"]
        # the listed record still carries its always-era MinIO key (not stripped).
        assert files[0].key == _MINIO_KEY
        assert off_env.lifecycle.total_calls == 0  # INV-SPM-3 zero-touch

    async def test_always_to_off_minio_download_ok(self, off_env: _OffEnv) -> None:
        """``always → off`` migration: a history file downloads OK via the MinIO
        route (``FileService.download_file``) while the sandbox route
        (``SessionService.download_file``) 409s — all with zero sandbox touch."""
        data, file = await off_env.file_service.download_file(_FILE_ID, _OWNER)

        assert data.read() == _FILE_BYTES
        assert file.key == _MINIO_KEY
        assert off_env.storage.download_calls == [_FILE_ID]  # MinIO route exercised

        # the sandbox download route is the 409 alternative under off (INV-SPM-7),
        # proving the MinIO route above is genuinely the sandbox-free path.
        with pytest.raises(SandboxDisabledError):
            await off_env.session_service.download_file(
                _SESSION_ID, "/home/ubuntu/report.pdf", _OWNER
            )
        # even the 409'd sandbox route pre-checks off BEFORE any lifecycle call.
        assert off_env.lifecycle.total_calls == 0  # INV-SPM-3 zero-touch
