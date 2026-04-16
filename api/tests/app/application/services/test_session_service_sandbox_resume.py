import asyncio
import io

import pytest

from app.application.services.session_service import SessionService
from app.domain.models.session import (
    SandboxBinding,
    SandboxBindingState,
    Session,
    SessionStatus,
)
from app.domain.models.tool_result import ToolResult


class FakeSessionRepo:
    def __init__(self, session: Session | None) -> None:
        self._session = session

    async def get_by_id(self, session_id: str):
        if not self._session:
            return None
        return self._session if self._session.id == session_id else None


class FakeUnitOfWork:
    def __init__(self, session: Session | None) -> None:
        self.session = FakeSessionRepo(session)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class FakeSandboxHandle:
    def __init__(self) -> None:
        self.download_calls: list[str] = []
        self.read_calls: list[str] = []

    async def download_file(self, filepath: str):
        self.download_calls.append(filepath)
        return io.BytesIO(b"%PDF-1.7 fake")

    async def read_file(self, filepath: str):
        self.read_calls.append(filepath)
        return ToolResult(
            success=True,
            data={"filepath": filepath, "content": "hello"},
        )


class FakeLifecycle:
    def __init__(self, handle: FakeSandboxHandle) -> None:
        self.handle = handle
        self.acquire_calls: list[str] = []
        self.resume_calls: list[str] = []

    async def acquire(self, session_id: str):
        from app.domain.errors.sandbox_lifecycle import SessionSuspendedError

        self.acquire_calls.append(session_id)
        raise SessionSuspendedError(session_id)

    async def resume(self, session_id: str):
        self.resume_calls.append(session_id)
        return self.handle


def make_uow_factory(session: Session | None):
    def factory() -> FakeUnitOfWork:
        return FakeUnitOfWork(session)

    return factory


def _make_suspended_session() -> Session:
    return Session(
        id="s1",
        title="demo",
        user_id="owner",
        status=SessionStatus.COMPLETED,
        sandbox_binding=SandboxBinding(
            id="sb-1",
            state=SandboxBindingState.SUSPENDED,
            generation=1,
        ),
    )


def test_download_file_resumes_suspended_sandbox() -> None:
    session = _make_suspended_session()
    handle = FakeSandboxHandle()
    lifecycle = FakeLifecycle(handle)
    service = SessionService(
        uow_factory=make_uow_factory(session),
        sandbox_lifecycle_service=lifecycle,
    )

    result = asyncio.run(
        service.download_file("s1", "/tmp/report.pdf", user_id="owner", is_admin=False)
    )

    assert result.read() == b"%PDF-1.7 fake"
    assert lifecycle.acquire_calls == ["s1"]
    assert lifecycle.resume_calls == ["s1"]
    assert handle.download_calls == ["/tmp/report.pdf"]


def test_read_file_resumes_suspended_sandbox() -> None:
    session = _make_suspended_session()
    handle = FakeSandboxHandle()
    lifecycle = FakeLifecycle(handle)
    service = SessionService(
        uow_factory=make_uow_factory(session),
        sandbox_lifecycle_service=lifecycle,
    )

    result = asyncio.run(
        service.read_file("s1", "/tmp/report.txt", user_id="owner", is_admin=False)
    )

    assert result.content == "hello"
    assert lifecycle.acquire_calls == ["s1"]
    assert lifecycle.resume_calls == ["s1"]
    assert handle.read_calls == ["/tmp/report.txt"]
