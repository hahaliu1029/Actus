import asyncio
import io

import pytest
from app.application.errors.exceptions import ForbiddenError
from app.application.services.file_service import FileService
from app.domain.models.file import File


class FakeFileRepo:
    def __init__(self, file: File | None) -> None:
        self._file = file
        self.saved_file: File | None = None

    async def get_by_id(self, file_id: str) -> File | None:
        if not self._file:
            return None
        return self._file if self._file.id == file_id else None

    async def save(self, file: File) -> None:
        self.saved_file = file
        self._file = file


class FakeUnitOfWork:
    def __init__(self, file: File | None) -> None:
        self.file = FakeFileRepo(file)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class FakeFileStorage:
    async def upload_file(self, upload_file):
        return File(id="uploaded", key="k", user_id=None)

    async def download_file(self, file_id: str):
        return io.BytesIO(b"content"), File(id=file_id, key="k", user_id="owner")

    async def delete_file(self, file_id: str) -> None:
        return None


def make_uow_factory(file: File | None):
    def factory() -> FakeUnitOfWork:
        return FakeUnitOfWork(file)

    return factory


class DummyUploadFile:
    filename = "demo.txt"
    content_type = "text/plain"


def test_get_file_info_rejects_other_user() -> None:
    service = FileService(
        uow_factory=make_uow_factory(File(id="f1", key="k", user_id="owner")),
        file_storage=FakeFileStorage(),
    )

    with pytest.raises(ForbiddenError):
        asyncio.run(service.get_file_info("f1", user_id="visitor", is_admin=False))


def test_get_file_info_allows_admin_cross_user() -> None:
    service = FileService(
        uow_factory=make_uow_factory(File(id="f1", key="k", user_id="owner")),
        file_storage=FakeFileStorage(),
    )

    result = asyncio.run(service.get_file_info("f1", user_id="admin", is_admin=True))
    assert result.id == "f1"


def test_download_rejects_orphan_file_for_normal_user() -> None:
    service = FileService(
        uow_factory=make_uow_factory(File(id="f1", key="k", user_id=None)),
        file_storage=FakeFileStorage(),
    )

    with pytest.raises(ForbiddenError):
        asyncio.run(service.download_file("f1", user_id="visitor", is_admin=False))


def test_upload_file_persists_user_id() -> None:
    uow = FakeUnitOfWork(file=None)
    service = FileService(
        uow_factory=lambda: uow,
        file_storage=FakeFileStorage(),
    )

    result = asyncio.run(
        service.upload_file(DummyUploadFile(), user_id="owner")
    )

    assert result.user_id == "owner"
    assert uow.file.saved_file is not None
    assert uow.file.saved_file.user_id == "owner"
