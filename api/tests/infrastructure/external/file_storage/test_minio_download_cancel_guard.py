"""SPM r24/R24-CLASS1 — ``MinioFileStorage.download_file`` read-UoW cancel guard.

The read-UoW ``__aexit__`` commit swallows ``CancelledError`` WITHOUT ``uncancel()``
(``db_uow.py:71``). Without the guard, a cancel landing on that commit would let
``download_file`` proceed to the external MinIO ``get_object`` (wasted I/O) and, if
the download then raised an ordinary error, the provisioner would misclassify the
outcome as ``failed`` rather than ``cancelled``. The guard — added after the
read-UoW exits and before the MinIO call — honors the swallowed cancel so the
MinIO call never happens.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.file import File
from app.infrastructure.external.file_storage.minio_file_storage import MinioFileStorage

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _SwallowCommitUoW:
    """Read-UoW whose ``__aexit__`` commit hangs, then swallows an injected
    cancel WITHOUT ``uncancel()`` — exactly the ``db_uow.py:71`` behavior."""

    def __init__(self) -> None:
        self.file = SimpleNamespace(
            get_by_id=AsyncMock(
                return_value=File(id="f1", key="k1", filename="f1.pdf")
            )
        )
        self.reached = asyncio.Event()
        self.hang = asyncio.Event()

    async def __aenter__(self) -> "_SwallowCommitUoW":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        self.reached.set()
        try:
            await self.hang.wait()
        except asyncio.CancelledError:
            pass  # swallow like db_uow.py:71 — NO uncancel()
        return False


async def test_download_cancel_on_read_uow_commit_skips_minio() -> None:
    uow = _SwallowCommitUoW()
    minio_store = MagicMock()
    minio_store.download_fileobj = AsyncMock()
    storage = MinioFileStorage(
        bucket="b", minio_store=minio_store, uow_factory=lambda: uow
    )

    task = asyncio.create_task(storage.download_file("f1"))
    await asyncio.wait_for(uow.reached.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    minio_store.download_fileobj.assert_not_called()
