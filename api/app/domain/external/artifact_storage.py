"""C2 v1 ArtifactStoragePort (spec §10.5 / §13).

Thin domain Protocol the PatchApplier uses to pull worker-produced patch
content from blob storage. Bound at composition root to the live MinIO
adapter (``infrastructure/external/file_storage/minio_file_storage`` —
``MinioFileStorage.put_content_addressed_bytes`` / ``get_bytes``).

Why a separate Protocol instead of importing ``MinioFileStorage`` directly:

- The applier lives in ``application/`` and must not directly import an
  ``infrastructure/`` concrete adapter (Clean Architecture boundary).
- Tests inject ``AsyncMock`` matching this Protocol — no MinIO server
  required.
- A future S3 / GCS / local-fs swap is a one-line composition change.
"""
from __future__ import annotations

from typing import Optional, Protocol


class ArtifactStoragePort(Protocol):
    """Content-addressed blob store contract.

    ``put_content_addressed_bytes`` returns the wire-stable ``content_ref``
    embedded in PR-4's ``FilePatchEntry.content_ref`` field. The applier
    then resolves that ref via ``get_bytes`` to fetch the actual bytes
    for write.
    """

    async def put_content_addressed_bytes(
        self,
        *,
        prefix: str,
        content: bytes,
        filename: Optional[str] = None,
    ) -> str: ...

    async def get_bytes(self, ref: str) -> bytes: ...
