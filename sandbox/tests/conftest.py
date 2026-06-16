"""Sandbox unit-test harness (C2-full S1).

Runs as its OWN pytest project: ``cd sandbox && uv run pytest``. It CANNOT be
collected under the api venv — both ``api/app`` and ``sandbox/app`` are
top-level ``app`` packages and co-importing them in one interpreter collides.
``[tool.pytest.ini_options] pythonpath = ["."]`` (sandbox/pyproject.toml) puts
``sandbox/`` on sys.path so ``import app.services.file`` resolves to the
sandbox app. Pure ``os``/``tempfile`` behavior — no Docker, no HTTP server.
"""
import io

import pytest
from fastapi import UploadFile


@pytest.fixture
def make_upload():
    """Wrap raw bytes in a minimal FastAPI UploadFile for upload_file tests.

    ``FileService.upload_file`` only ever touches ``file.file.read(size)``, so
    a BytesIO-backed UploadFile is a faithful stand-in for a multipart upload.
    """

    def _make(content: bytes, filename: str = "upload.bin") -> UploadFile:
        return UploadFile(file=io.BytesIO(content), filename=filename)

    return _make
