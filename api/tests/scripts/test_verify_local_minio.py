import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import verify_local_minio as verify


API_ROOT = Path(__file__).resolve().parents[2]


def _settings(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "minio_endpoint": "localhost:19000",
        "effective_minio_public_endpoint": "localhost:19000",
        "minio_secure": False,
        "effective_minio_public_secure": False,
        "minio_region": "us-east-1",
        "minio_bucket_name": verify.BUCKET_NAME,
        "minio_access_key": "access-key-must-not-leak",
        "minio_secret_key": "secret-key-must-not-leak",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("minio_endpoint", "minio:9000"),
        ("effective_minio_public_endpoint", "example.com:9000"),
        ("minio_secure", True),
        ("effective_minio_public_secure", True),
        ("minio_region", "eu-west-1"),
        ("minio_bucket_name", "production-bucket"),
    ],
)
def test_validate_acceptance_settings_rejects_each_mismatch(
    field: str,
    bad_value: object,
) -> None:
    with pytest.raises(RuntimeError, match=field):
        verify._validate_acceptance_settings(_settings(**{field: bad_value}))


def test_validate_acceptance_settings_accepts_exact_contract() -> None:
    verify._validate_acceptance_settings(_settings())


def test_module_help_entrypoint_from_api_root_does_not_construct_store() -> None:
    env = os.environ.copy()
    env.update(
        {
            "ENV": "test",
            "JWT_SECRET_KEY": "unit-test-secret",
            "MINIO_ENDPOINT": "production.invalid:9000",
            "MINIO_PUBLIC_ENDPOINT": "production.invalid:9000",
            "MINIO_REGION": "us-east-1",
            "MINIO_SECURE": "true",
            "MINIO_PUBLIC_SECURE": "true",
        }
    )

    completed = subprocess.run(
        [sys.executable, "-m", "scripts.verify_local_minio", "--help"],
        cwd=API_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    help_text = " ".join(completed.stdout.split())
    assert completed.returncode == 0, completed.stderr
    assert "{write,verify}" in help_text
    assert "write: store" in help_text
    assert "verify: verify" in help_text
    assert "one-shot cleanup" in help_text
    assert completed.stderr == ""


def test_validation_lists_mismatches_without_credentials() -> None:
    settings = _settings(
        minio_endpoint="minio:9000",
        effective_minio_public_endpoint="public.example.com:443",
        minio_secure=True,
        effective_minio_public_secure=True,
        minio_region="eu-west-1",
        minio_bucket_name="production-bucket",
    )

    with pytest.raises(RuntimeError) as exc_info:
        verify._validate_acceptance_settings(settings)

    message = str(exc_info.value)
    for field in (
        "minio_endpoint",
        "effective_minio_public_endpoint",
        "minio_secure",
        "effective_minio_public_secure",
        "minio_region",
        "minio_bucket_name",
    ):
        assert field in message
    assert settings.minio_access_key not in message
    assert settings.minio_secret_key not in message


@pytest.mark.anyio
async def test_main_rejects_bad_settings_before_store_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed = 0

    class ForbiddenStore:
        def __init__(self) -> None:
            nonlocal constructed
            constructed += 1
            raise AssertionError("MinioStore must not be constructed")

    monkeypatch.setattr(
        verify,
        "get_settings",
        lambda: _settings(minio_endpoint="minio:9000"),
        raising=False,
    )
    monkeypatch.setattr(verify, "MinioStore", ForbiddenStore)

    with pytest.raises(RuntimeError, match="minio_endpoint"):
        await verify.main("write")

    assert constructed == 0
