import pytest
from pydantic import ValidationError

from core.config import Settings


def _settings(**kwargs: object) -> Settings:
    return Settings(jwt_secret_key="unit-test-secret", **kwargs)


def test_default_public_endpoint_falls_back_to_internal_minio_settings() -> None:
    settings = _settings(
        _env_file=None,
        minio_endpoint="minio:9000",
        minio_secure=False,
        minio_public_endpoint=None,
        minio_public_secure=None,
    )

    assert settings.effective_minio_public_endpoint == "minio:9000"
    assert settings.effective_minio_public_secure is False


@pytest.mark.parametrize("public_endpoint", [None, "", "   "])
def test_unconfigured_public_endpoint_ignores_isolated_public_secure(
    public_endpoint: str | None,
) -> None:
    settings = _settings(
        _env_file=None,
        minio_endpoint="minio:9000",
        minio_secure=True,
        minio_public_endpoint=public_endpoint,
        minio_public_secure=False,
    )

    assert settings.minio_public_endpoint is None
    assert settings.effective_minio_public_endpoint == "minio:9000"
    assert settings.effective_minio_public_secure is True


def test_explicit_public_endpoint_uses_public_secure() -> None:
    settings = _settings(
        _env_file=None,
        minio_endpoint="minio:9000",
        minio_region="us-east-1",
        minio_secure=True,
        minio_public_endpoint="  localhost:9000  ",
        minio_public_secure=False,
    )

    assert settings.minio_public_endpoint == "localhost:9000"
    assert settings.effective_minio_public_endpoint == "localhost:9000"
    assert settings.effective_minio_public_secure is False


def test_public_endpoint_without_public_secure_falls_back_to_internal_secure() -> None:
    settings = _settings(
        _env_file=None,
        minio_region="us-east-1",
        minio_secure=False,
        minio_public_endpoint="localhost:9000",
        minio_public_secure=None,
    )

    assert settings.effective_minio_public_secure is False


@pytest.mark.parametrize("region", [None, "", "   "])
def test_public_endpoint_requires_non_blank_region(region: str | None) -> None:
    with pytest.raises(ValidationError, match="MINIO_REGION"):
        _settings(
            _env_file=None,
            minio_region=region,
            minio_public_endpoint="localhost:9000",
        )


def test_non_string_public_endpoint_is_not_coerced_to_string() -> None:
    with pytest.raises(ValidationError):
        _settings(
            _env_file=None,
            minio_region="us-east-1",
            minio_public_endpoint=9000,
        )


@pytest.mark.parametrize(
    "public_endpoint", [b" localhost:9000 ", b"   "]
)
def test_bytes_public_endpoint_is_not_decoded_to_string(
    public_endpoint: bytes,
) -> None:
    with pytest.raises(ValidationError):
        _settings(
            _env_file=None,
            minio_region="us-east-1",
            minio_public_endpoint=public_endpoint,
        )
