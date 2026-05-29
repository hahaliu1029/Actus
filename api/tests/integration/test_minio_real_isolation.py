"""[PR-9b-C C4] minio_real per-test bucket isolation.

CI-validated only: needs a reachable MinIO endpoint (env vars per
core/config.py:74-79). The compose MinIO is on the internal network and not
host-published locally, so these run in CI / against a live endpoint.
"""
import pytest

# ``sandbox`` marks these as excluded from the pg+redis CI recovery pass
# (``coordinator_recovery and not sandbox``); they need a reachable MinIO
# endpoint, which that pass does NOT provision — see PR-9b-C CI-infra note
# (provisioning MinIO/Docker in ci.yml is a separate decision).
pytestmark = [
    pytest.mark.integration,
    pytest.mark.anyio,
    pytest.mark.coordinator_recovery,
    pytest.mark.sandbox,
]


async def test_minio_real_bucket_name_prefixed(minio_real):
    """Per-test bucket name must use the isolation prefix."""
    assert minio_real.bucket_name.startswith("actus-test-")
    assert minio_real.client is not None


async def test_minio_real_bucket_exists_after_setup(minio_real):
    """The bucket the fixture advertises must actually exist on setup."""
    import asyncio

    assert minio_real.bucket_name.startswith("actus-test-")
    exists = await asyncio.to_thread(
        minio_real.client.bucket_exists, minio_real.bucket_name
    )
    assert exists is True
