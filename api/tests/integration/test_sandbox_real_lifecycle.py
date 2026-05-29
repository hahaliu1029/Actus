"""[PR-9b-C C5] sandbox_real per-test Docker container lifecycle.

CI-validated only: needs a Docker daemon (and either ``sandbox_address`` or
local image build). Marked ``sandbox`` so it is excluded from the default
``-m 'not ... and not sandbox ...'`` collection (pytest.ini:13); run with
``pytest -m sandbox``.
"""
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.anyio,
    pytest.mark.coordinator_recovery,
    pytest.mark.sandbox,
]


async def test_sandbox_real_created(sandbox_real):
    """The fixture must yield a live DockerSandbox instance."""
    assert sandbox_real is not None
