"""INV-D1-6：registry 异常 → enforce fail-closed / shadow fail-open（不需要 DB）。"""
import pytest

from app.infrastructure.external.governance.db_extension_admission import DbExtensionAdmissionPort


class _BoomFactory:
    def __call__(self):
        raise RuntimeError("db down")


@pytest.mark.anyio
async def test_check_many_enforce_fail_closed():
    port = DbExtensionAdmissionPort(_BoomFactory(), mode="enforce")
    out = await port.check_many("mcp", ["s1", "s2"])
    assert set(out) == {"s1", "s2"}
    for d in out.values():
        assert d.admitted is False and d.reason == "registry_unavailable"


@pytest.mark.anyio
async def test_check_many_shadow_fail_open():
    port = DbExtensionAdmissionPort(_BoomFactory(), mode="shadow")
    out = await port.check_many("mcp", ["s1"])
    assert out["s1"].admitted is True and out["s1"].reason == "registry_unavailable"


@pytest.mark.anyio
async def test_verify_observation_exception_same_semantics():
    from app.domain.external.extension_admission import Observation
    obs = Observation(category="artifact", payload="deadbeef", schema_version=1)
    enforce = DbExtensionAdmissionPort(_BoomFactory(), mode="enforce")
    shadow = DbExtensionAdmissionPort(_BoomFactory(), mode="shadow")
    assert (await enforce.verify_observation("skill", "k1", obs)).admitted is False
    assert (await shadow.verify_observation("skill", "k1", obs)).admitted is True
