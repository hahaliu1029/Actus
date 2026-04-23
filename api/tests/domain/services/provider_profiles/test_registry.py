import logging
import pytest

from app.domain.services.provider_profiles import get_profile, infer_provider_from_base_url
from app.domain.services.provider_profiles._base import ErrorClass
from app.domain.services.provider_profiles._registry import GENERIC_FINGERPRINTS


def test_get_profile_returns_registered() -> None:
    p = get_profile("openai_official")
    assert p.provider_id == "openai_official"
    p2 = get_profile("generic_openai")
    assert p2.provider_id == "generic_openai"


def test_get_profile_unknown_raises_configerror() -> None:
    from app.application.errors.exceptions import ConfigError
    with pytest.raises(ConfigError, match="unknown provider"):
        get_profile("kimi_xxx_not_real")


def test_infer_openai_official() -> None:
    assert infer_provider_from_base_url("https://api.openai.com/v1",
                                        model_name="gpt-4o") == "openai_official"


def test_infer_unknown_returns_generic_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert infer_provider_from_base_url("https://weird.example.com/v1",
                                            model_name="unknown") == "generic_openai"
    assert any("did not match any heuristic" in r.message for r in caplog.records)


def test_generic_fingerprints_rate_limit() -> None:
    codes = [fp.error_class for fp in GENERIC_FINGERPRINTS
             if fp.status_code == 429]
    assert ErrorClass.TRANSIENT_RATE_LIMIT in codes


def test_generic_fingerprints_auth() -> None:
    codes = [fp.error_class for fp in GENERIC_FINGERPRINTS
             if fp.status_code == 401]
    assert ErrorClass.TRANSIENT_AUTH in codes


def test_infer_moonshot_maps_to_kimi_k2_not_k2_6() -> None:
    """T13/T17: moonshot.ai 启发式只落 kimi_k2；K2.6 必须显式"""
    assert infer_provider_from_base_url(
        "https://api.moonshot.ai/v1", model_name="kimi-k2",
    ) == "kimi_k2"
    assert infer_provider_from_base_url(
        "https://api.moonshot.ai/v1", model_name="kimi-k2.6",
    ) == "kimi_k2"   # 即使 model_name 含 "k2.6" 也不自动识别；必须显式


def test_infer_kimi_keyword() -> None:
    assert infer_provider_from_base_url(
        "https://kimi-api.example/v1", model_name="",
    ) == "kimi_k2"
