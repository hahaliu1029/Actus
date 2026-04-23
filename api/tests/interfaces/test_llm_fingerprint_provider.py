from app.domain.models.app_config import LLMConfig
from app.interfaces.service_dependencies import _llm_fingerprint


def test_fingerprint_includes_provider() -> None:
    """T16: provider 不同 → fingerprint 不同（不会 cache 串）"""
    c1 = LLMConfig(base_url="https://x.example/v1", api_key="k", model_name="m",
                   provider="kimi_k2")
    c2 = LLMConfig(base_url="https://x.example/v1", api_key="k", model_name="m",
                   provider="openai_official")
    assert _llm_fingerprint(c1, supports_pdf_input=False) != \
           _llm_fingerprint(c2, supports_pdf_input=False)


def test_fingerprint_equal_when_provider_same() -> None:
    c1 = LLMConfig(base_url="https://x.example/v1", api_key="k", model_name="m",
                   provider="kimi_k2")
    c2 = LLMConfig(base_url="https://x.example/v1", api_key="k", model_name="m",
                   provider="kimi_k2")
    assert _llm_fingerprint(c1, supports_pdf_input=False) == \
           _llm_fingerprint(c2, supports_pdf_input=False)


def test_fingerprint_none_provider_matches_empty() -> None:
    """provider=None 视作空串"""
    c1 = LLMConfig(base_url="https://x.example/v1", api_key="k", model_name="m",
                   provider=None)
    c2 = LLMConfig(base_url="https://x.example/v1", api_key="k", model_name="m",
                   provider=None)
    assert _llm_fingerprint(c1, supports_pdf_input=False) == \
           _llm_fingerprint(c2, supports_pdf_input=False)


def test_fingerprint_strips_whitespace_to_match_build_llm() -> None:
    """Normalization parity with _build_llm: " openai_official " and
    "openai_official" resolve to the same profile, so they must also
    produce the same fingerprint — otherwise the adapter cache splits
    on semantically equivalent configs."""
    c1 = LLMConfig(base_url="https://x.example/v1", api_key="k", model_name="m",
                   provider="openai_official")
    c2 = LLMConfig(base_url="https://x.example/v1", api_key="k", model_name="m",
                   provider=" openai_official ")
    assert _llm_fingerprint(c1, supports_pdf_input=False) == \
           _llm_fingerprint(c2, supports_pdf_input=False)


def test_fingerprint_empty_and_whitespace_only_match_none() -> None:
    """Empty / whitespace-only / None all normalize to the same fingerprint slot."""
    base_kwargs = dict(base_url="https://x.example/v1", api_key="k", model_name="m")
    c_none = LLMConfig(**base_kwargs, provider=None)
    c_empty = LLMConfig(**base_kwargs, provider="")
    c_ws = LLMConfig(**base_kwargs, provider="   ")
    fp_none = _llm_fingerprint(c_none, supports_pdf_input=False)
    assert fp_none == _llm_fingerprint(c_empty, supports_pdf_input=False)
    assert fp_none == _llm_fingerprint(c_ws, supports_pdf_input=False)
