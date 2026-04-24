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


def test_p1_profiles_registered_on_package_import() -> None:
    """T-P1-R1 part: package-level import triggers side-effect registration
    for all 6 new P1 profiles (A7 P1 Spec §7.1).

    这个测试文件 top-level 已经 `from app.domain.services.provider_profiles
    import get_profile, infer_provider_from_base_url`；pytest 加载 test module
    时就完成了 `provider_profiles/__init__.py` 的首次 import 与 side-effect
    registration，所以只要 __init__.py 里真的有 6 条新 profile import，
    get_profile() 就能直接解析。**不做 sys.modules reload** —— reload 会让
    "旧 module 上的旧 callable" 与 "reload 后的新 module" 并存，测试形状反而
    更脆弱（codex v3 Round 7 P2-2）。
    """
    for pid in ("dashscope_qwen", "dashscope_qwen_vl",
                "anthropic_compat", "gemini_compat",
                "minimax", "glm"):
        assert get_profile(pid).provider_id == pid


def test_registry_has_12_entries() -> None:
    """T-P1-R1: _REGISTRY grows from 6 (P0) to 12 (P0 + P1)."""
    from app.domain.services.provider_profiles._registry import _REGISTRY
    assert set(_REGISTRY.keys()) == {
        # P0
        "generic_openai", "openai_official",
        "kimi_k2", "kimi_k2_6",
        "deepseek_reasoner", "deepseek_chat",
        # P1
        "dashscope_qwen", "dashscope_qwen_vl",
        "anthropic_compat", "gemini_compat",
        "minimax", "glm",
    }


# ---------- T-P1-R2: Model-filtered routing (Spec §8.3 pure allowlist v4+5) ----------


@pytest.mark.parametrize("model_name,expected", [
    # DashScope text flagship allowlist HIT (hybrid default-off, tool-capable)
    ("qwen-plus",              "dashscope_qwen"),
    ("qwen-plus-latest",       "dashscope_qwen"),
    ("qwen-turbo",             "dashscope_qwen"),
    ("qwen-flash",             "dashscope_qwen"),
    ("qwen-flash-latest",      "dashscope_qwen"),
    ("qwen3-max",              "dashscope_qwen"),
    ("qwen3-max-preview",      "dashscope_qwen"),
    # VL allowlist HIT (Qwen3-VL: hybrid + tool-capable)
    ("qwen3-vl-plus",          "dashscope_qwen_vl"),
    ("qwen3-vl-flash",         "dashscope_qwen_vl"),
    # NOT in allowlist → generic_openai (Round 4 Fact #1-3 + Round 3 Fact #1-5)
    ("qwen-max",               "generic_openai"),   # non-thinking only
    ("qwen-max-longcontext",   "generic_openai"),
    ("qwen-max-latest",        "generic_openai"),
    ("qwen-vl-max",            "generic_openai"),   # Qwen2.5-VL no-thinking no-tools
    ("qwen-vl-plus",           "generic_openai"),
    ("qwen3.5-plus",           "generic_openai"),   # hybrid default-on
    ("qwen3.5-flash",          "generic_openai"),
    ("qwen-omni-turbo",        "generic_openai"),   # no thinking toggle
    ("qwen-omni-turbo-latest", "generic_openai"),
    ("qwen3-omni-flash",       "generic_openai"),   # hybrid thinking
    ("qwen3.5-omni-plus",      "generic_openai"),
    ("qwen3.5-omni-flash",     "generic_openai"),
    ("qwq-32b",                "generic_openai"),   # always-on, no toggle
    ("qwq-plus",               "generic_openai"),
    ("qwen3-235b-a22b-thinking-2507", "generic_openai"),
    ("qwen3-30b-a3b-thinking-2507",   "generic_openai"),
])
def test_dashscope_model_routing(model_name: str, expected: str) -> None:
    """T-P1-R2a: DashScope pure allowlist heuristic (25 cases)."""
    assert infer_provider_from_base_url(
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name=model_name,
    ) == expected


@pytest.mark.parametrize("model_name,expected", [
    ("claude-sonnet-4-6",        "anthropic_compat"),
    ("claude-sonnet-4-6-latest", "anthropic_compat"),
    ("claude-haiku-4-5",         "anthropic_compat"),
    # NOT in allowlist
    ("claude-opus-4-7",         "generic_openai"),  # no manual thinking
    ("claude-opus-4-7-preview", "generic_openai"),
    ("claude-sonnet-4-0",       "generic_openai"),  # legacy
    ("claude-3-5-sonnet",       "generic_openai"),  # legacy
])
def test_anthropic_model_routing(model_name: str, expected: str) -> None:
    """T-P1-R2b: Anthropic allowlist (7 cases)."""
    assert infer_provider_from_base_url(
        "https://api.anthropic.com/v1/",
        model_name=model_name,
    ) == expected


@pytest.mark.parametrize("model_name,expected", [
    ("gemini-2.5-flash",        "gemini_compat"),
    ("gemini-2.5-flash-8b",     "gemini_compat"),
    ("gemini-2.5-flash-latest", "gemini_compat"),
    # NOT in allowlist
    ("gemini-2.5-pro",       "generic_openai"),  # thinking always-on
    ("gemini-2.5-pro-exp",   "generic_openai"),
    ("gemini-3-pro-preview", "generic_openai"),  # Thinking MEDIUM bug
    ("gemini-3-flash",       "generic_openai"),
])
def test_gemini_model_routing(model_name: str, expected: str) -> None:
    """T-P1-R2c: Gemini allowlist (7 cases)."""
    assert infer_provider_from_base_url(
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        model_name=model_name,
    ) == expected


def test_minimax_base_url_match_regardless_of_model() -> None:
    """MiniMax uses base_url-only match; no model-name filter."""
    for bu in (
        "https://api.minimax.io/v1",
        "https://api.minimax.chat/v1",
        "https://api.minimaxi.com/v1",
    ):
        assert infer_provider_from_base_url(bu, model_name="MiniMax-M2") == "minimax"
        assert infer_provider_from_base_url(bu, model_name="") == "minimax"


def test_glm_base_url_match_regardless_of_model() -> None:
    """GLM uses base_url-only match; no model-name filter."""
    for bu in (
        "https://open.bigmodel.cn/api/paas/v4",
        "https://api.zhipu.example/v1",  # keyword match
    ):
        assert infer_provider_from_base_url(bu, model_name="glm-4.6") == "glm"
        assert infer_provider_from_base_url(bu, model_name="glm-5v-turbo") == "glm"


def test_allowlist_miss_routes_to_generic_openai_without_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """T-P1-R2d: heuristic 命中 base_url 但 model_name 不在 allowlist →
    直接返回 'generic_openai' 且 get_profile(result) 成功解析 — 无 WARN 日志
    (Spec §8.3)."""
    import logging
    caplog.set_level(logging.WARNING, logger="app.domain.services.provider_profiles._registry")

    miss_cases = [
        # DashScope hybrid default-on (NOT in allowlist)
        ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen3.5-plus"),
        # Anthropic Opus 4.7 (no manual thinking, NOT in allowlist)
        ("https://api.anthropic.com/v1/", "claude-opus-4-7"),
        # Gemini 3 preview (Thinking MEDIUM bug, NOT in allowlist)
        ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-3-pro-preview"),
    ]
    for base_url, model_name in miss_cases:
        result = infer_provider_from_base_url(base_url, model_name=model_name)
        assert result == "generic_openai", (base_url, model_name)
        # 真正调用 get_profile() 以 lock 端到端可解析（pre-A7 P1 的 silent-fallback
        # 路径会抛 ConfigError，这里必须成功）
        assert get_profile(result).provider_id == "generic_openai"

    # 关键断言：不应有 "not registered" / "falling back to generic_openai" WARN
    for record in caplog.records:
        assert "not registered" not in record.message
        assert "falling back to generic_openai" not in record.message


def test_llm_config_provider_docstring_lists_all_registered_ids() -> None:
    """C4 strict: `LLMConfig.provider` attribute docstring must enumerate the
    registered provider_ids exactly — no missing new ids, no stale bare 'dashscope'.

    AST-extracts the attribute docstring block attached to `provider: str | None`
    (not whole-file scan) so the test doesn't false-positive on comments or
    other fields that happen to contain the same token. Word-boundary match
    keeps `dashscope_qwen` / `dashscope_qwen_vl` distinct from bare `dashscope`.
    """
    import ast
    import re
    from pathlib import Path
    from app.domain.services.provider_profiles._registry import _REGISTRY

    # parents: [0]=provider_profiles, [1]=services, [2]=domain, [3]=tests, [4]=api
    src_path = (
        Path(__file__).resolve().parents[4] / "app" / "domain" / "models" / "app_config.py"
    )
    assert src_path.exists(), f"app_config.py not found at {src_path}"

    tree = ast.parse(src_path.read_text(encoding="utf-8"))

    def _find_provider_docstring(tree: ast.Module) -> str | None:
        for cls in ast.walk(tree):
            if not (isinstance(cls, ast.ClassDef) and cls.name == "LLMConfig"):
                continue
            body = cls.body
            for i, stmt in enumerate(body):
                is_provider_field = (
                    isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                    and stmt.target.id == "provider"
                )
                if not is_provider_field:
                    continue
                # Attribute docstring is the very next stmt: Expr(Constant(str)).
                if i + 1 < len(body):
                    nxt = body[i + 1]
                    if (
                        isinstance(nxt, ast.Expr)
                        and isinstance(nxt.value, ast.Constant)
                        and isinstance(nxt.value.value, str)
                    ):
                        return nxt.value.value
        return None

    provider_doc = _find_provider_docstring(tree)
    assert provider_doc is not None, "LLMConfig.provider attribute docstring not found"

    # 1) 每个 registered id 必须在 provider docstring 中出现（word boundary 匹配，
    #    避免 'kimi_k2' 也同时匹配 'kimi_k2_6' 等前缀场景）
    for pid in sorted(_REGISTRY.keys()):
        assert re.search(rf"\b{re.escape(pid)}\b", provider_doc), (
            f"provider_id '{pid}' missing from LLMConfig.provider docstring"
        )

    # 2) 废弃 id 'dashscope' 不得出现为独立 token（前缀形 dashscope_qwen /
    #    dashscope_qwen_vl 内的 'dashscope' 因后随 '_' 不成词边界，不会误匹配）
    assert re.search(r"\bdashscope\b", provider_doc) is None, (
        "Unregistered bare 'dashscope' token remains in LLMConfig.provider docstring"
    )

    # NOTE: 不再做"精确 token 集合相等"断言。那种检查要靠正则抠 "constants:"
    # 之后的 id 列表段，会把测试与当前 docstring 的标点/换行写法绑死，变成
    # 格式耦合而非语义 contract（codex v4 Round 8 P3-1）。Steps 1-2 合起来
    # 已锁住 "registered ids 都在 docstring 里 + 废弃 id 已撤" 的核心契约。
