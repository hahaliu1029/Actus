"""D1a §5.1 canonicalizer 契约（纯函数，HASH_SCHEMA_VERSION=1）。"""
from app.domain.models.app_config import MCPServerConfig
from app.domain.services.extension_hashing import (
    a2a_card_projection,
    a2a_config_fingerprint,
    a2a_surface_hash,
    canonical_json,
    entry_content_hash,
    mcp_config_fingerprint,
    mcp_surface_hash,
)


class TestMcpSurfaceHash:
    TOOLS = [
        {"name": "b_tool", "description": "B", "input_schema": {"type": "object"}},
        {"name": "a_tool", "description": "A", "input_schema": {"type": "object"}},
    ]

    def test_order_invariant(self):
        # 按 tool name 排序后 canonical——远端仅重排不产生新 hash
        assert mcp_surface_hash(self.TOOLS) == mcp_surface_hash(list(reversed(self.TOOLS)))

    def test_description_change_changes_hash(self):
        changed = [dict(self.TOOLS[0], description="B'"), self.TOOLS[1]]
        assert mcp_surface_hash(changed) != mcp_surface_hash(self.TOOLS)

    def test_input_schema_wire_alias_accepted(self):
        # MCP wire 用 inputSchema；内部路径用 input_schema——canonical 前归一
        wire = [{"name": "a_tool", "description": "A", "inputSchema": {"type": "object"}}]
        internal = [{"name": "a_tool", "description": "A", "input_schema": {"type": "object"}}]
        assert mcp_surface_hash(wire) == mcp_surface_hash(internal)


class TestMcpConfigFingerprint:
    def _cfg(self, **kw):
        base = dict(transport="stdio", command="npx", args=["-y", "server"],
                    env={"API_KEY": "secret-1"}, headers={"X-Auth": "tok"})
        base.update(kw)
        return MCPServerConfig(**base)

    def test_excludes_enabled_and_values(self):
        # spec §5.1：排除 enabled 与 header/env 值——secret 值变更不触发指纹变化（文档化接受）
        assert mcp_config_fingerprint(self._cfg(enabled=True)) == mcp_config_fingerprint(self._cfg(enabled=False))
        assert (mcp_config_fingerprint(self._cfg(env={"API_KEY": "secret-1"}))
                == mcp_config_fingerprint(self._cfg(env={"API_KEY": "secret-2"})))

    def test_key_set_changes_fingerprint(self):
        assert (mcp_config_fingerprint(self._cfg(env={"API_KEY": "x"}))
                != mcp_config_fingerprint(self._cfg(env={"API_KEY": "x", "EXTRA": "y"})))

    def test_keys_sorted_dedup_args_order_preserved(self):
        # R32#7：headers/env 键名排序去重；args 保持原序（命令参数顺序有语义）
        assert (mcp_config_fingerprint(self._cfg(env={"B": "1", "A": "2"}))
                == mcp_config_fingerprint(self._cfg(env={"A": "9", "B": "8"})))
        assert (mcp_config_fingerprint(self._cfg(args=["-y", "server"]))
                != mcp_config_fingerprint(self._cfg(args=["server", "-y"])))


class TestA2aCard:
    FULL = {
        "url": "https://agent.example", "name": "Summarizer", "description": "d",
        "skills": [{"id": "s2"}, {"id": "s1"}],
        "capabilities": {"streaming": True},
        "securitySchemes": {"bearer": {"type": "http"}},
    }

    def test_full_card_deterministic(self):
        assert a2a_surface_hash(self.FULL) == a2a_surface_hash(dict(self.FULL))

    def test_missing_fields_canonical_null_no_error(self):
        # R2#7：缺失字段 → canonical null；不报错
        h = a2a_surface_hash({"name": "X"})
        assert isinstance(h, str) and len(h) == 64

    def test_unknown_auth_scheme_included_verbatim(self):
        # H2：scheme 名按 source 命名空间收录（wrap {"src","v"}）——名仍原样保留在 v。
        card = dict(self.FULL, securitySchemes={"x-custom-scheme": {}})
        proj = a2a_card_projection(card)
        assert '{"src":"securitySchemes","v":"x-custom-scheme"}' in proj["auth_schemes"]

    def test_array_reorder_and_dup_hash_invariant(self):
        # R32#7：数组元素 canonical 字符串化 → 排序+去重——重排/重复同义列表项不产生新 hash
        reordered = dict(self.FULL, skills=[{"id": "s1"}, {"id": "s2"}, {"id": "s1"}])
        assert a2a_surface_hash(reordered) == a2a_surface_hash(self.FULL)

    def test_dynamic_field_not_hashed(self):
        # 不 hash 全卡：非投影字段（如注入的 enabled）变化不影响 hash
        assert a2a_surface_hash(dict(self.FULL, enabled=False)) == a2a_surface_hash(self.FULL)


class TestA2aMalformedShapeF7:
    """F7 回归（§5.1 canonical-form intent）：
    (a) 缺失 capabilities (None) 与非法标量 capabilities 必须区分（不塌缩成同一 hash）；
    (b) authentication.schemes 的 dict 项经 canonical_json（键排序）而非 str()（dict-repr 键序非确定）。"""

    def test_absent_vs_invalid_capabilities_hash_differently(self):
        # 缺失 vs 非法标量：语义不同 → hash 必须不同
        absent = {"name": "X"}                               # capabilities 缺失
        invalid = {"name": "X", "capabilities": "streaming"}  # 非法标量形状
        assert a2a_surface_hash(absent) != a2a_surface_hash(invalid)

    def test_invalid_scalar_capabilities_projects_sentinel(self):
        proj = a2a_card_projection({"name": "X", "capabilities": 42})
        assert proj["capabilities"] == "<invalid>"
        assert a2a_card_projection({"name": "X"})["capabilities"] is None

    def test_dict_scheme_entries_order_invariant(self):
        # authentication.schemes 的 dict 项：键序不同的同一 dict → hash 相同
        card1 = {"name": "X", "authentication": {"schemes": [{"type": "http", "scheme": "bearer"}]}}
        card2 = {"name": "X", "authentication": {"schemes": [{"scheme": "bearer", "type": "http"}]}}
        assert a2a_surface_hash(card1) == a2a_surface_hash(card2)

    def test_dict_scheme_distinct_from_different_dict(self):
        card1 = {"name": "X", "authentication": {"schemes": [{"scheme": "bearer"}]}}
        card2 = {"name": "X", "authentication": {"schemes": [{"scheme": "basic"}]}}
        assert a2a_surface_hash(card1) != a2a_surface_hash(card2)


class TestA2aSchemeCanonicalG3:
    """G3 回归（§5.1 canonical-form intent）：authentication.schemes 的**每个元素**（含
    plain string）统一经 canonical_json 投影——旧代码对非 dict/list 用 str()，导致一个
    dict scheme 与「恰等于其 canonical JSON 的字符串」塌缩成同一 auth_schemes 条目（跨类型
    碰撞，rug-pull 可借此伪装）。"""

    def test_cross_type_scheme_no_collision(self):
        # 审计 repro：dict scheme vs 恰等于其 canonical JSON 的 string——必须 hash 不同
        card_dict = {"name": "X",
                     "authentication": {"schemes": [{"type": "oauth2", "flows": ["pkce"]}]}}
        card_str = {"name": "X",
                    "authentication": {"schemes": ['{"flows":["pkce"],"type":"oauth2"}']}}
        assert a2a_surface_hash(card_dict) != a2a_surface_hash(card_str)

    def test_string_scheme_json_quoted_and_deterministic(self):
        # string scheme 经 canonical_json 且按 source 命名空间 wrap（H2）→ 确定且可排序。
        card = {"name": "X", "authentication": {"schemes": ["basic", "bearer"]}}
        proj = a2a_card_projection(card)
        assert proj["auth_schemes"] == [
            '{"src":"authentication.schemes","v":"basic"}',
            '{"src":"authentication.schemes","v":"bearer"}',
        ]
        assert a2a_surface_hash(card) == a2a_surface_hash(dict(card))


class TestA2aSchemeCrossSourceH2:
    """H2 回归（round-3 审计，§5.1 canonical-form intent）：securitySchemes 与
    authentication.schemes 两来源的投影落入**同一** auth_schemes 哈希空间——securitySchemes
    的 key 恰为某 authentication.schemes 元素的 canonical JSON 字符串时跨来源塌缩，一种 auth
    结构可满足另一种的 surface pin。修：每 scheme 元素按 source 命名空间 wrap
    `{"src": <source>, "v": <element>}` 后再 canonical，两 provenance 占不相交哈希空间。"""

    def test_cross_source_same_literal_hashes_differently(self):
        # 审计 exact repro：securitySchemes key == '{"type":"oauth2"}'（已是 JSON 串）
        # vs authentication.schemes=[{"type":"oauth2"}]（dict 投影）——必须 hash 不同。
        card_sec = {"name": "X", "securitySchemes": {'{"type":"oauth2"}': {}}}
        card_auth = {"name": "X", "authentication": {"schemes": [{"type": "oauth2"}]}}
        assert a2a_surface_hash(card_sec) != a2a_surface_hash(card_auth)

    def test_source_label_present_in_both_projections(self):
        # 两来源的 auth_schemes 元素都带各自的 src 标签。
        sec_proj = a2a_card_projection({"name": "X", "securitySchemes": {"bearer": {}}})
        auth_proj = a2a_card_projection(
            {"name": "X", "authentication": {"schemes": [{"scheme": "bearer"}]}})
        assert all('"src":"securitySchemes"' in e for e in sec_proj["auth_schemes"])
        assert all('"src":"authentication.schemes"' in e for e in auth_proj["auth_schemes"])

    def test_dict_scheme_key_order_still_identical(self):
        # G3 不回归 F7：dict scheme 键序不同仍 canonical → 同一 hash
        card1 = {"name": "X", "authentication": {"schemes": [{"type": "http", "scheme": "bearer"}]}}
        card2 = {"name": "X", "authentication": {"schemes": [{"scheme": "bearer", "type": "http"}]}}
        assert a2a_surface_hash(card1) == a2a_surface_hash(card2)


class TestA2aConfigFingerprint:
    def test_base_url_only(self):
        assert a2a_config_fingerprint("https://a") != a2a_config_fingerprint("https://b")
        assert a2a_config_fingerprint("https://a") == a2a_config_fingerprint("https://a")


class TestEntryContentHash:
    def test_secret_value_participates(self):
        # R50#1：完整 dump 含 secrets 值——仅 secret 值差异也必须区分（所有权精确等值）
        e1 = {"transport": "stdio", "env": {"KEY": "v1"}, "enabled": True}
        e2 = {"transport": "stdio", "env": {"KEY": "v2"}, "enabled": True}
        assert entry_content_hash(e1) != entry_content_hash(e2)

    def test_key_order_invariant(self):
        assert (entry_content_hash({"a": 1, "b": 2})
                == entry_content_hash({"b": 2, "a": 1}))


def test_canonical_json_cjk_stable():
    # ensure_ascii=False：CJK 不转义，跨进程稳定
    assert canonical_json({"名": "值"}) == '{"名":"值"}'
