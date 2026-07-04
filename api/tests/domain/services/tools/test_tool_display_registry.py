"""B10 §7.1 — tool_display_registry 单测.

覆盖: 38 canonical 全量分类锁 + family 兜底 (SV4) + fail-closed None
+ INV-B10-3 (registry ↔ _STATIC_RISK 一致性) + INV-B10-7 (import gate, §7.5d).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.domain.services.risk_assessor import _STATIC_RISK, RiskLevel
from app.domain.services.tools.tool_display_registry import (
    _CANONICAL_DISPLAY,
    DISPLAY_ICON_VOCABULARY,
    DisplayAttachment,
    ToolDisplayMeta,
    resolve_display_attachment,
    resolve_display_meta,
)
from app.domain.services.tools.tool_source_resolver import (
    _CANONICAL_TOOL_IDENTITIES,
    ToolSource,
)

# spec §4.1 全量分类表逐行锁定 (name, icon, read_only, destructive)
_EXPECTED_TABLE: list[tuple[str, str, bool, bool]] = [
    ("browser_view", "browser", True, False),
    ("browser_navigate", "browser", False, False),
    ("browser_click", "browser", False, False),
    ("browser_input", "browser", False, False),
    ("browser_move_mouse", "browser", False, False),
    ("browser_press_key", "browser", False, False),
    ("browser_select_option", "browser", False, False),
    ("browser_scroll_up", "browser", True, False),
    ("browser_scroll_down", "browser", True, False),
    ("browser_console_exec", "browser", False, True),
    ("browser_console_view", "browser", True, False),
    ("browser_restart", "browser", False, False),
    ("shell_execute", "terminal", False, True),
    ("shell_read_output", "terminal", True, False),
    ("shell_wait_process", "terminal", True, False),
    ("shell_write_input", "terminal", False, False),
    ("shell_kill_process", "terminal", False, True),
    ("file_read", "file", True, False),
    ("file_write", "file-edit", False, False),
    ("file_str_replace", "file-edit", False, False),
    ("file_find_in_content", "file", True, False),
    ("file_find_by_name", "file", True, False),
    ("file_list", "file", True, False),
    ("file_view", "file", False, False),
    ("message_notify_user", "message", False, False),
    ("message_ask_user", "message", False, False),
    ("search_web", "search", True, False),
    ("memory_search", "memory", True, False),
    ("memory_get", "memory", True, False),
    ("memory_save", "memory", False, False),
    ("get_remote_agent_cards", "a2a", True, False),
    ("call_remote_agent", "a2a", False, False),
    ("list_mcp_tools", "mcp", True, False),
    ("get_mcp_tool", "mcp", False, False),
    ("brainstorm_skill", "skill", False, False),
    ("generate_skill", "skill", False, False),
    ("install_skill", "skill", False, False),
    ("get_skill_guide", "skill", True, False),
]


class TestCanonicalCoverage:
    def test_registry_keys_exactly_match_canonical_identities(self):
        assert set(_CANONICAL_DISPLAY) == set(_CANONICAL_TOOL_IDENTITIES)

    def test_expected_table_is_complete(self):
        assert len(_EXPECTED_TABLE) == 38
        assert {row[0] for row in _EXPECTED_TABLE} == set(_CANONICAL_DISPLAY)

    @pytest.mark.parametrize(
        "name,icon,read_only,destructive",
        _EXPECTED_TABLE,
        ids=[row[0] for row in _EXPECTED_TABLE],
    )
    def test_classification_matches_spec_table(
        self, name: str, icon: str, read_only: bool, destructive: bool
    ):
        meta = _CANONICAL_DISPLAY[name]
        assert meta == ToolDisplayMeta(
            display_icon=icon, read_only=read_only, destructive=destructive
        )

    def test_summary_counts(self):
        # spec §4.1 汇总: read_only=16, destructive=3
        assert sum(1 for m in _CANONICAL_DISPLAY.values() if m.read_only) == 16
        assert {n for n, m in _CANONICAL_DISPLAY.items() if m.destructive} == {
            "shell_execute", "browser_console_exec", "shell_kill_process",
        }

    def test_all_icons_in_controlled_vocabulary(self):
        for name, meta in _CANONICAL_DISPLAY.items():
            assert meta.display_icon in DISPLAY_ICON_VOCABULARY, name


class TestFamilyFallback:
    """SV4: family 兜底按已解析 tool_source.source 分派, 用无前缀真实动态名测 (R5#4)."""

    def test_dynamic_mcp_tool_gets_mcp_family_meta(self):
        src = ToolSource(source="mcp", category="mcp", canonical_name="weather_lookup")
        meta = resolve_display_meta("weather_lookup", src)
        assert meta == ToolDisplayMeta(
            display_icon="mcp", read_only=False, destructive=False
        )

    def test_dynamic_skill_tool_gets_skill_family_meta(self):
        src = ToolSource(source="skill", category="skill", canonical_name="pdf_extract")
        meta = resolve_display_meta("pdf_extract", src)
        assert meta == ToolDisplayMeta(
            display_icon="skill", read_only=False, destructive=False
        )

    def test_dynamic_a2a_tool_gets_a2a_family_meta(self):
        src = ToolSource(source="a2a", category="a2a", canonical_name="translator_agent")
        meta = resolve_display_meta("translator_agent", src)
        assert meta == ToolDisplayMeta(
            display_icon="a2a", read_only=False, destructive=False
        )

    def test_native_source_canonical_miss_returns_none(self):
        # native family 无兜底 (spec §4.1 表只列 mcp/skill/a2a) — fail-closed None
        src = ToolSource(source="native", category="file", canonical_name="file_read")
        assert resolve_display_meta("file_read_v2", src) is None

    def test_none_tool_source_returns_none(self):
        # INV-B10-1(a): resolver 失败/幻觉名 → None
        assert resolve_display_meta("totally_hallucinated_tool", None) is None

    def test_canonical_hit_wins_over_family_fallback(self):
        src = ToolSource(
            source="mcp", category="mcp discovery", canonical_name="list_mcp_tools"
        )
        meta = resolve_display_meta("list_mcp_tools", src)
        assert meta is not None
        assert meta.read_only is True  # canonical 表值, 非 family 的恒 False


class TestStaticRiskConsistency:
    """INV-B10-3: HIGH ⊆ destructive; read_only ∩ (≥MEDIUM) = ∅ (防两表漂移)."""

    def test_high_static_risk_is_subset_of_destructive(self):
        high = {t for t, lvl in _STATIC_RISK.items() if lvl is RiskLevel.HIGH}
        destructive = {n for n, m in _CANONICAL_DISPLAY.items() if m.destructive}
        assert high <= destructive

    def test_read_only_disjoint_from_medium_or_higher_risk(self):
        medium_plus = {
            t for t, lvl in _STATIC_RISK.items()
            if lvl in (RiskLevel.MEDIUM, RiskLevel.HIGH)
        }
        read_only = {n for n, m in _CANONICAL_DISPLAY.items() if m.read_only}
        assert read_only.isdisjoint(medium_plus)


class TestImportGate:
    """INV-B10-7 / §7.5d: registry 模块零外部层 import."""

    def test_registry_module_is_domain_pure(self):
        src_path = (
            Path(__file__).resolve().parents[4]
            / "app" / "domain" / "services" / "tools" / "tool_display_registry.py"
        )
        tree = ast.parse(src_path.read_text(encoding="utf-8"))
        banned = (
            "fastapi", "sqlalchemy",
            "app.application", "app.interfaces", "app.infrastructure",
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not any(
                    name == b or name.startswith(b + ".") for b in banned
                ), f"banned import {name!r} in tool_display_registry.py"


class TestResolveDisplayAttachment:
    """spec §4.2 attach helper — 纯函数, flag 语义, INV-B10-2 永不抛出."""

    def test_disabled_returns_all_none(self):
        att = resolve_display_attachment("file_read", enabled=False)
        assert att == DisplayAttachment(
            tool_source=None, display_icon=None, read_only=None, destructive=None
        )

    def test_disabled_ignores_existing_tool_source(self):
        # flag-off 语义: attachment 全 None; CALLED 构造点不消费 att.tool_source
        # (保持既有 tool_source=tool_source 实参), 故不产生行为变化 (R12#1)
        src = ToolSource(source="native", category="file", canonical_name="file_read")
        att = resolve_display_attachment(
            "file_read", enabled=False, existing_tool_source=src
        )
        assert att.tool_source is None

    def test_enabled_canonical_tool_resolves_source_and_meta(self):
        att = resolve_display_attachment("file_read", enabled=True)
        assert att.tool_source is not None
        assert att.tool_source.source == "native"
        assert att.tool_source.category == "file"
        assert att.display_icon == "file"
        assert att.read_only is True
        assert att.destructive is False

    def test_enabled_existing_tool_source_not_overwritten(self):
        # D6: existing 优先不覆盖; R8#3: 已有 tool_source 时仍跑 registry
        src = ToolSource(source="mcp", category="mcp", canonical_name="weather_lookup")
        att = resolve_display_attachment(
            "weather_lookup", enabled=True, existing_tool_source=src
        )
        assert att.tool_source is src
        assert att.display_icon == "mcp"      # family 兜底
        assert att.read_only is False          # INV-B10-1(b): 策略位恒 false
        assert att.destructive is False

    def test_enabled_hallucinated_name_never_raises(self):
        # INV-B10-2: resolver 失败 → tool_source=None, 三元组全 None
        att = resolve_display_attachment("no_such_tool_xyz", enabled=True)
        assert att.tool_source is None
        assert att.display_icon is None
        assert att.read_only is None
        assert att.destructive is None

    def test_enabled_destructive_tool(self):
        att = resolve_display_attachment("shell_execute", enabled=True)
        assert att.destructive is True
        assert att.read_only is False
        assert att.display_icon == "terminal"
