"""T21 — PluginManifest 校验 + secret-literal 探测（D1a §8.1）。

纯 pydantic 校验：manifest_version 前向门、id/version charset+路径安全门、
extra=forbid、合法 roundtrip、mcp config secret-literal 提示性 finding。
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.domain.models.app_config import MCPServerConfig, MCPTransport
from app.domain.models.plugin_manifest import (
    PluginManifest,
    UnsupportedManifestVersionError,
    detect_secret_literals,
)


def _valid_raw() -> dict:
    return {
        "manifest_version": 1,
        "id": "org.example.research-pack",
        "name": "Research Pack",
        "version": "1.2.0",
        "description": "A research bundle",
        "components": {
            "skills": [{"id": "web-clipper", "path": "skills/web-clipper"}],
            "mcp_servers": [
                {"id": "search", "config": {"transport": "streamable_http",
                                            "url": "https://search.test/mcp"}}
            ],
            "a2a_agents": [{"id": "summarizer", "base_url": "https://sum.test"}],
        },
    }


# ------------------------------------------------------------- manifest_version --
def test_manifest_version_gt_one_rejected():
    """manifest_version > 1 → UnsupportedManifestVersionError（前向兼容门，hermes 先例）。"""
    raw = _valid_raw()
    raw["manifest_version"] = 2
    with pytest.raises(UnsupportedManifestVersionError):
        PluginManifest.parse(raw)


def test_manifest_version_one_ok():
    manifest = PluginManifest.parse(_valid_raw())
    assert manifest.manifest_version == 1


# --------------------------------------------------------------------- id gate --
def test_id_invalid_charset_rejected():
    """id 含非法字符（大写/空格）→ pydantic ValidationError。"""
    raw = _valid_raw()
    raw["id"] = "Org Example"
    with pytest.raises(PydanticValidationError):
        PluginManifest.parse(raw)


def test_id_too_short_rejected():
    """id 过短（< 3 chars，regex {2,63} 尾段）→ ValidationError。"""
    raw = _valid_raw()
    raw["id"] = "ab"
    with pytest.raises(PydanticValidationError):
        PluginManifest.parse(raw)


# ---------------------------------------------------------------- version gate --
def test_version_path_traversal_rejected():
    """R46#5 断言⑤：version=\"../evil\" → ValidationError（路径穿越关闭——首字符门 + 无 /）。"""
    raw = _valid_raw()
    raw["version"] = "../evil"
    with pytest.raises(PydanticValidationError):
        PluginManifest.parse(raw)


def test_version_leading_dot_rejected():
    """version=\".hidden\" → 拒（首字符必须字母数字，排除 . / .. 段）。"""
    raw = _valid_raw()
    raw["version"] = ".hidden"
    with pytest.raises(PydanticValidationError):
        PluginManifest.parse(raw)


# ------------------------------------------------------------- forbid / roundtrip --
def test_extra_forbid():
    """extra=\"forbid\"：manifest 顶层未知键 → ValidationError。"""
    raw = _valid_raw()
    raw["unexpected_key"] = "boom"
    with pytest.raises(PydanticValidationError):
        PluginManifest.parse(raw)


def test_valid_full_manifest_roundtrip():
    """合法完整 manifest：全字段解析 + 组件三类保真。"""
    manifest = PluginManifest.parse(_valid_raw())
    assert manifest.id == "org.example.research-pack"
    assert manifest.name == "Research Pack"
    assert manifest.version == "1.2.0"
    assert [c.id for c in manifest.components.skills] == ["web-clipper"]
    assert manifest.components.skills[0].path == "skills/web-clipper"
    assert manifest.components.mcp_servers[0].id == "search"
    assert isinstance(manifest.components.mcp_servers[0].config, MCPServerConfig)
    assert manifest.components.a2a_agents[0].base_url == "https://sum.test"


def test_component_expected_hash_optional():
    """每类组件条目 optional expected_hash（R3#12 统一字段名）。"""
    raw = _valid_raw()
    raw["components"]["skills"][0]["expected_hash"] = "sha256:abc"
    manifest = PluginManifest.parse(raw)
    assert manifest.components.skills[0].expected_hash == "sha256:abc"
    assert manifest.components.mcp_servers[0].expected_hash is None


# ------------------------------------------------------------- secret literals --
def test_secret_literal_env_flagged_dangerous():
    """§8.1：env 值非 ${VAR} 占位符且命中 credential 正则 → secret_literal finding
    （severity=critical → dangerous verdict；提示性防误提交）。"""
    config = MCPServerConfig(
        transport=MCPTransport.STDIO, command="run",
        env={"OPENAI_API_KEY": "sk-abcdefghijklmnop0123456789"},
    )
    findings = detect_secret_literals(config)
    assert len(findings) == 1
    assert findings[0].category == "secret_literal"
    assert findings[0].severity == "critical"
    assert "env" in findings[0].path


def test_secret_literal_placeholder_not_flagged():
    """${VAR} 占位符形式 → 零 finding（合法环境变量引用）。"""
    config = MCPServerConfig(
        transport=MCPTransport.STDIO, command="run",
        env={"OPENAI_API_KEY": "${OPENAI_API_KEY}"},
    )
    assert detect_secret_literals(config) == []


def test_secret_literal_headers_flagged():
    """headers 值同受检（Bearer token 内联）。"""
    config = MCPServerConfig(
        transport=MCPTransport.STREAMABLE_HTTP, url="https://x.test/mcp",
        headers={"Authorization": "Bearer sk-livetoken0123456789abcdef"},
    )
    findings = detect_secret_literals(config)
    assert any(f.category == "secret_literal" for f in findings)


def test_secret_literal_benign_value_not_flagged():
    """普通非凭据值（端口/布尔/日志级别）→ 零 finding（低误报）。"""
    config = MCPServerConfig(
        transport=MCPTransport.STDIO, command="run",
        env={"PORT": "8080", "DEBUG": "true", "LOG_LEVEL": "info"},
    )
    assert detect_secret_literals(config) == []
