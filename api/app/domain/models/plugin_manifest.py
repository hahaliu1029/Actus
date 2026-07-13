"""D1a §8.1 Plugin 元容器 manifest（`plugin.json`）领域模型。

纯 pydantic 校验（domain 纯度：仅 pydantic + stdlib + 同层 domain 模型）：

- ``PluginManifest``（``extra="forbid"``）：``manifest_version`` 前向兼容门（``>1`` 拒装，
  hermes 先例）；``id`` 稳定身份 charset 门；``version`` **路径安全门**（直接构成
  ``/app/data/plugins/{id}/{version}/`` 存储段——首字符字母数字排除 ``.``/``..``、charset
  无 ``/`` 不可穿越，R46#4）。
- ``detect_secret_literals``：mcp ``config`` 的 ``env``/``headers`` 值若非 ``${VAR}`` 占位符
  形式且命中 credential 正则 → ``secret_literal`` dangerous finding（§8.1 **提示性防误提交**，
  非 secret isolation 承诺——Base64/拆分可绕过，stdio 环境继承见 §1.3/F26）。
"""
from __future__ import annotations

import re
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from app.domain.models.app_config import MCPServerConfig
from app.domain.models.extension_governance import GovernanceScanFinding

# id 稳定身份（reverse-DNS 风格建议非强制）；首字符字母数字 → {id} 路径段排除 `..`
MANIFEST_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]{2,63}$"
# version 路径安全门（R46#4）：首字符字母数字排除 `.`/`..` 段；charset 无 `/` 不可穿越
MANIFEST_VERSION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"

# GovernanceScanFinding str 字段上限（与 skill_service/extension_scan 同值）
_FINDING_FIELD_MAX = 256

# 占位符形式（`${VAR}` / `$VAR` 均视作环境引用，不是字面 secret）——起始 `$` 即豁免
_PLACEHOLDER_RE = re.compile(r"^\$")
# credential 值形状（高特异——近零 benign 误伤，与 extension_scan._CREDENTIAL_VALUE_RES 对齐）
_SECRET_VALUE_RES = (
    re.compile(r"^eyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]*$"),  # JWT
    re.compile(r"^sk-[A-Za-z0-9_-]{16,}$"),                       # OpenAI-style secret key
    re.compile(r"^AKIA[0-9A-Z]{16}$"),                           # AWS access key id
    re.compile(r"^gh[pousr]_[A-Za-z0-9]{20,}$"),                 # GitHub PAT/token family
    re.compile(r"(?i)^bearer\s+\S+$"),                           # explicit Bearer <token>
    re.compile(r"^xox[baprs]-[A-Za-z0-9-]{10,}$"),              # Slack token family
)


class UnsupportedManifestVersionError(Exception):
    """``manifest_version > 1``——前向兼容门（interfaces 映射 422 拒装，T24）。

    plain ``Exception`` 保 domain 纯度（不引 application.errors）；HTTP 映射归 interfaces。
    """

    def __init__(self, version: int) -> None:
        super().__init__(f"unsupported plugin manifest_version: {version}")
        self.version = version


class PluginSkillComponent(BaseModel):
    """skill 组件条目：``id`` = manifest 内声明名，``path`` = bundle 内子目录。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    path: str
    expected_hash: str | None = None


class PluginMCPComponent(BaseModel):
    """mcp 组件条目：``config`` = ``MCPServerConfig`` 同构（declared id 即 server_name）。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    config: MCPServerConfig
    expected_hash: str | None = None


class PluginA2AComponent(BaseModel):
    """a2a 组件条目：``base_url`` = 远端 agent；落地 ext_id = preflight 预分配 uuid（R19#F1）。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    base_url: str
    expected_hash: str | None = None


class PluginComponents(BaseModel):
    """§8.1 三类组件容器。"""

    model_config = ConfigDict(extra="forbid")

    skills: list[PluginSkillComponent] = Field(default_factory=list)
    mcp_servers: list[PluginMCPComponent] = Field(default_factory=list)
    a2a_agents: list[PluginA2AComponent] = Field(default_factory=list)


class PluginManifest(BaseModel):
    """`plugin.json` 结构（§8.1）。经 ``parse()`` 统一入口构造——version 前向门先于
    pydantic 字段校验（version=2 需抛 ``UnsupportedManifestVersionError`` 而非 pydantic
    ``ValidationError``）。"""

    model_config = ConfigDict(extra="forbid")

    manifest_version: int
    id: str = Field(pattern=MANIFEST_ID_PATTERN)
    name: str
    version: str = Field(pattern=MANIFEST_VERSION_PATTERN)
    description: str = ""
    components: PluginComponents = Field(default_factory=PluginComponents)

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "PluginManifest":
        """唯一构造入口：前向兼容门（``manifest_version > 1`` → 拒）先于字段校验。"""
        version = raw.get("manifest_version")
        # bool 是 int 子类——排除；仅正整数 >1 触发前向门
        if isinstance(version, int) and not isinstance(version, bool) and version > 1:
            raise UnsupportedManifestVersionError(version)
        return cls.model_validate(raw)


def _looks_like_secret_literal(value: str) -> bool:
    if not value or _PLACEHOLDER_RE.match(value):
        return False
    return any(rx.match(value) for rx in _SECRET_VALUE_RES)


def detect_secret_literals(config: MCPServerConfig) -> list[GovernanceScanFinding]:
    """§8.1 secret 检查：``env``/``headers`` 值非占位符且命中 credential 正则 → dangerous
    ``secret_literal`` finding（severity=critical → dangerous verdict）。

    **提示性防误提交**（非 isolation 承诺）：只捕高特异凭据形状，避免误伤端口/布尔/日志级别
    等 benign 值；path 字段截断 ≤256 防 pydantic 校验失败（key 用户可控）。
    """
    findings: list[GovernanceScanFinding] = []
    for source_name, mapping in (("env", config.env), ("headers", config.headers)):
        if not mapping:
            continue
        for key, value in mapping.items():
            if not isinstance(value, str):
                continue
            if _looks_like_secret_literal(value):
                findings.append(
                    GovernanceScanFinding(
                        category="secret_literal",
                        severity="critical",  # → dangerous（_findings_to_summary 映射）
                        pattern_id="secret_literal",
                        path=f"{source_name}:{key}"[:_FINDING_FIELD_MAX],
                        line=None,
                    )
                )
    return findings
