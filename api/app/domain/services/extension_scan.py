"""D1a §7.3/§3.1 扩展扫描与来源脱敏（canonicalize + T18 扫描/policy）。

- ``canonicalize_source_ref``（T5）：URI userinfo/凭据 query 剥离 + 敏感本地路径拒存。
- ``scan_mcp_entry`` / ``scan_a2a_entry``（§7.3）：复用 ``SkillsGuard`` 静态正则，
  按类别子集扫 config 拼接文本 + 表面 descriptions/卡片文本 → 投影为
  ``GovernanceScanSummary``（禁原始 match 文本入库，INV-D1-7）。
- ``evaluate_install_policy``（§7.2）：三态决策表（off/shadow/enforce × safe/caution/dangerous）。
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.domain.models.extension_governance import (
    GovernanceScanFinding,
    GovernanceScanSummary,
)
from app.domain.services.skills_guard import ScanFinding, SkillsGuard
from app.domain.services.trust_matrix import get_install_decision

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.domain.models.app_config import MCPServerConfig

# 凭据 query 参数用**三层判定**（round-3→5 审计收敛）。此判定是纵深防御的一层（userinfo 已
# 另剥，audit.py 对存储 payload 另有凭据值 regex）；lexical 无法在无字典下完美分离全小写连写的
# `masterkey`（凭据）与 `monkey`（benign）——按 INV-D1-7「宁删勿留」，over-redaction 是可接受
# 的安全方向（与 audit.py R3-P3 accepted limitation 对称；见 _STEM_SUFFIXES 注）。
#   token 化：切 -/_/./[/] 与 camelCase 驼峰边界，每 token 去尾部数字串（apikey2→apikey）。
#   Tier A（整 token，短/歧义词）：只作**整 token** 命中——design/author/region/signal 含这些
#     字母序列但非整 token → 不误伤。
#   Tier B（子串，长/高特异词）：拼接后**包含**即命中——高特异词几乎不出现在 benign 名里，
#     安全覆盖 access_token2 / clientSecret 等复合变体。
#   Tier C（stem 后缀，round-5 收敛）：凭据复合词把敏感 stem 作**尾部中心名**
#     （master·key / auth·key / api·key / signing·key），benign 前缀词把它作前缀（key·word /
#     key·board）或巧合后缀（mon·key / don·key）。故「整拼接名 endswith stem 且更长」判敏感——
#     捕获全部 `<qualifier>key/auth/sig/mac` 凭据；代价是 monkey/donkey/turnkey 一并 redact
#     （accepted per INV-D1-7），而 keyword/keyboard（stem 作前缀）不误伤。
# 敏感 iff Tier A 或 Tier B 或 Tier C。
_TIER_A_WHOLE_TOKENS = frozenset({"key", "sig", "auth", "iv", "mac"})
_TIER_B_SUBSTRINGS = (
    "token", "secret", "password", "passwd", "signature", "credential",
    "apikey", "accesskey", "authorization", "privatekey", "clientsecret",
)
# Tier C：拼接名 endswith 这些 stem 之一且比 stem 长 → 复合凭据名（endswith 而非 contains，
# 使 stem 作前缀的 benign 词 keyword/keyboard 不误伤）。
_STEM_SUFFIXES = ("key", "auth", "sig", "mac")
# Tier D（value-shape，round-6 收敛）：name-based 三层无法穷举 credential *名词*（jwt/otp/pat…
# 是全新词而非 key/auth 拼写变体——枚举永无止境）。故补一层**看值不看名**：任一 query 值命中
# 高特异凭据形状（JWT / sk-openai / AWS AKIA / GitHub PAT / 明确 Bearer）即脱敏该参数，与参数名
# 无关——一举封死「未知 credential 参数名」整类（audit.py 对存储 payload 亦有值层 regex，对称）。
# 形状高特异 → 近零 benign 误伤（git SHA / UUID / 版本号均不匹配这些前缀锚定的模式）。
_CREDENTIAL_VALUE_RES = (
    re.compile(r"^eyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]*$"),  # JWT (header.payload.sig)
    re.compile(r"^sk-[A-Za-z0-9_-]{16,}$"),                      # OpenAI-style secret key
    re.compile(r"^AKIA[0-9A-Z]{16}$"),                           # AWS access key id
    re.compile(r"^gh[pousr]_[A-Za-z0-9]{20,}$"),                 # GitHub PAT/token family
    re.compile(r"^(?i:bearer)\s+\S+$"),                          # explicit Bearer <token>
    re.compile(r"^xox[baprs]-[A-Za-z0-9-]{10,}$"),              # Slack token family
)


def _query_value_is_credential(value: str) -> bool:
    return any(rx.match(value) for rx in _CREDENTIAL_VALUE_RES)
# 分隔符：-/_/. 与方括号 [ ]（auth[token] 之类嵌套/数组参数名）。
_TOKEN_DELIM_RE = re.compile(r"[-_.\[\]]+")
# camelCase 驼峰边界：小写/数字→大写（authToken→auth|Token），或大写→大写+小写
# （缩写词尾，XMLParser→XML|Parser）。
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_TRAILING_DIGITS_RE = re.compile(r"\d+$")
_SENSITIVE_PATH_MARKERS = (".ssh", ".env", "credentials", "secrets", "id_rsa")


def _path_is_sensitive(path: str) -> bool:
    lowered = path.lower()
    return any(marker in lowered for marker in _SENSITIVE_PATH_MARKERS)


def _query_key_tokens(key: str) -> list[str]:
    """切分隔符 + camelCase 驼峰边界 → 每 token 去尾部数字串 → lowercase。"""
    tokens: list[str] = []
    for chunk in _TOKEN_DELIM_RE.split(key):
        for piece in _CAMEL_BOUNDARY_RE.split(chunk):
            stripped = _TRAILING_DIGITS_RE.sub("", piece).lower()
            if stripped:
                tokens.append(stripped)
    return tokens


def _query_key_is_sensitive(key: str) -> bool:
    tokens = _query_key_tokens(key)
    if any(tok in _TIER_A_WHOLE_TOKENS for tok in tokens):   # Tier A：整 token
        return True
    joined = "".join(tokens)                                  # 去分隔符+去尾数字后拼接
    if any(sub in joined for sub in _TIER_B_SUBSTRINGS):     # Tier B：高特异词子串
        return True
    return any(                                              # Tier C：stem 作尾部中心名
        joined.endswith(stem) and len(joined) > len(stem)
        for stem in _STEM_SUFFIXES
    )


def canonicalize_source_ref(ref: str | None) -> str | None:
    """R7#6：持久化与一切响应只用 canonical 值——剥 URI userinfo 与凭据 query 参数；
    可疑本地敏感路径拒原样入库。R5#3：file: URI 的 path 同受敏感路径检查
    （否则 file:///home/x/.env 走 URI 分支原样保留，本地检查永远到不了）。

    R2b#3：本函数是纵深防御一层（plugin preflight 在 try 块**之前**调用它、audit
    sanitize 亦复用），**必须永不抛**。``urlsplit``/``urlunsplit`` 对畸形 URL（如
    ``https://[invalid`` → ``Invalid IPv6 URL``）抛裸 ``ValueError``，无 handler 映射
    → 全局 500。畸形 URL 无法安全解析剥离凭据 → 按 INV-D1-7「宁删勿留」过度脱敏兜底
    （返回 ``<redacted-source-ref>``，绝不原样保留可能内嵌 userinfo 的畸形串）。"""
    if ref is None:
        return None
    if "://" in ref:
        try:
            parts = urlsplit(ref)
            if parts.scheme == "file" and _path_is_sensitive(parts.path):
                return "<redacted-local-path>"
            netloc = parts.netloc.rsplit("@", 1)[-1]          # 剥 userinfo
            query = urlencode([                                # 名敏感(A/B/C) 或 值命中凭据形状(D) → 剥
                (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                if not (_query_key_is_sensitive(k) or _query_value_is_credential(v))
            ])
            return urlunsplit((parts.scheme, netloc, parts.path, query, ""))
        except ValueError:
            # 畸形 URL（urlsplit/urlunsplit 崩溃）——不可安全解析 → 过度脱敏（INV-D1-7）
            return "<redacted-source-ref>"
    if _path_is_sensitive(ref):
        return "<redacted-local-path>"
    return ref


# ======================================================================
# §7.3 MCP/A2A 扫描（复用 SkillsGuard，不自建引擎）
# ======================================================================

# 配置静态检查类别子集（spec §7.3 line517 权威）：注意 supply_chain（curl|sh）不在其内——
# 该扩大留 D1c 外接扫描器，本期不越权改子集。
_CONFIG_SCAN_CATEGORIES = frozenset(
    {"injection", "destructive", "network", "credential_exposure"}
)
# 表面注入扫描类别（tool-poisoning 启发式，spec §7.3）：工具 descriptions / 卡片文本。
_SURFACE_SCAN_CATEGORIES = frozenset({"injection", "credential_exposure"})
# GovernanceScanFinding str 字段上限（与 skill_service._GOV_FINDING_FIELD_MAX 同值）。
_GOV_FINDING_FIELD_MAX = 256

PolicyDecision = Literal[
    "allow", "allow_with_warnings", "need_acknowledge", "need_force"
]


def _findings_to_summary(findings: list[ScanFinding]) -> GovernanceScanSummary:
    """内存 ``ScanFinding`` 列表 → 持久化/DTO 唯一 scan 形态（§7.3 / INV-D1-7）。

    verdict = severity 映射的 max（critical→dangerous / high|medium→caution / 否则 safe，
    镜像 ``ScanReport.from_findings``）；findings 截断 ≤50、字段截断 ≤256、**丢弃原始
    match 文本**（``GovernanceScanFinding`` 无 match 字段——结构性保证）。
    """

    def _s(value: Any) -> str:
        return str(value or "")[:_GOV_FINDING_FIELD_MAX]

    projected = [
        GovernanceScanFinding(
            category=_s(f.category),
            severity=_s(f.severity),
            pattern_id=_s(f.pattern_id),
            path=_s(f.file),   # ScanFinding.file → 受控 schema.path
            line=f.line,
        )
        for f in findings[:50]
    ]
    if any(f.severity == "critical" for f in findings):
        verdict = "dangerous"
    elif any(f.severity in ("high", "medium") for f in findings):
        verdict = "caution"
    else:
        verdict = "safe"
    return GovernanceScanSummary(
        verdict=verdict, finding_count=len(findings), findings=projected
    )


def _is_insecure_http(url: str | None) -> bool:
    """``http://``（非 https）→ True。"""
    return bool(url) and url.strip().lower().startswith("http://")


def _insecure_http_finding() -> ScanFinding:
    """`http://` 明文传输 → 提示性 caution finding（§7.3）；match 空——不落原文。"""
    return ScanFinding(
        pattern_id="insecure_http_url",
        category="network",
        severity="medium",       # → caution（_findings_to_summary 映射）
        file="config",
        line=1,
        match="",
    )


def _scan_surface_text(
    guard: SkillsGuard, text: str | None, path: str
) -> list[ScanFinding]:
    """对单段表面文本（工具/卡片 description）跑受限类别扫描；空文本零 finding。"""
    if not text:
        return []
    return guard._scan_text(
        str(text), file_path=path, line_num=1,
        categories=_SURFACE_SCAN_CATEGORIES,
    )


def scan_mcp_entry(
    server_name: str,
    config: MCPServerConfig,
    surface_payload: list | None,
) -> GovernanceScanSummary:
    """§7.3 MCP 扫描：

    - **配置静态检查**：stdio ``command``/``args`` 拼接文本 + ``url`` → 类别子集
      {injection, destructive, network, credential_exposure}；``http://`` → caution finding。
    - **表面注入扫描**：probe 观测到的工具 ``description`` → {injection, credential_exposure}。

    ``surface_payload``：probe 成功时的 ``[{name, description, input_schema}]``（None=probe
    失败/未观测，只扫 config）。返回值经 ``_findings_to_summary`` 投影，禁原始 match 入库。
    """
    guard = SkillsGuard()
    findings: list[ScanFinding] = []

    parts: list[str] = []
    if config.command:
        parts.append(str(config.command))
    if config.args:
        parts.extend(str(arg) for arg in config.args)
    if config.url:
        parts.append(str(config.url))
    config_text = " ".join(part for part in parts if part)
    if config_text:
        findings.extend(
            guard._scan_text(
                config_text, file_path="config", line_num=1,
                categories=_CONFIG_SCAN_CATEGORIES,
            )
        )
    if _is_insecure_http(config.url):
        findings.append(_insecure_http_finding())

    for tool in surface_payload or []:
        if not isinstance(tool, Mapping):
            continue
        findings.extend(
            _scan_surface_text(
                guard, tool.get("description"),
                path=f"surface:{tool.get('name')}",
            )
        )
    return _findings_to_summary(findings)


def _a2a_skill_text(skill: Any) -> str:
    """a2a 卡片 skill 元素 → 可扫描文本（dict=name+description 拼接 / str=原样）。"""
    if isinstance(skill, Mapping):
        return " ".join(
            str(skill.get(key))
            for key in ("name", "description")
            if skill.get(key)
        )
    if isinstance(skill, str):
        return skill
    return ""


def scan_a2a_entry(
    base_url: str,
    card: Mapping | None,
) -> GovernanceScanSummary:
    """§7.3 A2A 扫描：``base_url``（config 类别子集 + ``http://`` → caution）+
    卡片 ``description`` / ``skills`` 文本（{injection, credential_exposure}）。"""
    guard = SkillsGuard()
    findings: list[ScanFinding] = []

    if base_url:
        findings.extend(
            guard._scan_text(
                str(base_url), file_path="config", line_num=1,
                categories=_CONFIG_SCAN_CATEGORIES,
            )
        )
        if _is_insecure_http(base_url):
            findings.append(_insecure_http_finding())

    if isinstance(card, Mapping):
        findings.extend(
            _scan_surface_text(guard, card.get("description"), path="surface:card")
        )
        skills = card.get("skills")
        if isinstance(skills, list):
            for index, skill in enumerate(skills):
                findings.extend(
                    _scan_surface_text(
                        guard, _a2a_skill_text(skill),
                        path=f"surface:skill[{index}]",
                    )
                )
    return _findings_to_summary(findings)


# ======================================================================
# §7.2 INSTALL_POLICY 泛化（三态决策表，R1#11）
# ======================================================================


def evaluate_install_policy(
    mode: str,
    verdict: str,
    *,
    acknowledged: bool,
    forced: bool,
) -> PolicyDecision:
    """§7.2 三态决策表：``INSTALL_POLICY["user_installed"]`` 三元组（allow/warn/block
    按 verdict）+ mode 分流。

    - **off** → ``allow``（恒，治理不拦）；
    - **shadow** → safe=``allow`` / (caution|dangerous)=``allow_with_warnings``（记录不拦）；
    - **enforce** → safe=``allow`` / caution=``need_acknowledge`` 除非 ``acknowledged``
      （→ ``allow_with_warnings``）/ dangerous=``need_force`` 除非 ``forced``
      （→ ``allow_with_warnings``）。unknown verdict 回退 dangerous（``get_install_decision``）。
    """
    if mode == "off":
        return "allow"
    tier = get_install_decision("user_installed", verdict)  # allow / warn / block
    if tier == "allow":
        return "allow"
    if mode == "shadow":
        return "allow_with_warnings"
    # enforce
    if tier == "warn":   # caution
        return "allow_with_warnings" if acknowledged else "need_acknowledge"
    # tier == "block"（dangerous 或 unknown verdict 回退）
    return "allow_with_warnings" if forced else "need_force"
