"""D1a §5.1 canonicalizer 纯函数族（HASH_SCHEMA_VERSION=1；升级 checklist 见 spec §14）。

per-kind pin 对象：
- mcp surface_hash：按 tool name 排序的 [{name, description, input_schema}]
- mcp config_fingerprint：{transport, command, args(原序), url, headers_keys, env_keys}
  ——排除 enabled 与 header/env 值（secrets 不入指纹；secret 值变更不触发 pin 重置，文档化接受）
- a2a surface_hash：卡片投影（不 hash 全卡——动态非安全字段防误伤）
- a2a config_fingerprint：{base_url}
- skill/plugin artifact_hash：复用 SkillsGuard.compute_content_hash（本文件不重复实现）
- entry_content_hash：§3.4 第三族（saga 所有权比对专用）——完整 dump **含 secrets 值**；
  仅入 operations.steps JSON（Admin-only），禁入 registry 列/audit/§5.1 pin 词表。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

from app.domain.models.app_config import MCPServerConfig


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def mcp_surface_hash(tools: Sequence[Mapping[str, Any]]) -> str:
    projected = sorted(
        (
            {
                "name": t.get("name"),
                "description": t.get("description"),
                # MCP wire=inputSchema / 内部=input_schema——归一后 canonical
                "input_schema": t.get("input_schema", t.get("inputSchema")),
            }
            for t in tools
        ),
        key=lambda t: str(t["name"]),
    )
    return sha256_hex(canonical_json(projected))


def mcp_config_fingerprint(config: MCPServerConfig) -> str:
    payload = {
        "transport": getattr(config.transport, "value", config.transport),
        "command": config.command,
        "args": list(config.args) if config.args is not None else None,  # 原序：参数顺序有语义
        "url": config.url,
        "headers_keys": sorted({str(k) for k in (config.headers or {})}),
        "env_keys": sorted({str(k) for k in (config.env or {})}),
    }
    return sha256_hex(canonical_json(payload))


def _sort_dedup_by_canonical(items: Iterable[Any]) -> list[Any]:
    seen: dict[str, Any] = {}
    for item in items:
        key = canonical_json(item)
        if key not in seen:
            seen[key] = item
    return [seen[k] for k in sorted(seen)]


def a2a_card_projection(card: Mapping[str, Any]) -> dict[str, Any]:
    """R2#7 缺失策略：缺失字段 → canonical null。

    F7 硬化（§5.1 canonical-form intent）：
    - capabilities **缺失/null → None**（canonical null）；**非法标量形状 → 哨兵 "<invalid>"**
      （区别于缺失——否则「缺 capabilities」与「capabilities=非法标量」两张语义不同的卡塌缩成
      同一 hash，rug-pull 可借非法形状伪装成缺失绕过 TOFU）。
    - authentication.schemes 的 **每个元素统一经 canonical_json 序列化**（含 plain string，
      §5.1「每元素各自 canonical JSON 字符串化」）：旧代码对非 dict/list 用 `str()`，使一个
      dict scheme 与「恰等于其 canonical JSON 的字符串」塌缩成同一 auth_schemes 条目（跨类型
      碰撞，rug-pull 可借此伪装，G3）；string 经 canonical_json → JSON 加引号形态（`"basic"`），
      保留类型信息且 dict 键排序确定（同一 scheme 键序不同不产生假 mismatch）。
    - **H2 硬化（跨来源命名空间）**：securitySchemes（取 key 名）与 authentication.schemes
      （取元素）两来源的每个投影元素各自 wrap `{"src": <source>, "v": <element>}` 后再
      canonical——否则 securitySchemes 的 key 恰为某 authentication.schemes 元素的 canonical
      JSON 字符串时，两来源塌缩进同一 auth_schemes 条目（一种 auth 结构可满足另一种的 surface
      pin，G4/H2）；wrap 后两 provenance 占不相交哈希空间。"""
    skills = card.get("skills")
    skills = _sort_dedup_by_canonical(skills) if isinstance(skills, list) else []
    capabilities = card.get("capabilities")
    if capabilities is None:
        cap_value: Any = None
    elif isinstance(capabilities, list):
        cap_value = _sort_dedup_by_canonical(capabilities)
    elif isinstance(capabilities, Mapping):
        cap_value = dict(capabilities)  # 对象形态原样（canonical_json sort_keys 已定序）
    else:
        cap_value = "<invalid>"  # 非法标量形状——与缺失(None) 显式区分
    auth_schemes: set[str] = set()
    sec = card.get("securitySchemes")
    if isinstance(sec, Mapping):
        # 未知 wire key 原样收录 scheme 名——但按 source 命名空间收录（H2）
        auth_schemes.update(
            canonical_json({"src": "securitySchemes", "v": str(k)}) for k in sec
        )
    auth = card.get("authentication")
    if isinstance(auth, Mapping) and isinstance(auth.get("schemes"), list):
        for s in auth["schemes"]:
            # G3：每元素统一 canonical_json（含 string，键序确定）；
            # H2：再按 source 命名空间 wrap——securitySchemes 与 authentication.schemes
            # 两来源占**不相交**哈希空间（否则 securitySchemes key 恰为某元素 canonical
            # JSON 串时跨来源塌缩，一种 auth 结构可满足另一种的 surface pin）。
            auth_schemes.add(canonical_json({"src": "authentication.schemes", "v": s}))
    return {
        "url": card.get("url"),
        "name": card.get("name"),
        "description": card.get("description"),
        "skills": skills,
        "capabilities": cap_value,
        "auth_schemes": sorted(auth_schemes),
    }


def a2a_surface_hash(card: Mapping[str, Any]) -> str:
    return sha256_hex(canonical_json(a2a_card_projection(card)))


def a2a_config_fingerprint(base_url: str) -> str:
    return sha256_hex(canonical_json({"base_url": base_url}))


def entry_content_hash(entry_dump: Mapping[str, Any]) -> str:
    return sha256_hex(canonical_json(dict(entry_dump)))
