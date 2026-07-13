"""D1a §4.2 装配咽喉 gate helpers（G1-G5 逻辑单点；port=None 恒等直通=off 语义）。

统一约定（§4.2）：shadow 下检测类 reason 只记录不剔除；行政/结构类
（quarantined/disabled/deleted/parent_blocked）双 mode 均剔除——分流已内化在
AdmissionDecision.admitted（port 侧 decide_admitted），helper 只按 admitted 执行。
helper 对 port 异常的兜底（port 合同内部已 catch；此处防实现 bug/Fake 抛出）**同样
按 mode 分流**（R1#4/INV-D1-6）：enforce=fail-closed 剔除/阻断、shadow=fail-open
直通 + warn，绝不向会话冒泡 500——经 `_fallback_admitted(port)` 统一判定。
"""
from __future__ import annotations

import logging
from typing import Any

from app.domain.external.extension_admission import Observation
from app.domain.models.app_config import A2AConfig, MCPConfig
from app.domain.models.extension_governance import HASH_SCHEMA_VERSION
from app.domain.services.extension_admission_logic import decide_admitted
from app.domain.services.extension_hashing import (
    a2a_config_fingerprint,
    mcp_config_fingerprint,
)

logger = logging.getLogger(__name__)


def _fallback_admitted(port: Any) -> bool:
    """port 抛异常时的兜底判定（INV-D1-6）：按 port.mode 对 registry_unavailable 分流。"""
    return decide_admitted(getattr(port, "mode", "shadow"), "registry_unavailable")

async def filter_mcp_config(port: Any, mcp_config: "MCPConfig") -> tuple["MCPConfig", dict[str, str]]:
    """G1（mcp）：未准入项剔除；返回 (过滤后 config, server→fingerprint map 供 G2 复用)。"""
    if port is None or not mcp_config or not mcp_config.mcpServers:
        return mcp_config, {}
    fingerprints = {name: mcp_config_fingerprint(cfg)
                    for name, cfg in mcp_config.mcpServers.items()}
    try:
        decisions = await port.check_many("mcp", list(fingerprints), config_fingerprints=fingerprints)
    except Exception:
        logger.warning("G1 mcp admission errored", exc_info=True)
        if _fallback_admitted(port):                      # R1#4：shadow fail-open
            return mcp_config, fingerprints
        return mcp_config.model_copy(update={"mcpServers": {}}), fingerprints   # enforce fail-closed
    kept = {name: cfg for name, cfg in mcp_config.mcpServers.items()
            if decisions.get(name) is None or decisions[name].admitted}
    removed = set(mcp_config.mcpServers) - set(kept)
    if removed:
        logger.info("G1 mcp admission removed servers=%s reasons=%s", sorted(removed),
                    {n: decisions[n].reason for n in removed})
    return mcp_config.model_copy(update={"mcpServers": kept}), fingerprints


async def filter_a2a_config(port: Any, a2a_config: "A2AConfig") -> tuple["A2AConfig", dict[str, str]]:
    if port is None or not a2a_config or not a2a_config.a2a_servers:
        return a2a_config, {}
    fingerprints = {s.id: a2a_config_fingerprint(s.base_url) for s in a2a_config.a2a_servers}
    try:
        decisions = await port.check_many("a2a", list(fingerprints), config_fingerprints=fingerprints)
    except Exception:
        logger.warning("G1 a2a admission errored", exc_info=True)
        if _fallback_admitted(port):
            return a2a_config, fingerprints
        return a2a_config.model_copy(update={"a2a_servers": []}), fingerprints
    kept = [s for s in a2a_config.a2a_servers
            if decisions.get(s.id) is None or decisions[s.id].admitted]
    return a2a_config.model_copy(update={"a2a_servers": kept}), fingerprints


async def filter_skills(port: Any, skills: list) -> list:
    """G4：**最终**选择集过滤（F22——team force-include 也不得越过 admission）。"""
    if port is None or not skills:
        return skills
    keys = [s.id for s in skills]   # Skill 业务键=id（skill.py:60）
    try:
        decisions = await port.check_many("skill", keys)
    except Exception:
        logger.warning("G4 skill admission errored", exc_info=True)
        return skills if _fallback_admitted(port) else []   # R1#4 mode 分流
    return [s for s in skills
            if decisions.get(s.id) is None or decisions[s.id].admitted]


async def skill_invoke_gate(port: Any, skill_key: str) -> str | None:
    """G4b/G5 前置 check：不准入 → 返回 reason（调用方转 extension_unavailable）；None=放行。"""
    if port is None:
        return None
    try:
        decisions = await port.check_many("skill", [skill_key])
    except Exception:
        logger.warning("skill admission gate errored for %s", skill_key, exc_info=True)
        return None if _fallback_admitted(port) else "registry_unavailable"   # R1#4
    d = decisions.get(skill_key)
    return None if d is None or d.admitted else d.reason


async def verify_skill_artifact(port: Any, skill_key: str, content_hash: str) -> bool:
    """G5 artifact verify（payload=已算 hash，R9#3/G5 零重复 I/O）。False=拒同步+阻断。"""
    if port is None:
        return True
    obs = Observation(category="artifact", payload=content_hash,
                      schema_version=HASH_SCHEMA_VERSION)
    try:
        return (await port.verify_observation("skill", skill_key, obs)).admitted
    except Exception:
        logger.warning("G5 artifact verify errored for %s", skill_key, exc_info=True)
        return _fallback_admitted(port)   # R1#4：enforce=False 拒同步+阻断


async def verify_mcp_surfaces(
    port: Any, tools_by_server: dict[str, list], config_fingerprints: dict[str, str],
) -> set[str]:
    """G2：initialize 完成后（schemas 在手）、工具构建前 per-server verify。
    verify 用的就是即将构建工具的同一份内存对象——无 TOCTOU 窗口（§4.2）。"""
    if port is None:
        return set()
    blocked: set[str] = set()
    for server_name, tools in tools_by_server.items():
        payload = [{"name": t.name, "description": t.description,
                    "input_schema": getattr(t, "inputSchema", None)} for t in tools]
        obs = Observation(category="surface", payload=payload,
                          schema_version=HASH_SCHEMA_VERSION,
                          under_config_fingerprint=config_fingerprints.get(server_name))
        try:
            decision = await port.verify_observation("mcp", server_name, obs)
        except Exception:
            logger.warning("G2 verify errored for %s", server_name, exc_info=True)
            if not _fallback_admitted(port):      # R1#4：enforce fail-closed 剔除该 server
                blocked.add(server_name)
            continue
        if not decision.admitted:
            blocked.add(server_name)
    return blocked


async def verify_a2a_cards(
    port: Any, agent_cards: dict[str, Any], config_fingerprints: dict[str, str],
) -> set[str]:
    """G3：拉卡后 per-agent verify；mismatch → 调用方从卡片集移除。"""
    if port is None:
        return set()
    blocked: set[str] = set()
    for agent_id, card in agent_cards.items():
        obs = Observation(category="surface", payload=card,
                          schema_version=HASH_SCHEMA_VERSION,
                          under_config_fingerprint=config_fingerprints.get(agent_id))
        try:
            decision = await port.verify_observation("a2a", agent_id, obs)
        except Exception:
            logger.warning("G3 verify errored for %s", agent_id, exc_info=True)
            if not _fallback_admitted(port):
                blocked.add(agent_id)
            continue
        if not decision.admitted:
            blocked.add(agent_id)
    return blocked
