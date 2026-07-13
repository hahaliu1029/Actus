"""D1a Task 13 — D1-6 fail-closed 矩阵收口（§4.2 / INV-D1-6 / R1#4）。

装配咽喉六门（G1/G2/G3/G4/G4b/G5）× 两种 port 失效形态 × 两 mode = 24 组合。

两种失效形态**语义必须一致**（R1#4）：
- 形态 A「port 返回 registry_unavailable 决策」：port 合同内部已把 registry 不可用
  折叠成 ``AdmissionDecision(reason="registry_unavailable")``（admitted 由
  ``decide_admitted`` 分流：detection 类 → shadow 准入 / enforce 拦）。
- 形态 B「port 方法直接 raise」：防实现 bug / 半坏 Fake——helper 内 try/except 兜底，
  经 ``_fallback_admitted(port)`` 按 ``port.mode`` 分流（同 ``decide_admitted`` 口径）。

两形态在同 mode 下**结果一致**：
- enforce → 剔除 / 阻断（G4b 返回 ``"registry_unavailable"`` → 调用方转
  extension_unavailable 工具错误，**不 500**）；
- shadow → 直通 + warn。

且任一门在任一失效形态下**都不得向会话冒泡异常**（返回可继续形态：G1/G4 过滤结果集、
G2/G3 blocked 集、G4b reason|None、G5 bool）。raise 形态 port 带 ``mode`` 属性
（Protocol 合同——``_fallback_admitted`` 的分流依据）。
"""
from __future__ import annotations

import pytest

from app.domain.external.extension_admission import AdmissionDecision
from app.domain.models.app_config import (
    A2AConfig,
    A2AServerConfig,
    MCPConfig,
    MCPServerConfig,
)
from app.domain.services.extension_admission_gates import (
    filter_a2a_config,
    filter_mcp_config,
    filter_skills,
    skill_invoke_gate,
    verify_a2a_cards,
    verify_mcp_surfaces,
    verify_skill_artifact,
)
from app.domain.services.extension_admission_logic import decide_admitted

pytestmark = pytest.mark.anyio

_UNAVAILABLE = "registry_unavailable"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _Skill:
    def __init__(self, key: str):
        self.id = key


class _Tool:
    def __init__(self, name: str):
        self.name = name
        self.description = "d"
        self.inputSchema = {"type": "object"}


class DecisionPort:
    """形态 A：port 把 registry 不可用折叠成 registry_unavailable 决策
    （admitted 由 decide_admitted 按 mode 分流——与真实 port 合同同源）。"""

    def __init__(self, mode: str):
        self.mode = mode

    def _decision(self) -> AdmissionDecision:
        return AdmissionDecision(decide_admitted(self.mode, _UNAVAILABLE), _UNAVAILABLE, None)

    async def check_many(self, kind, ext_ids, config_fingerprints=None):
        return {e: self._decision() for e in ext_ids}

    async def verify_observation(self, kind, ext_id, obs):
        return self._decision()


class RaisePort:
    """形态 B：port 方法直接 raise（半坏实现）——helper try/except 兜底走
    _fallback_admitted(port.mode)。带 mode 属性（Protocol 合同）。"""

    def __init__(self, mode: str):
        self.mode = mode

    async def check_many(self, kind, ext_ids, config_fingerprints=None):
        raise RuntimeError("registry down")

    async def verify_observation(self, kind, ext_id, obs):
        raise RuntimeError("registry down")


# --- 六门 → 归一化「是否直通」适配器（True=直通/未阻断，False=剔除/阻断）------- #


async def _g1_mcp(port) -> bool:
    out, _fps = await filter_mcp_config(
        port, MCPConfig(mcpServers={"s1": MCPServerConfig(url="https://s1")})
    )
    return bool(out.mcpServers)


async def _g1_a2a(port) -> bool:
    out, _fps = await filter_a2a_config(
        port, A2AConfig(a2a_servers=[A2AServerConfig(id="a1", base_url="https://a")])
    )
    return bool(out.a2a_servers)


async def _g2(port) -> bool:
    blocked = await verify_mcp_surfaces(port, {"s1": [_Tool("t1")]}, {"s1": "fp"})
    return blocked == set()


async def _g3(port) -> bool:
    blocked = await verify_a2a_cards(port, {"a1": {"name": "A"}}, {"a1": "fp"})
    return blocked == set()


async def _g4(port) -> bool:
    out = await filter_skills(port, [_Skill("k1")])
    return bool(out)


async def _g4b(port) -> bool:
    reason = await skill_invoke_gate(port, "k1")
    return reason is None


async def _g5(port) -> bool:
    return await verify_skill_artifact(port, "k1", "hash1")


# G1 拆 mcp/a2a 两个 check_many 消费者（同门两表面），共 7 个可直通性适配器。
_GATES = {
    "G1_mcp": _g1_mcp,
    "G1_a2a": _g1_a2a,
    "G2": _g2,
    "G3": _g3,
    "G4": _g4,
    "G4b": _g4b,
    "G5": _g5,
}
_FORMS = {"decision": DecisionPort, "raise": RaisePort}
_MODES = ("shadow", "enforce")


@pytest.mark.parametrize("gate_name", list(_GATES))
@pytest.mark.parametrize("form_name", list(_FORMS))
@pytest.mark.parametrize("mode", _MODES)
async def test_fail_closed_matrix(gate_name, form_name, mode):
    """24 组合（7 门 × 2 形态 × 2 mode，G1 拆两表面故 28）：helper 不冒泡异常、
    返回可继续形态；enforce 剔除/阻断、shadow 直通。两失效形态同 mode 下结果一致。"""
    port = _FORMS[form_name](mode)
    passed = await _GATES[gate_name](port)  # 绝不 raise——冒泡即 fail
    assert passed is (mode == "shadow"), (
        f"{gate_name}/{form_name}/{mode}: enforce 应剔除/阻断（passed=False）、"
        f"shadow 应直通（passed=True），实际 passed={passed}"
    )


@pytest.mark.parametrize("form_name", list(_FORMS))
async def test_g4b_enforce_returns_registry_unavailable_reason(form_name):
    """G4b enforce：两形态都返回 ``registry_unavailable`` reason 字符串
    （调用方据此转 extension_unavailable 工具错误，不 500）。"""
    port = _FORMS[form_name]("enforce")
    reason = await skill_invoke_gate(port, "k1")
    assert reason == _UNAVAILABLE


@pytest.mark.parametrize("form_name", list(_FORMS))
def test_ports_expose_mode_attribute(form_name):
    """两形态 port 均带 mode（Protocol 合同——fail-closed 分流依据）。"""
    for mode in _MODES:
        assert _FORMS[form_name](mode).mode == mode
