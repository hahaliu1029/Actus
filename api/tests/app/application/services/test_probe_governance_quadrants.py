"""D1a probe 观测四象限（D1 mode × B9 probe flag）+ R1#20 观测生产链（spec §5.2 R2#4）。

四象限：
- off×on   → admission_port=None：探测照常，**零 registry 交互**（无 verify_observation）。
- shadow×on→ 探测成功 → verify_observation（surface + 自算 §5.1 指纹）；观测**不拦** probe。
- enforce×on→ 同上；隔离语义在 port 内（mismatch）——probe 主流程不受治理裁决影响。
- *×off    → probe flag off：后台 tick 早退 / 手动 probe ProbeDisabledError——零观测。

R2#F7：手动路（``_probe_locked``）+ 后台路（``_background_probe``）**两处成功写点都接**。
"""
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.extension_probe_service import (
    ExtensionProbeService,
    ProbeDisabledError,
    ProbeOutcome,
)
from app.domain.external.extension_admission import AdmissionDecision
from app.domain.models.app_config import (
    A2AConfig,
    A2AServerConfig,
    AppConfig,
    MCPConfig,
    MCPServerConfig,
    MCPTransport,
)
from app.domain.services.extension_hashing import (
    a2a_config_fingerprint,
    mcp_config_fingerprint,
)

T0 = datetime(2026, 7, 11, 12, 0, 0, tzinfo=timezone.utc)


class FakeClock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class FakeProber:
    """成功探测填充 surface_payload（mcp=工具列表 / a2a=原始卡 dict）。"""

    def __init__(self) -> None:
        self.mcp_surface = [
            {"name": "t1", "description": "desc", "input_schema": {}},
        ]
        self.a2a_surface = {"name": "agent", "description": "card"}
        self.calls: list[tuple[str, str]] = []

    async def probe_mcp(self, server_name, config) -> ProbeOutcome:
        self.calls.append(("mcp", server_name))
        return ProbeOutcome(
            ok=True, latency_ms=10, tool_count=len(self.mcp_surface),
            surface_payload=self.mcp_surface,
        )

    async def probe_a2a(self, config) -> ProbeOutcome:
        self.calls.append(("a2a", config.id))
        return ProbeOutcome(
            ok=True, latency_ms=10, display_name="agent",
            surface_payload=self.a2a_surface,
        )


class FakePort:
    """记录 verify_observation / check_many 调用（治理观测断言用）。"""

    def __init__(self, mode: str = "shadow", admitted: bool = True) -> None:
        self.mode = mode
        self._admitted = admitted
        self.verify_calls: list = []          # [(kind, ext_id, Observation)]
        self.check_many_calls: list = []

    async def verify_observation(self, kind, ext_id, obs) -> AdmissionDecision:
        self.verify_calls.append((kind, ext_id, obs))
        return AdmissionDecision(
            admitted=self._admitted, reason="ok", row_revision=1,
            observation_outcome="persisted",
        )

    async def check_many(self, kind, ext_ids, config_fingerprints=None):
        self.check_many_calls.append((kind, list(ext_ids)))
        return {
            e: AdmissionDecision(admitted=True, reason="ok", row_revision=1)
            for e in ext_ids
        }


class World:
    def __init__(self) -> None:
        self.mcp: dict[str, MCPServerConfig] = {
            "srv": MCPServerConfig(
                transport=MCPTransport.STDIO, command="npx", enabled=True
            ),
        }
        self.a2a = [
            A2AServerConfig(id="a2a-1", base_url="http://remote:9000", enabled=True)
        ]
        self.flag = True

    def config(self) -> AppConfig:
        return AppConfig.model_construct(
            llm_config=MagicMock(), agent_config=MagicMock(),
            mcp_config=MCPConfig(mcpServers=dict(self.mcp)),
            a2a_config=A2AConfig(a2a_servers=list(self.a2a)),
        )


def _make(world=None, *, prober=None, clock=None, admission_port=None, flag=True):
    world = world or World()
    world.flag = flag
    skill_repo = MagicMock()
    skill_repo.list_with_diagnostics = AsyncMock(return_value=[])
    svc = ExtensionProbeService(
        config_provider=world.config,
        skill_repository=skill_repo,
        prober=prober or FakeProber(),
        probe_flag_provider=lambda: world.flag,
        clock=clock or FakeClock(),
        rng=lambda: 0.5,
        admission_port=admission_port,
    )
    return world, svc


# —— R1#20 观测生产链核心（手动路）——


@pytest.mark.anyio
async def test_manual_probe_success_produces_surface_observation():
    world = World()
    port = FakePort(mode="shadow")
    prober = FakeProber()
    _, svc = _make(world, prober=prober, admission_port=port)

    record = await svc.probe_one_manual("mcp", "srv")

    assert record.state == "reachable"
    assert len(port.verify_calls) == 1
    kind, ext_id, obs = port.verify_calls[0]
    assert (kind, ext_id) == ("mcp", "srv")
    assert obs.category == "surface"
    assert obs.payload == prober.mcp_surface          # surface_payload 透传
    assert obs.under_config_fingerprint == mcp_config_fingerprint(world.mcp["srv"])


@pytest.mark.anyio
async def test_manual_probe_no_port_zero_registry_interaction():
    # off 象限：admission_port=None → 探测行为零变化、零 registry 交互
    world = World()
    prober = FakeProber()
    _, svc = _make(world, prober=prober, admission_port=None)

    record = await svc.probe_one_manual("mcp", "srv")

    assert record.state == "reachable"
    assert record.tool_count == 1
    assert prober.calls == [("mcp", "srv")]   # 探测照常执行


# —— 四象限 ——


@pytest.mark.anyio
async def test_quadrant_shadow_on_observes_without_blocking():
    world = World()
    port = FakePort(mode="shadow", admitted=True)
    _, svc = _make(world, admission_port=port, flag=True)

    record = await svc.probe_one_manual("mcp", "srv")

    assert record.state == "reachable"        # 观测不影响探测结果
    assert len(port.verify_calls) == 1


@pytest.mark.anyio
async def test_quadrant_enforce_on_observes_probe_unaffected_by_mismatch():
    # enforce + mismatch（admitted=False）：隔离语义在 port 内；probe 主流程不受影响
    world = World()
    port = FakePort(mode="enforce", admitted=False)
    _, svc = _make(world, admission_port=port, flag=True)

    record = await svc.probe_one_manual("mcp", "srv")

    assert record.state == "reachable"        # verify 是辅助证据，不改探测记录
    assert len(port.verify_calls) == 1
    _, _, obs = port.verify_calls[0]
    assert obs.category == "surface"
    assert obs.under_config_fingerprint == mcp_config_fingerprint(world.mcp["srv"])


@pytest.mark.anyio
async def test_quadrant_flag_off_background_no_observation():
    # *×off：flag off → tick_once 早退 → 无探测 → 无观测（不影响 bind gate）
    world = World()
    port = FakePort(mode="enforce")
    _, svc = _make(world, admission_port=port, flag=False)

    await svc.tick_once()

    assert port.verify_calls == []
    assert port.check_many_calls == []


@pytest.mark.anyio
async def test_quadrant_flag_off_manual_probe_disabled_no_observation():
    world = World()
    port = FakePort(mode="enforce")
    _, svc = _make(world, admission_port=port, flag=False)

    with pytest.raises(ProbeDisabledError):
        await svc.probe_one_manual("mcp", "srv")

    assert port.verify_calls == []


# —— 后台路（R2#F7 第二写点）——


@pytest.mark.anyio
async def test_background_probe_success_produces_surface_observation():
    world = World()
    port = FakePort(mode="shadow")
    clock = FakeClock()
    _, svc = _make(world, admission_port=port, clock=clock, flag=True)

    # 首轮 tick：仅播种 unknown（next_probe_at=now+60），不探测 → 零观测
    await svc.tick_once()
    assert port.verify_calls == []

    # 推进过到期点，再 tick → mcp + a2a 均探测 → 两处观测（后台写点接通）
    clock.advance(61)
    await svc.tick_once()

    obs_by_key = {(k, e): o for k, e, o in port.verify_calls}
    assert ("mcp", "srv") in obs_by_key
    assert ("a2a", "a2a-1") in obs_by_key
    assert obs_by_key[("mcp", "srv")].category == "surface"
    assert obs_by_key[("mcp", "srv")].under_config_fingerprint == (
        mcp_config_fingerprint(world.mcp["srv"])
    )
    assert obs_by_key[("a2a", "a2a-1")].under_config_fingerprint == (
        a2a_config_fingerprint("http://remote:9000")
    )
