"""D1a §4.2 G1/G4 helper 合同（FakePort；shadow/enforce 分流已内化在 admitted）。"""
import pytest

from app.domain.external.extension_admission import AdmissionDecision
from app.domain.models.app_config import A2AConfig, A2AServerConfig, MCPConfig, MCPServerConfig
from app.domain.services.extension_admission_gates import (
    filter_a2a_config,
    filter_mcp_config,
    filter_skills,
    skill_invoke_gate,
    verify_a2a_cards,
    verify_mcp_surfaces,
    verify_skill_artifact,
)


class FakePort:
    def __init__(self, decisions):
        self.decisions = decisions
        self.calls = []

    async def check_many(self, kind, ext_ids, config_fingerprints=None):
        self.calls.append((kind, tuple(ext_ids), dict(config_fingerprints or {})))
        return {e: self.decisions.get(e, AdmissionDecision(True, "ok", 1)) for e in ext_ids}


def _mcp(names):
    return MCPConfig(mcpServers={n: MCPServerConfig(url=f"https://{n}") for n in names})


class _Skill:
    def __init__(self, key):
        self.id = key   # Skill 业务键=id（domain/models/skill.py:60；目录名/ext_id 同源）


@pytest.mark.anyio
async def test_none_port_identity():
    cfg = _mcp(["s1"])
    out, fps = await filter_mcp_config(None, cfg)
    assert out is cfg and fps == {}
    skills = [_Skill("k1")]
    assert await filter_skills(None, skills) is skills


@pytest.mark.anyio
async def test_g1_removes_non_admitted_and_passes_fingerprints():
    port = FakePort({"bad": AdmissionDecision(False, "config_drift", 2)})
    out, fps = await filter_mcp_config(port, _mcp(["good", "bad"]))
    assert set(out.mcpServers) == {"good"}
    kind, ids, sent_fps = port.calls[0]
    assert kind == "mcp" and set(ids) == {"good", "bad"}
    assert set(sent_fps) == {"good", "bad"}          # R1#3：随查随带指纹
    assert set(fps) == {"good", "bad"}               # G2 复用（R34#2 under_config 同源）


@pytest.mark.anyio
async def test_g1_a2a_filters_by_uuid():
    port = FakePort({"id-2": AdmissionDecision(False, "quarantined", 5)})
    cfg = A2AConfig(a2a_servers=[A2AServerConfig(id="id-1", base_url="https://a"),
                                 A2AServerConfig(id="id-2", base_url="https://b")])
    out, fps = await filter_a2a_config(port, cfg)
    assert [s.id for s in out.a2a_servers] == ["id-1"]
    assert set(fps) == {"id-1", "id-2"}


@pytest.mark.anyio
async def test_g4_removes_non_admitted_skills():
    port = FakePort({"k2": AdmissionDecision(False, "unpinned", 1)})
    out = await filter_skills(port, [_Skill("k1"), _Skill("k2")])
    assert [s.id for s in out] == ["k1"]
    assert port.calls[0][0] == "skill"


@pytest.mark.anyio
async def test_shadow_detection_not_removed_because_admitted_true():
    # shadow 分流在 port 内（decide_admitted）——helper 只看 admitted
    port = FakePort({"k2": AdmissionDecision(True, "unpinned", 1)})
    out = await filter_skills(port, [_Skill("k1"), _Skill("k2")])
    assert len(out) == 2


@pytest.mark.anyio
async def test_port_exception_mode_split():
    # R1#4/INV-D1-6：port 直接抛异常 → shadow fail-open 直通 / enforce fail-closed 全剔除
    class Boom:
        def __init__(self, mode):
            self.mode = mode
        async def check_many(self, *a, **k):
            raise RuntimeError("x")
    skills = [_Skill("k1")]
    assert await filter_skills(Boom("shadow"), skills) == skills
    assert await filter_skills(Boom("enforce"), skills) == []
    cfg = _mcp(["s1"])
    out_shadow, _ = await filter_mcp_config(Boom("shadow"), cfg)
    assert set(out_shadow.mcpServers) == {"s1"}
    out_enforce, _ = await filter_mcp_config(Boom("enforce"), cfg)
    assert out_enforce.mcpServers == {}


# ---------------------------------------------------------------------------
# T11: G2（MCP 表面校验）+ G3（A2A 卡片校验）
# ---------------------------------------------------------------------------


class _Tool:
    def __init__(self, name):
        self.name = name
        self.description = "d"
        self.inputSchema = {"type": "object"}


class FakeVerifyPort:
    def __init__(self, blocked=()):
        self.blocked = set(blocked)
        self.observations = []

    async def verify_observation(self, kind, ext_id, obs):
        self.observations.append((kind, ext_id, obs))
        from app.domain.external.extension_admission import AdmissionDecision
        return AdmissionDecision(ext_id not in self.blocked,
                                 "pin_mismatch" if ext_id in self.blocked else "ok", 1)


@pytest.mark.anyio
async def test_g2_blocks_mismatched_server_and_threads_under_config():
    port = FakeVerifyPort(blocked={"srv-bad"})
    blocked = await verify_mcp_surfaces(
        port,
        {"srv-ok": [_Tool("t1")], "srv-bad": [_Tool("t2")]},
        {"srv-ok": "fp-ok", "srv-bad": "fp-bad"},
    )
    assert blocked == {"srv-bad"}
    for kind, ext_id, obs in port.observations:
        assert kind == "mcp" and obs.category == "surface"
        assert obs.under_config_fingerprint == {"srv-ok": "fp-ok", "srv-bad": "fp-bad"}[ext_id]
        # payload 是 [{name, description, input_schema}] 形态（T5 canonicalizer 入参合同）
        assert obs.payload[0]["name"] in ("t1", "t2")


@pytest.mark.anyio
async def test_g3_blocks_and_verifies_cards():
    port = FakeVerifyPort(blocked={"id-2"})
    blocked = await verify_a2a_cards(
        port,
        {"id-1": {"name": "A"}, "id-2": {"name": "B"}},
        {"id-1": "fp1", "id-2": "fp2"},
    )
    assert blocked == {"id-2"}
    assert all(kind == "a2a" and obs.category == "surface"
               for kind, _e, obs in port.observations)


@pytest.mark.anyio
async def test_g2_none_port_identity_and_exception_mode_split():
    assert await verify_mcp_surfaces(None, {"s": [_Tool("t")]}, {}) == set()

    class Boom:
        def __init__(self, mode):
            self.mode = mode
        async def verify_observation(self, *a, **k):
            raise RuntimeError("x")
    # R1#4：shadow 直通 / enforce 剔除该 server
    assert await verify_mcp_surfaces(Boom("shadow"), {"s": [_Tool("t")]}, {}) == set()
    assert await verify_mcp_surfaces(Boom("enforce"), {"s": [_Tool("t")]}, {}) == {"s"}


# ---------------------------------------------------------------------------
# T12: G4b（invoke 前门 status check）+ G5（artifact verify）
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_g4b_gate_returns_reason_only_when_blocked():
    port = FakePort({"k-bad": AdmissionDecision(False, "quarantined", 1)})
    assert await skill_invoke_gate(port, "k-ok") is None
    assert await skill_invoke_gate(port, "k-bad") == "quarantined"
    assert await skill_invoke_gate(None, "k-bad") is None   # off 直通


@pytest.mark.anyio
async def test_g5_artifact_verify_maps_admitted():
    port = FakeVerifyPort(blocked={"k-bad"})
    assert await verify_skill_artifact(port, "k-ok", "hash1") is True
    assert await verify_skill_artifact(port, "k-bad", "hash1") is False
    obs = port.observations[0][2]
    assert obs.category == "artifact" and obs.payload == "hash1"   # payload=已算 hash（R9#3）


@pytest.mark.anyio
async def test_query_count_contract_per_invoke():
    # R3#4：per skill-invoke ≤ G4b 1 check + G5 1 check + 1 条件 verify
    port = FakePort({})
    vport = FakeVerifyPort()
    await skill_invoke_gate(port, "k")
    await skill_invoke_gate(port, "k")        # 模拟 G5 前置 check 复用同方法
    await verify_skill_artifact(vport, "k", "h")
    assert len(port.calls) == 2 and len(vport.observations) == 1
