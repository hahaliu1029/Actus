"""B9 run-scoped liveness 注册表（spec §4 R13#2/R14#1）。"""
from app.domain.services.runtime_liveness_registry import RuntimeLivenessRegistry


def test_acquire_release_roundtrip():
    reg = RuntimeLivenessRegistry()
    reg.begin_run("r1")
    reg.acquire("mcp", "srv", "r1")
    snap = reg.snapshot()
    assert snap.active[("mcp", "srv")] == frozenset({"r1"})
    reg.release("mcp", "srv", "r1")
    assert ("mcp", "srv") not in reg.snapshot().active     # prune 空 key


def test_release_idempotent_missing_key_no_raise():
    reg = RuntimeLivenessRegistry()
    reg.release("mcp", "ghost", "r1")     # 不抛
    reg.release("mcp", "ghost", "r1")


def test_end_run_clears_active_even_if_release_was_skipped():
    """R14#1 兜底：release 被吞后 end_run 仍清零该 run 的所有 active。"""
    reg = RuntimeLivenessRegistry()
    reg.begin_run("r1")
    reg.acquire("mcp", "a", "r1")
    reg.acquire("a2a", "b", "r1")
    reg.end_run("r1")
    assert reg.snapshot().active == {}


def test_degraded_run_scoped_reset():
    """R13#2：mark_degraded(run) → degraded=True；最后一个 degraded run 结束即复位。"""
    reg = RuntimeLivenessRegistry()
    reg.begin_run("r1"); reg.begin_run("r2")
    reg.mark_degraded("r1")
    assert reg.snapshot().degraded is True
    reg.end_run("r2")
    assert reg.snapshot().degraded is True      # r1 还在
    reg.end_run("r1")
    assert reg.snapshot().degraded is False     # 不粘住


def test_two_runs_same_extension_counted():
    reg = RuntimeLivenessRegistry()
    reg.begin_run("r1"); reg.begin_run("r2")
    reg.acquire("mcp", "srv", "r1")
    reg.acquire("mcp", "srv", "r2")
    assert reg.snapshot().active[("mcp", "srv")] == frozenset({"r1", "r2"})
    reg.end_run("r1")
    assert reg.snapshot().active[("mcp", "srv")] == frozenset({"r2"})


def test_snapshot_is_isolated_copy():
    reg = RuntimeLivenessRegistry()
    reg.begin_run("r1"); reg.acquire("mcp", "srv", "r1")
    snap = reg.snapshot()
    reg.end_run("r1")
    assert snap.active[("mcp", "srv")] == frozenset({"r1"})   # 旧快照不被后续变更污染
