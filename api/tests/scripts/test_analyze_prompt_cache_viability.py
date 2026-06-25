"""B5.5 T2 — unit tests for the offline prompt-cache viability analyzer.

The script lives under ``api/scripts/`` (not a package), so it is loaded by
file path via importlib rather than ``import scripts.*``.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analyze_prompt_cache_viability.py"
)
_spec = importlib.util.spec_from_file_location(
    "analyze_prompt_cache_viability", _SCRIPT
)
assert _spec and _spec.loader
mod = importlib.util.module_from_spec(_spec)
# Register in sys.modules BEFORE exec so the @dataclass module lookup
# (dataclasses._is_type → sys.modules.get(cls.__module__)) resolves.
sys.modules[_spec.name] = mod
_spec.loader.exec_module(mod)


def _llm(session_id, ts, sph, tools, sysbytes=4096):
    return {
        "ts": ts,
        "system_prompt_hash": sph,
        "system_prompt_bytes": sysbytes,
        "tools_hash": tools,
        "lang": "zh",
        "provider": "openai",
        "trace_id": "t" + str(ts),
        "request_id": "r" + str(ts),
        "session_id": session_id,
    }


def _assembly(session_id, fallback):
    return {
        "ts": "2026-06-25T00:00:00+00:00",
        "sections_included": ["identity"],
        "sections_dropped": [],
        "tokens_used": 1000,
        "lang": "zh",
        "provider": "openai",
        "mode": "full",
        "version_hash": "v1",
        "fallback_used": fallback,
        "trace_id": "ta",
        "request_id": "ra",
        "session_id": session_id,
    }


def test_stable_session_is_full_prefix_hit():
    # sysbytes 4096 / bpt 4.0 = 1024 tok >= min_prefix; ts gaps 1s < TTL.
    llm = [
        _llm("s1", "2026-06-25T00:00:01+00:00", "PH", "TH"),
        _llm("s1", "2026-06-25T00:00:02+00:00", "PH", "TH"),
        _llm("s1", "2026-06-25T00:00:03+00:00", "PH", "TH"),
    ]
    rep = mod.analyze([], llm)
    assert rep.sessions == 1
    assert rep.sessions_with_real_id == 1
    assert rep.total_prefix_pairs == 2
    assert rep.prefix_hit_pairs == 2
    assert rep.prefix_hit_rate == 1.0
    assert rep.avg_modal_sph_share == 1.0


def test_changing_prefix_has_no_hits():
    llm = [
        _llm("s1", "2026-06-25T00:00:01+00:00", "A", "TA"),
        _llm("s1", "2026-06-25T00:00:02+00:00", "B", "TB"),
        _llm("s1", "2026-06-25T00:00:03+00:00", "C", "TC"),
    ]
    rep = mod.analyze([], llm)
    assert rep.total_prefix_pairs == 2
    assert rep.prefix_hit_pairs == 0
    assert rep.prefix_hit_rate == 0.0


def test_mixed_valid_bad_ts_no_false_adjacency():
    # A bad-ts call between two stable valid calls must NOT be reordered aside
    # so the two valid calls become adjacent (regression: the old ts-sort moved
    # bad-ts records to the end). In append order both pairs touch the bad-ts
    # record and miss; the two valid calls never become a pair.
    llm = [
        _llm("s1", "2026-06-25T00:00:01+00:00", "PH", "TH"),
        _llm("s1", "bad-ts", "PH", "TH"),
        _llm("s1", "2026-06-25T00:00:03+00:00", "PH", "TH"),
    ]
    rep = mod.analyze([], llm)
    assert rep.total_prefix_pairs == 2
    assert rep.prefix_hit_pairs == 0  # both pairs include the unverifiable-ts call


def test_out_of_ttl_pair_not_counted():
    # stable + eligible, but 600s apart (> 300s TTL) → not a cache hit.
    llm = [
        _llm("s1", "2026-06-25T00:00:00+00:00", "PH", "TH"),
        _llm("s1", "2026-06-25T00:10:00+00:00", "PH", "TH"),
    ]
    rep = mod.analyze([], llm)
    assert rep.total_prefix_pairs == 1
    assert rep.prefix_hit_pairs == 0
    assert rep.pairs_out_of_ttl == 1
    assert rep.sph_stable_pairs == 1  # raw stability still observed


def test_below_min_prefix_not_counted():
    # stable + within TTL, but prefix far below min_prefix_tokens → not cacheable.
    llm = [
        _llm("s1", "2026-06-25T00:00:01+00:00", "PH", "TH", sysbytes=100),
        _llm("s1", "2026-06-25T00:00:02+00:00", "PH", "TH", sysbytes=100),
    ]
    rep = mod.analyze([], llm)
    assert rep.prefix_hit_pairs == 0
    assert rep.pairs_below_min_prefix == 1
    assert rep.est_cached_read_tokens == 0


def test_missing_system_prompt_hash_is_not_stable():
    # None == None must NOT read as a stable prompt.
    llm = [
        _llm("s1", "2026-06-25T00:00:01+00:00", None, "TH"),
        _llm("s1", "2026-06-25T00:00:02+00:00", None, "TH"),
    ]
    rep = mod.analyze([], llm)
    assert rep.sph_stable_pairs == 0
    assert rep.prefix_hit_pairs == 0


def test_missing_tools_hash_is_not_stable():
    llm = [
        _llm("s1", "2026-06-25T00:00:01+00:00", "PH", None),
        _llm("s1", "2026-06-25T00:00:02+00:00", "PH", None),
    ]
    rep = mod.analyze([], llm)
    assert rep.tools_stable_pairs == 0
    assert rep.prefix_hit_pairs == 0


def test_anon_records_do_not_merge_into_fake_session():
    # neither session_id nor trace_id → unique singleton keys, never merged.
    llm = [
        {"ts": "2026-06-25T00:00:01+00:00", "system_prompt_hash": "PH",
         "system_prompt_bytes": 4096, "tools_hash": "TH", "session_id": None,
         "trace_id": None},
        {"ts": "2026-06-25T00:00:02+00:00", "system_prompt_hash": "PH",
         "system_prompt_bytes": 4096, "tools_hash": "TH", "session_id": None,
         "trace_id": None},
    ]
    rep = mod.analyze([], llm)
    assert rep.sessions == 2  # two singletons, not one "trace:None" group
    assert rep.total_prefix_pairs == 0  # no cross-record pairs fabricated


def test_malformed_ts_does_not_count_as_hit():
    # unparseable ts → TTL window cannot be verified → conservative no-hit.
    llm = [
        _llm("s1", "not-a-timestamp", "PH", "TH"),
        _llm("s1", "also-bad", "PH", "TH"),
    ]
    rep = mod.analyze([], llm)
    assert rep.total_prefix_pairs == 1
    assert rep.prefix_hit_pairs == 0


def test_savings_estimate_counts_only_hit_prefix_bytes():
    # 3 stable in-ttl eligible calls → 2 hit pairs, each adds cur bytes/bpt.
    llm = [
        _llm("s1", "2026-06-25T00:00:01+00:00", "PH", "TH", sysbytes=4096),
        _llm("s1", "2026-06-25T00:00:02+00:00", "PH", "TH", sysbytes=4096),
        _llm("s1", "2026-06-25T00:00:03+00:00", "PH", "TH", sysbytes=4096),
    ]
    rep = mod.analyze([], llm, bytes_per_token=4.0)
    assert rep.prefix_hit_pairs == 2
    assert rep.est_cached_read_tokens == 2 * (4096 // 4)  # 2 hits × 1024 tok
    assert 0.0 < rep.est_cache_savings_rate <= 1.0


def test_fallback_rate():
    assembly = [
        _assembly("s1", False),
        _assembly("s1", True),
        _assembly("s1", False),
        _assembly("s1", True),
    ]
    rep = mod.analyze(assembly, [])
    assert rep.fallback_used == 2
    assert rep.fallback_rate == 0.5


def test_null_session_id_with_trace_falls_back_and_notes_coverage():
    llm = [
        _llm(None, "2026-06-25T00:00:01+00:00", "PH", "TH"),
        _llm(None, "2026-06-25T00:00:02+00:00", "PH", "TH"),
    ]
    rep = mod.analyze([], llm)
    assert rep.llm_with_session_id == 0
    assert rep.sessions_with_real_id == 0
    assert any("session_id" in n for n in rep.notes)


def test_real_session_id_grouping():
    llm = [
        _llm("sX", "2026-06-25T00:00:01+00:00", "PH", "TH"),
        _llm("sX", "2026-06-25T00:00:02+00:00", "PH", "TH"),
        _llm("sY", "2026-06-25T00:00:01+00:00", "QH", "UH"),
    ]
    rep = mod.analyze([], llm)
    assert rep.sessions == 2
    assert rep.sessions_with_real_id == 2
    assert rep.llm_with_session_id == 3
    assert rep.total_prefix_pairs == 1  # sX has 1 pair, sY has 0


def test_bytes_per_token_zero_raises():
    with pytest.raises(ValueError):
        mod.analyze([], [], bytes_per_token=0)


def test_evaluate_gate_go():
    rep = mod.ViabilityReport(
        sessions_with_real_id=10, total_prefix_pairs=50,
        fallback_rate=0.0, prefix_hit_rate=0.8,
    )
    verdict, _ = mod.evaluate_gate(rep, fallback_max=0.05, hit_min=0.5, min_sessions=5)
    assert verdict == "GO"


def test_evaluate_gate_nogo_on_high_fallback():
    rep = mod.ViabilityReport(
        sessions_with_real_id=10, total_prefix_pairs=50,
        fallback_rate=0.2, prefix_hit_rate=0.8,
    )
    verdict, _ = mod.evaluate_gate(rep, fallback_max=0.05, hit_min=0.5, min_sessions=5)
    assert verdict == "NO-GO"


def test_evaluate_gate_insufficient_on_few_sessions():
    rep = mod.ViabilityReport(
        sessions_with_real_id=1, total_prefix_pairs=30,
        fallback_rate=0.0, prefix_hit_rate=0.9,
    )
    verdict, reasons = mod.evaluate_gate(
        rep, fallback_max=0.05, hit_min=0.5, min_sessions=5
    )
    assert verdict == "INSUFFICIENT-DATA"
    assert any("min_sessions" in r for r in reasons)


def test_evaluate_gate_insufficient_on_few_pairs():
    # plenty of sessions claimed but only one measured pair → not GO.
    rep = mod.ViabilityReport(
        sessions_with_real_id=10, total_prefix_pairs=1,
        fallback_rate=0.0, prefix_hit_rate=1.0,
    )
    verdict, reasons = mod.evaluate_gate(
        rep, fallback_max=0.05, hit_min=0.5, min_sessions=5
    )
    assert verdict == "INSUFFICIENT-DATA"
    assert any("min_pairs" in r for r in reasons)


def test_read_jsonl_skips_malformed(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(
        '{"a": 1}\n'
        "not json\n"
        "\n"  # blank line ignored
        '{"b": 2}\n'
        "[1,2,3]\n",  # non-dict json → malformed
        encoding="utf-8",
    )
    records, malformed = mod._read_jsonl(p)
    assert len(records) == 2
    assert malformed == 2


def test_read_jsonl_missing_file(tmp_path):
    records, malformed = mod._read_jsonl(tmp_path / "nope.jsonl")
    assert records == []
    assert malformed == 0


def test_all_none_hash_session_not_counted_as_stable_modal():
    llm = [
        _llm("s1", "2026-06-25T00:00:01+00:00", None, None),
        _llm("s1", "2026-06-25T00:00:02+00:00", None, None),
    ]
    rep = mod.analyze([], llm)
    # no present hashes → no modal/distinct contribution (not a fake 1.0 share)
    assert rep.avg_modal_sph_share == 0.0
    assert rep.avg_distinct_sph_per_session == 0.0
    assert rep.prefix_hit_pairs == 0


def test_main_rejects_zero_bytes_per_token(tmp_path):
    rc = mod.main(["--bytes-per-token", "0", "--telemetry-dir", str(tmp_path)])
    assert rc == 2
