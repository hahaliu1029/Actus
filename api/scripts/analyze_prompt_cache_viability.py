#!/usr/bin/env python3
"""B5.5 prompt-cache viability analyzer (offline, stdlib-only).

Reads the two prompt-telemetry JSONL streams written by
``JsonlPromptTelemetry`` (``app/infrastructure/telemetry/prompt_telemetry.py``)
and reports whether enabling Anthropic-style prompt caching — a
``cache_control`` breakpoint placed AFTER the stable ``system_prompt`` + tools
prefix — would yield real cache hits.

A consecutive pair of LLM calls in the same session counts as a CACHE HIT only
when ALL of:
  - both calls expose a non-null ``system_prompt_hash`` and they are equal, AND
  - both calls expose a non-null ``tools_hash`` and they are equal, AND
  - the second call lands within ``cache_ttl_seconds`` of the first (Anthropic
    prompt caches expire after ~5 minutes — calls further apart re-pay the full
    input cost), AND
  - the cached prefix is large enough to be cacheable at all
    (``system_prompt_bytes / bytes_per_token >= min_prefix_tokens``).
The first call of each session is always a cache WRITE (miss), so only the
N-1 consecutive pairs are scored.

Inputs (default dir ``api/data/telemetry/prompt/``):
  - ``assembly.jsonl``       — one record per ``PromptAssembler.assemble``.
      fields: ts, sections_included[], sections_dropped[], tokens_used,
              lang, provider, mode, version_hash, fallback_used,
              trace_id, request_id, session_id
  - ``llm_invocation.jsonl`` — one record per LLM adapter invoke.
      fields: ts, system_prompt_hash, system_prompt_bytes, tools_hash,
              lang, provider, trace_id, request_id, session_id

The per-session grouping key is ``session_id`` (the field B5.5 T1 wired so it
is non-null in production). Records whose ``session_id`` is null fall back to
``trace:<trace_id>`` grouping and are counted separately so a low-coverage
sample (e.g. pre-T1 local data) is visible rather than silently merged. A
record with neither id gets a unique singleton key so unrelated records never
collapse into one fake session.

Within a session, records are processed in APPEND (file) order — chronological
for an append-only JSONL — and are NOT re-sorted; ``ts`` is consulted only to
measure the TTL gap of a pair. An unparseable or non-monotonic ``ts`` fails the
TTL check (no hit) rather than reordering the sequence.

Decision thresholds live in the B5.5 acceptance runbook
(``docs/runbooks/b5.5-prompt-cache-acceptance.md``); this tool surfaces the raw
numbers and, when thresholds are supplied, prints a GO / NO-GO line.

Usage::

    uv run python scripts/analyze_prompt_cache_viability.py \
        [--telemetry-dir DIR] [--json] [--bytes-per-token N] \
        [--min-prefix-tokens N] [--cache-ttl-seconds N] \
        [--fallback-max F] [--hit-min H] [--min-sessions N]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

# Rough bytes→token ratio. system_prompt_bytes is UTF-8 BYTES; English ~4
# bytes/token, zh CJK ~5 bytes/token (a 3-byte char ≈ 0.6 token). Overridable
# via --bytes-per-token. Only affects the token ESTIMATE, never the hit RATE.
DEFAULT_BYTES_PER_TOKEN = 4.0
# Anthropic prompt caching has a minimum cacheable prefix (~1024 tokens for the
# smaller models). Prefixes below this can't be cached even if stable.
DEFAULT_MIN_PREFIX_TOKENS = 1024
# Anthropic prompt-cache TTL: a cached prefix expires ~5 minutes after its last
# use. Consecutive calls further apart than this re-pay the full input cost.
DEFAULT_CACHE_TTL_SECONDS = 300.0


def _read_jsonl(path: Path) -> tuple[list[dict], int]:
    """Return (records, malformed_line_count). Missing file → ([], 0)."""
    if not path.exists():
        return [], 0
    records: list[dict] = []
    malformed = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(obj, dict):
                records.append(obj)
            else:
                malformed += 1
    return records, malformed


def _session_key(rec: dict, idx: int) -> tuple[str, bool]:
    """(group_key, has_session_id). Prefer session_id; else trace_id; else a
    unique per-record key so id-less records never merge into one fake session.
    """
    sid = rec.get("session_id")
    if isinstance(sid, str) and sid:
        return f"sid:{sid}", True  # namespaced so a real id can't collide with a fallback key
    tid = rec.get("trace_id")
    if isinstance(tid, str) and tid:
        return f"trace:{tid}", False
    return f"_anon:{idx}", False


def _parse_ts(value: Any) -> float | None:
    """ISO-8601 → epoch seconds; None when absent/unparseable (so callers can
    refuse to score a TTL window they cannot verify)."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _percentiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0, "median": 0, "p90": 0, "max": 0}
    ordered = sorted(values)
    # nearest-rank p90 without numpy
    p90_idx = max(0, min(len(ordered) - 1, round(0.9 * (len(ordered) - 1))))
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90": ordered[p90_idx],
        "max": ordered[-1],
    }


@dataclass
class ViabilityReport:
    # sample sufficiency
    assembly_records: int = 0
    llm_records: int = 0
    malformed_lines: int = 0
    sessions: int = 0  # distinct group keys among llm records
    sessions_with_real_id: int = 0  # group keys that came from a real session_id
    llm_with_session_id: int = 0
    assembly_with_session_id: int = 0
    # modular-assembler health
    fallback_used: int = 0
    fallback_rate: float = 0.0
    # prefix stability (llm_invocation)
    total_prefix_pairs: int = 0  # consecutive call-pairs across all sessions
    prefix_hit_pairs: int = 0  # pairs that would be real cache hits (stable+TTL+eligible)
    prefix_hit_rate: float = 0.0
    sph_stable_pairs: int = 0  # raw: system_prompt_hash present & unchanged
    tools_stable_pairs: int = 0  # raw: tools_hash present & unchanged
    pairs_out_of_ttl: int = 0  # stable+eligible but beyond the cache window
    pairs_below_min_prefix: int = 0  # stable+in-ttl but prefix too small to cache
    # per-session shape
    avg_calls_per_session: float = 0.0
    avg_distinct_sph_per_session: float = 0.0
    avg_modal_sph_share: float = 0.0  # mean over sessions of (modal hash count / calls)
    # size / token estimate
    system_prompt_bytes: dict = field(default_factory=dict)
    bytes_per_token: float = DEFAULT_BYTES_PER_TOKEN
    min_prefix_tokens: int = DEFAULT_MIN_PREFIX_TOKENS
    cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS
    calls_above_min_prefix: int = 0
    est_total_prompt_tokens: int = 0
    est_cached_read_tokens: int = 0  # tokens that WOULD be cache-reads at the observed hit rate
    est_cache_savings_rate: float = 0.0
    notes: list = field(default_factory=list)


def analyze(
    assembly: Iterable[dict],
    llm: Iterable[dict],
    *,
    malformed: int = 0,
    bytes_per_token: float = DEFAULT_BYTES_PER_TOKEN,
    min_prefix_tokens: int = DEFAULT_MIN_PREFIX_TOKENS,
    cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
) -> ViabilityReport:
    if bytes_per_token <= 0:
        raise ValueError("bytes_per_token must be > 0")
    assembly = list(assembly)
    llm = list(llm)
    rep = ViabilityReport(
        assembly_records=len(assembly),
        llm_records=len(llm),
        malformed_lines=malformed,
        bytes_per_token=bytes_per_token,
        min_prefix_tokens=min_prefix_tokens,
        cache_ttl_seconds=cache_ttl_seconds,
    )

    # --- modular-assembler health (assembly.fallback_used) ----------------- #
    rep.fallback_used = sum(1 for r in assembly if r.get("fallback_used") is True)
    rep.assembly_with_session_id = sum(
        1 for i, r in enumerate(assembly) if _session_key(r, i)[1]
    )
    if assembly:
        rep.fallback_rate = rep.fallback_used / len(assembly)

    # --- group llm calls by session, ordered by ts ------------------------- #
    groups: dict[str, list[dict]] = defaultdict(list)
    group_is_real: dict[str, bool] = {}
    for i, r in enumerate(llm):
        key, has_id = _session_key(r, i)
        groups[key].append(r)
        group_is_real[key] = group_is_real.get(key, False) or has_id
        if has_id:
            rep.llm_with_session_id += 1
    rep.sessions = len(groups)
    rep.sessions_with_real_id = sum(1 for v in group_is_real.values() if v)

    bytes_values: list[int] = []
    calls_per_session: list[int] = []
    distinct_sph_per_session: list[int] = []
    modal_shares: list[float] = []

    def _eligible(rec: dict) -> bool:
        b = rec.get("system_prompt_bytes")
        return isinstance(b, int) and (b / bytes_per_token) >= min_prefix_tokens

    for key, recs in groups.items():
        # Records are kept in APPEND (file) order, which is chronological for an
        # append-only JSONL written by a single agent run. ts is used ONLY for
        # the TTL gap below, never to reorder: reordering around unparseable
        # timestamps could move a bad-ts call aside and fabricate adjacency
        # between two calls that were not actually consecutive. A non-monotonic
        # or unparseable ts simply fails the TTL check → no hit.
        n = len(recs)
        calls_per_session.append(n)
        # Exclude missing hashes from the modal/distinct diagnostics so an
        # all-missing session does not masquerade as one perfectly-stable hash.
        present_sph = [
            r.get("system_prompt_hash")
            for r in recs
            if r.get("system_prompt_hash") is not None
        ]
        if present_sph:
            distinct_sph_per_session.append(len(set(present_sph)))
            modal_count = Counter(present_sph).most_common(1)[0][1]
            modal_shares.append(modal_count / len(present_sph))
        for r in recs:
            b = r.get("system_prompt_bytes")
            if isinstance(b, int):
                bytes_values.append(b)
                rep.est_total_prompt_tokens += int(b / bytes_per_token)
                if (b / bytes_per_token) >= min_prefix_tokens:
                    rep.calls_above_min_prefix += 1
        # consecutive-pair scoring (a cache breakpoint after system+tools)
        for prev, cur in zip(recs, recs[1:]):
            rep.total_prefix_pairs += 1
            prev_sph, cur_sph = prev.get("system_prompt_hash"), cur.get("system_prompt_hash")
            prev_th, cur_th = prev.get("tools_hash"), cur.get("tools_hash")
            # both hashes must be PRESENT (non-None) and equal — a missing hash
            # must never read as "stable" via None == None.
            sph_same = prev_sph is not None and prev_sph == cur_sph
            tools_same = prev_th is not None and prev_th == cur_th
            if sph_same:
                rep.sph_stable_pairs += 1
            if tools_same:
                rep.tools_stable_pairs += 1
            if not (sph_same and tools_same):
                continue
            # eligibility: prefix big enough to cache at all
            if not _eligible(cur):
                rep.pairs_below_min_prefix += 1
                continue
            # TTL: only a hit if cur lands within the cache window of prev.
            prev_ts, cur_ts = _parse_ts(prev.get("ts")), _parse_ts(cur.get("ts"))
            if prev_ts is None or cur_ts is None:
                continue  # cannot verify the window → do not count a hit
            gap = cur_ts - prev_ts
            if not (0 <= gap <= cache_ttl_seconds):
                rep.pairs_out_of_ttl += 1
                continue
            rep.prefix_hit_pairs += 1
            cb = cur.get("system_prompt_bytes")
            if isinstance(cb, int):
                rep.est_cached_read_tokens += int(cb / bytes_per_token)

    if rep.total_prefix_pairs:
        rep.prefix_hit_rate = rep.prefix_hit_pairs / rep.total_prefix_pairs
    if calls_per_session:
        rep.avg_calls_per_session = statistics.mean(calls_per_session)
    if distinct_sph_per_session:
        rep.avg_distinct_sph_per_session = statistics.mean(distinct_sph_per_session)
    if modal_shares:
        rep.avg_modal_sph_share = statistics.mean(modal_shares)
    rep.system_prompt_bytes = _percentiles(bytes_values)
    if rep.est_total_prompt_tokens:
        rep.est_cache_savings_rate = (
            rep.est_cached_read_tokens / rep.est_total_prompt_tokens
        )

    # --- coverage / honesty notes ------------------------------------------ #
    if rep.llm_records and rep.llm_with_session_id == 0:
        rep.notes.append(
            "0% of llm_invocation records carry a session_id — grouping fell "
            "back to trace_id / per-record. Likely PRE-T1 data; per-session "
            "viability is NOT measurable until T1-bound sessions are captured."
        )
    elif rep.llm_records and rep.llm_with_session_id < rep.llm_records:
        rep.notes.append(
            f"{rep.llm_with_session_id}/{rep.llm_records} llm records carry a "
            "session_id; the rest fell back to trace_id / per-record grouping."
        )
    if rep.system_prompt_bytes.get("median", 0):
        rep.notes.append(
            "Token counts are ESTIMATES (system_prompt_bytes / bytes_per_token); "
            "tools-prefix tokens are NOT in telemetry, so savings is "
            "system-prompt-only (conservative)."
        )
    return rep


def evaluate_gate(
    rep: ViabilityReport,
    *,
    fallback_max: float | None,
    hit_min: float | None,
    min_sessions: int | None,
) -> tuple[str, list[str]]:
    """Return (verdict, reasons). verdict ∈ {GO, NO-GO, INSUFFICIENT-DATA}."""
    reasons: list[str] = []
    insufficient = False
    if min_sessions is not None and rep.sessions_with_real_id < min_sessions:
        insufficient = True
        reasons.append(
            f"sessions_with_real_id={rep.sessions_with_real_id} < "
            f"min_sessions={min_sessions}"
        )
    # Require enough measured pairs, not just one — a single stable pair must
    # not be allowed to drive a GO. Tie the floor to min_sessions when given.
    min_pairs = max(2, min_sessions) if min_sessions is not None else 2
    if rep.total_prefix_pairs < min_pairs:
        insufficient = True
        reasons.append(
            f"total_prefix_pairs={rep.total_prefix_pairs} < min_pairs={min_pairs} "
            "(too few measured pairs)"
        )
    if insufficient:
        return "INSUFFICIENT-DATA", reasons

    go = True
    if fallback_max is not None:
        ok = rep.fallback_rate <= fallback_max
        reasons.append(
            f"fallback_rate={rep.fallback_rate:.3f} {'<=' if ok else '>'} "
            f"max={fallback_max}"
        )
        go = go and ok
    if hit_min is not None:
        ok = rep.prefix_hit_rate >= hit_min
        reasons.append(
            f"prefix_hit_rate={rep.prefix_hit_rate:.3f} {'>=' if ok else '<'} "
            f"min={hit_min}"
        )
        go = go and ok
    reasons.append(f"(evidence: {rep.total_prefix_pairs} pairs across "
                   f"{rep.sessions_with_real_id} real sessions)")
    return ("GO" if go else "NO-GO"), reasons


def format_human(rep: ViabilityReport, gate: tuple[str, list[str]] | None) -> str:
    b = rep.system_prompt_bytes
    sph = (rep.sph_stable_pairs / rep.total_prefix_pairs) if rep.total_prefix_pairs else 0.0
    tools = (rep.tools_stable_pairs / rep.total_prefix_pairs) if rep.total_prefix_pairs else 0.0
    lines = [
        "=== B5.5 prompt-cache viability ===",
        "",
        "[sample]",
        f"  assembly records      : {rep.assembly_records}",
        f"  llm_invocation records: {rep.llm_records}",
        f"  malformed lines       : {rep.malformed_lines}",
        f"  sessions (groups)     : {rep.sessions} "
        f"({rep.sessions_with_real_id} from real session_id)",
        f"  llm w/ session_id     : {rep.llm_with_session_id}/{rep.llm_records}",
        f"  avg calls / session   : {rep.avg_calls_per_session:.2f}",
        "",
        "[modular assembler health]",
        f"  fallback_used         : {rep.fallback_used}/{rep.assembly_records} "
        f"(rate {rep.fallback_rate:.3f})",
        "",
        "[prefix stability — cache-hit potential]",
        f"  consecutive pairs     : {rep.total_prefix_pairs}",
        f"  prefix HIT rate       : {rep.prefix_hit_rate:.3f} "
        f"(stable + within {rep.cache_ttl_seconds:.0f}s TTL + >= "
        f"{rep.min_prefix_tokens} tok)",
        f"    raw sph stability   : {sph:.3f}",
        f"    raw tools stability : {tools:.3f}",
        f"    excluded — out of TTL  : {rep.pairs_out_of_ttl}",
        f"    excluded — below min   : {rep.pairs_below_min_prefix}",
        f"  avg distinct sph/sess : {rep.avg_distinct_sph_per_session:.2f}",
        f"  avg modal sph share   : {rep.avg_modal_sph_share:.3f}",
        "",
        "[size / token estimate]",
        f"  system_prompt_bytes   : min={b.get('min')} median={b.get('median')} "
        f"p90={b.get('p90')} max={b.get('max')}",
        f"  bytes/token           : {rep.bytes_per_token}",
        f"  calls >= {rep.min_prefix_tokens} tok prefix: {rep.calls_above_min_prefix}/{rep.llm_records}",
        f"  est total prompt tok  : {rep.est_total_prompt_tokens}",
        f"  est cache-read tok    : {rep.est_cached_read_tokens} "
        f"(savings rate {rep.est_cache_savings_rate:.3f})",
    ]
    if rep.notes:
        lines += ["", "[notes]"]
        lines += [f"  - {n}" for n in rep.notes]
    if gate is not None:
        verdict, reasons = gate
        lines += ["", f"[verdict] {verdict}"]
        lines += [f"  - {r}" for r in reasons]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--telemetry-dir",
        default="api/data/telemetry/prompt",
        help="dir containing assembly.jsonl + llm_invocation.jsonl "
        "(default: api/data/telemetry/prompt)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--bytes-per-token", type=float, default=DEFAULT_BYTES_PER_TOKEN)
    parser.add_argument("--min-prefix-tokens", type=int, default=DEFAULT_MIN_PREFIX_TOKENS)
    parser.add_argument("--cache-ttl-seconds", type=float, default=DEFAULT_CACHE_TTL_SECONDS)
    parser.add_argument("--fallback-max", type=float, default=None,
                        help="GO requires fallback_rate <= this")
    parser.add_argument("--hit-min", type=float, default=None,
                        help="GO requires prefix_hit_rate >= this")
    parser.add_argument("--min-sessions", type=int, default=None,
                        help="below this many real-session_id sessions → INSUFFICIENT-DATA")
    args = parser.parse_args(argv)

    if args.bytes_per_token <= 0:
        sys.stderr.write("--bytes-per-token must be > 0\n")
        return 2

    base = Path(args.telemetry_dir)
    assembly, m1 = _read_jsonl(base / "assembly.jsonl")
    llm, m2 = _read_jsonl(base / "llm_invocation.jsonl")
    if not assembly and not llm:
        sys.stderr.write(
            f"no telemetry found under {base} "
            "(expected assembly.jsonl / llm_invocation.jsonl)\n"
        )
        return 2

    rep = analyze(
        assembly,
        llm,
        malformed=m1 + m2,
        bytes_per_token=args.bytes_per_token,
        min_prefix_tokens=args.min_prefix_tokens,
        cache_ttl_seconds=args.cache_ttl_seconds,
    )
    gate = None
    if any(v is not None for v in (args.fallback_max, args.hit_min, args.min_sessions)):
        gate = evaluate_gate(
            rep,
            fallback_max=args.fallback_max,
            hit_min=args.hit_min,
            min_sessions=args.min_sessions,
        )

    if args.json:
        payload = asdict(rep)
        if gate is not None:
            payload["verdict"] = gate[0]
            payload["verdict_reasons"] = gate[1]
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(format_human(rep, gate) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
