// ui/src/lib/lifecycle/__tests__/reducer.test.ts
// C7 §5 规则 0-5 穷举（含 R1#2 反例、错过-retried 自愈、INV-C7-8 四象限）。
import { describe, expect, it } from "vitest";

import { reduceLifecycleEvent } from "../reducer";
import {
  SUPPORTED_EVENTS,
  type LifecycleUnitState,
  type LifecycleWireData,
} from "../types";

function ev(partial: Partial<LifecycleWireData> & { seq: number | null }): LifecycleWireData {
  return {
    type: "lifecycle",
    lifecycle_type: "task",
    event: "started",
    state: "running",
    unit_id: "u1",
    epoch: 0,
    ...partial,
  } as LifecycleWireData;
}

function applied(prev: LifecycleUnitState | undefined, e: LifecycleWireData): LifecycleUnitState {
  const out = reduceLifecycleEvent(prev, e);
  if (out.kind === "dropped") throw new Error(`unexpected drop: ${out.why}`);
  return out.next;
}

describe("parity with backend vocabulary", () => {
  it("mirrors 21 supported pairs", () => {
    const total = Object.values(SUPPORTED_EVENTS).reduce((n, ks) => n + ks.length, 0);
    expect(total).toBe(21);
    expect(SUPPORTED_EVENTS.plan).not.toContain("failed");     // R5#P1b
    expect(SUPPORTED_EVENTS.step).not.toContain("cancelled");  // §4.2 诚实缺口
    expect(SUPPORTED_EVENTS.subagent).not.toContain("progress");
    expect(SUPPORTED_EVENTS.task).toContain("retried");
  });
});

describe("rule 0 — first-seen init", () => {
  it("initializes and applies state including sticky (R9#P3a)", () => {
    const s = applied(undefined, ev({ event: "completed", state: "completed", seq: 5 }));
    expect(s).toMatchObject({ epoch: 0, lastSeq: 5, state: "completed", sticky: true });
  });

  it("task first-seen with high epoch is adopted (reconnect scenario)", () => {
    const s = applied(undefined, ev({ epoch: 2, seq: 9 }));
    expect(s.epoch).toBe(2);
  });

  it("non-task first-seen with epoch != 0 is dropped, not initialized (R10#B1)", () => {
    const outPos = reduceLifecycleEvent(
      undefined,
      ev({ lifecycle_type: "tool", event: "started", state: "pending", epoch: 1, seq: 1 }),
    );
    expect(outPos).toMatchObject({ kind: "dropped", why: "nontask_epoch" });
    const outNeg = reduceLifecycleEvent(
      undefined,
      ev({ lifecycle_type: "tool", event: "started", state: "pending", epoch: -1, seq: 1 }),
    );
    expect(outNeg).toMatchObject({ kind: "dropped", why: "nontask_epoch" });
  });
});

describe("rule 1 — stale epoch dropped unconditionally", () => {
  it("R1#2 counterexample: late terminal from old attempt cannot override new epoch", () => {
    let s = applied(undefined, ev({ seq: 1 }));                       // started e0
    s = applied(s, ev({ event: "retried", epoch: 1, seq: 10, reason: "retry_from_suspend" })); // reopen e1
    const late = reduceLifecycleEvent(s, ev({ event: "completed", state: "completed", epoch: 0, seq: 5 }));
    expect(late).toMatchObject({ kind: "dropped", why: "stale_epoch" });
    expect(s.state).toBe("running");
  });
});

describe("rule 2 — implicit reopen (task only)", () => {
  it("missed retried self-heals: any higher-epoch task event reopens sticky terminal", () => {
    let s = applied(undefined, ev({ seq: 1 }));
    s = applied(s, ev({ event: "failed", state: "failed", seq: 2, reason: "watchdog_timeout" }));
    expect(s.sticky).toBe(true);
    // 错过 retried(e1)，直接收到 e1 的 progress——自愈重开（0→1 甚至 0→2 跳变）
    s = applied(s, ev({ event: "progress", epoch: 1, seq: 20, reason: "finishing" }));
    expect(s).toMatchObject({ epoch: 1, sticky: false, state: "running" });
  });

  it("non-task epoch mismatch after init is contract violation → drop (INV-C7-8)", () => {
    const s = applied(undefined, ev({ lifecycle_type: "tool", event: "started", state: "pending", epoch: 0, seq: 1 }));
    const out = reduceLifecycleEvent(
      s, ev({ lifecycle_type: "tool", event: "completed", state: "completed", epoch: 1, seq: 2 }),
    );
    expect(out).toMatchObject({ kind: "dropped", why: "nontask_epoch" });
  });
});

describe("rule 3 — seq monotonic guard (same epoch)", () => {
  it("drops seq <= lastSeq", () => {
    const s = applied(undefined, ev({ seq: 5 }));
    expect(reduceLifecycleEvent(s, ev({ event: "progress", seq: 5 }))).toMatchObject({
      kind: "dropped", why: "stale_seq",
    });
    expect(reduceLifecycleEvent(s, ev({ event: "progress", seq: 4 }))).toMatchObject({
      kind: "dropped", why: "stale_seq",
    });
  });
});

describe("rule 4 — sticky no-op still advances lastSeq (R8#P3c)", () => {
  it("duplicate terminal after sticky is no-op but cursor moves", () => {
    let s = applied(undefined, ev({ event: "completed", state: "completed", seq: 3 }));
    const out = reduceLifecycleEvent(s, ev({ event: "completed", state: "completed", seq: 7 }));
    expect(out.kind).toBe("noop");
    if (out.kind === "noop") s = out.next;
    expect(s).toMatchObject({ state: "completed", sticky: true, lastSeq: 7 });
  });

  it("INV-C7-5: duplicate terminal sequence converges idempotently (double-source)", () => {
    let s = applied(undefined, ev({
      lifecycle_type: "subagent", event: "started", state: "running", unit_id: "child-b", seq: 1,
    }));
    s = applied(s, ev({
      lifecycle_type: "subagent", event: "cancelled", state: "cancelled", unit_id: "child-b",
      seq: 2, reason: "sibling_cancel",
    }));
    const dup = reduceLifecycleEvent(s, ev({
      lifecycle_type: "subagent", event: "cancelled", state: "cancelled", unit_id: "child-b", seq: 9,
    }));
    expect(dup.kind).toBe("noop");
  });
});

describe("rule 5 — retried explicit reopen + started idempotence", () => {
  it("retried carries epoch=current+1 and lands on rule 2", () => {
    let s = applied(undefined, ev({ event: "completed", state: "completed", seq: 2 }));
    s = applied(s, ev({ event: "retried", epoch: 1, seq: 8, reason: "retry_from_suspend" }));
    expect(s).toMatchObject({ epoch: 1, state: "running", sticky: false, lastEvent: "retried" });
  });

  it("duplicate started(e0) on running unit is harmless (R8#P3d — dropped by seq or applied as running)", () => {
    const s = applied(undefined, ev({ seq: 1 }));
    const out = reduceLifecycleEvent(s, ev({ seq: 2 }));  // 又一个 started(e0)
    if (out.kind === "dropped") throw new Error("should not drop");
    expect(out.next.state).toBe("running");
  });
});

describe("per-type subset validation", () => {
  it("drops unsupported pairs (e.g. plan.failed / subagent.progress)", () => {
    expect(reduceLifecycleEvent(undefined, ev({
      lifecycle_type: "plan", event: "failed", state: "failed", seq: 1,
    }))).toMatchObject({ kind: "dropped", why: "unsupported_pair" });
    expect(reduceLifecycleEvent(undefined, ev({
      lifecycle_type: "subagent", event: "progress", state: "running", seq: 1,
    }))).toMatchObject({ kind: "dropped", why: "unsupported_pair" });
  });
});
