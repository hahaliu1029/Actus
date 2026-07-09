// ui/src/lib/lifecycle/__tests__/wire-contract.test.ts
// C7 PR6 — FE↔BE 契约：点分名/payload/子集与后端 test_lifecycle_sse.py 同构断言。
import { describe, expect, it } from "vitest";

import { SUPPORTED_EVENTS, type LifecycleEventKind, type LifecycleType } from "../types";

// 后端 wire 样例的字面复刻（api/tests/interfaces/schemas/test_lifecycle_sse.py 同源；
// 若后端改 shape，两边 fixture 同步改——这就是「双写即契约」的意图）
const SAMPLE = {
  event: "lifecycle.task.retried",
  data: {
    type: "lifecycle",
    lifecycle_type: "task",
    event: "retried",
    state: "running",
    unit_id: "sess-1",
    epoch: 1,
    seq: 99,
    reason: "retry_from_suspend",
    detail: {
      trigger: "user", previous_state: "suspended",
      retry_budget_remaining: 2, original_outcome: null, note: null,
    },
    parent_unit_id: null,
    correlation: null,
    source_event_type: null, source_event_id: null, source_seq: null,
  },
};

describe("wire contract", () => {
  it("dotted name segments equal payload discriminators", () => {
    const [ns, t, k] = SAMPLE.event.split(".");
    expect(ns).toBe("lifecycle");
    expect(t).toBe(SAMPLE.data.lifecycle_type);
    expect(k).toBe(SAMPLE.data.event);
  });

  it("every supported pair yields a well-formed dotted name", () => {
    const names = new Set<string>();
    (Object.keys(SUPPORTED_EVENTS) as LifecycleType[]).forEach((t) => {
      SUPPORTED_EVENTS[t].forEach((k: LifecycleEventKind) => {
        const name = `lifecycle.${t}.${k}`;
        expect(name.split(".")).toHaveLength(3);
        names.add(name);
      });
    });
    expect(names.size).toBe(21); // 与后端 SUPPORTED_EVENTS 总数同构
  });

  it("detail whitelist mirrors backend LifecycleDetailV1 (no attempt field — R12#P2)", () => {
    expect(Object.keys(SAMPLE.data.detail)).toEqual(
      expect.arrayContaining(["trigger", "previous_state", "retry_budget_remaining", "original_outcome", "note"]),
    );
    expect(SAMPLE.data.detail).not.toHaveProperty("attempt");
  });
});
