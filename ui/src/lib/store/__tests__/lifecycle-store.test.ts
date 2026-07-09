// ui/src/lib/store/__tests__/lifecycle-store.test.ts
// C7 PR6 — dispatcher/store 集成：flag 门控、五入口不泄漏、reducer 接入。
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  dispatchLifecycleWireEvent,
  isLifecycleWireEventType,
  normalizeAndRouteSessionEvents,
  routeLifecycleRecords,
} from "@/lib/lifecycle/dispatch";
import { useLifecycleStore } from "@/lib/store/lifecycle-store";
import type { SessionEventRecord } from "@/lib/event-normalize";

function wire(eventType: string, unitId = "u1", seq = 1, epoch = 0) {
  const [, lifecycleType, kind] = eventType.split(".");
  return {
    type: "lifecycle",
    lifecycle_type: lifecycleType,
    event: kind,
    state: kind === "completed" ? "completed" : kind === "started" ? "running" : "running",
    unit_id: unitId,
    epoch,
    seq,
    event_id: `evt-${seq}`,
  };
}

describe("dispatchLifecycleWireEvent", () => {
  beforeEach(() => {
    vi.stubEnv("NEXT_PUBLIC_LIFECYCLE_UI_ENABLED", "true");
    useLifecycleStore.getState().reset();
  });
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("consumes lifecycle.* and populates the store", () => {
    const consumed = dispatchLifecycleWireEvent("lifecycle.step.started", wire("lifecycle.step.started", "s1"));
    expect(consumed).toBe(true);
    expect(useLifecycleStore.getState().units["lifecycle:step:s1"]).toMatchObject({
      state: "running", lastEvent: "started",
    });
  });

  it("flag-off: still consumed (intercepted) but store untouched", () => {
    vi.stubEnv("NEXT_PUBLIC_LIFECYCLE_UI_ENABLED", "false");
    const consumed = dispatchLifecycleWireEvent("lifecycle.step.started", wire("lifecycle.step.started", "s2"));
    expect(consumed).toBe(true);   // 拦截出旧管线（零行为差异指旧时间线，不是漏进去）
    expect(useLifecycleStore.getState().units).toEqual({});
  });

  it("non-lifecycle events are not consumed", () => {
    expect(dispatchLifecycleWireEvent("message", { message: "hi" })).toBe(false);
    expect(isLifecycleWireEventType("tool")).toBe(false);
  });

  it("dotted-name/payload mismatch is dropped with warning (contract)", () => {
    const bad = { ...wire("lifecycle.step.started", "s3"), lifecycle_type: "tool" };
    dispatchLifecycleWireEvent("lifecycle.step.started", bad);
    expect(useLifecycleStore.getState().units).toEqual({});
  });
});

describe("normalizeAndRouteSessionEvents — 不泄漏进旧时间线（§7 contract）", () => {
  beforeEach(() => {
    vi.stubEnv("NEXT_PUBLIC_LIFECYCLE_UI_ENABLED", "true");
    useLifecycleStore.getState().reset();
  });
  afterEach(() => vi.unstubAllEnvs());

  it("routes lifecycle records to the store and strips them from timeline", () => {
    const events: SessionEventRecord[] = [
      { event: "message", data: { event_id: "m1", message: "hello" } },
      { event: "lifecycle.tool.started", data: wire("lifecycle.tool.started", "tc1", 2) },
      { event: "step", data: { event_id: "s1", id: "step-1", status: "running", description: "" } },
      { event: "lifecycle.tool.completed", data: wire("lifecycle.tool.completed", "tc1", 3) },
    ];
    const rest = normalizeAndRouteSessionEvents(events);
    expect(rest.every((e) => !e.event.startsWith("lifecycle."))).toBe(true);
    expect(rest.map((e) => e.event)).toEqual(["message", "step"]);
    expect(useLifecycleStore.getState().units["lifecycle:tool:tc1"]).toMatchObject({
      state: "completed", sticky: true,
    });
  });

  it("flag-off: lifecycle records are dropped, timeline unchanged", () => {
    vi.stubEnv("NEXT_PUBLIC_LIFECYCLE_UI_ENABLED", "false");
    const rest = routeLifecycleRecords([
      { event: "lifecycle.plan.started", data: wire("lifecycle.plan.started", "p1") },
      { event: "done", data: { event_id: "d1" } },
    ]);
    expect(rest.map((e) => e.event)).toEqual(["done"]);
    expect(useLifecycleStore.getState().units).toEqual({});
  });
});

describe("入口接线 source pin（五入口全部走分流；漏一个=泄漏）", () => {
  it("session-store.ts uses normalizeAndRouteSessionEvents at every normalize call site", async () => {
    const fs = await import("node:fs");
    const src = fs.readFileSync("src/lib/store/session-store.ts", "utf-8");
    // 允许 import 行出现 normalizeSessionEvents；调用点必须全部是 AndRoute 版本
    const bareCalls = src.match(/(?<!AndRoute)normalizeSessionEvents\(/g) ?? [];
    expect(bareCalls).toHaveLength(0);
    const routedCalls = src.match(/normalizeAndRouteSessionEvents\(/g) ?? [];
    expect(routedCalls.length).toBeGreaterThanOrEqual(5);
  });

  it("session.ts intercepts before the SSEEventData cast", async () => {
    const fs = await import("node:fs");
    const src = fs.readFileSync("src/lib/api/session.ts", "utf-8");
    const dispatchIdx = src.indexOf("dispatchLifecycleWireEvent(");
    const castIdx = src.indexOf('as SSEEventData["type"]');
    expect(dispatchIdx).toBeGreaterThan(-1);
    expect(castIdx).toBeGreaterThan(-1);
    expect(dispatchIdx).toBeLessThan(castIdx);
  });
});
