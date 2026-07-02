import { describe, expect, it } from "vitest";

import { deriveRole } from "@/lib/agent-tree";

describe("deriveRole", () => {
  it("maps root worker_type to root regardless of preset", () => {
    expect(deriveRole("root", null)).toBe("root");
    expect(deriveRole("root", "coordinator_step")).toBe("root");
  });

  it("maps subagent presets to their role", () => {
    expect(deriveRole("subagent", "coordinator_step")).toBe("coordinator_child");
    expect(deriveRole("subagent", "subagent_research")).toBe("research_child");
    expect(deriveRole("subagent", "something_else")).toBe("subagent");
    expect(deriveRole("subagent", null)).toBe("subagent");
  });
});

import { buildTree, flattenTree } from "@/lib/agent-tree";
import type { ChildSessionItem } from "@/lib/api/types";

function child(overrides: Partial<ChildSessionItem> & { id: string }): ChildSessionItem {
  return {
    id: overrides.id,
    parent_session_id: overrides.parent_session_id ?? "root",
    worker_type: overrides.worker_type ?? "subagent",
    tool_filter_preset: overrides.tool_filter_preset ?? "subagent_research",
    status: overrides.status ?? "running",
    title: overrides.title ?? null,
    created_at: overrides.created_at ?? null,
    updated_at: overrides.updated_at ?? null,
  };
}

const root = {
  sessionId: "root",
  status: "running" as const,
  title: "主会话",
  createdAt: null,
  updatedAt: null,
};

describe("buildTree", () => {
  it("nests descendants under their parents and returns a root node", () => {
    const tree = buildTree(root, [
      child({ id: "a", parent_session_id: "root" }),
      child({ id: "b", parent_session_id: "a" }),
    ]);
    expect(tree.sessionId).toBe("root");
    expect(tree.role).toBe("root");
    expect(tree.children.map((c) => c.sessionId)).toEqual(["a"]);
    expect(tree.children[0].children.map((c) => c.sessionId)).toEqual(["b"]);
  });

  it("attaches orphans (absent parent) under the root", () => {
    const tree = buildTree(root, [child({ id: "x", parent_session_id: "ghost" })]);
    expect(tree.children.map((c) => c.sessionId)).toEqual(["x"]);
  });

  it("guards against a self-parenting row (attaches under root, no cycle)", () => {
    const tree = buildTree(root, [child({ id: "self", parent_session_id: "self" })]);
    expect(tree.children.map((c) => c.sessionId)).toEqual(["self"]);
    expect(tree.children[0].children).toEqual([]);
  });

  it("never duplicates the root and yields root-only for empty descendants", () => {
    const tree = buildTree(root, [child({ id: "root", parent_session_id: "root" })]);
    expect(tree.children).toEqual([]);
    expect(buildTree(root, []).children).toEqual([]);
  });
});

describe("flattenTree", () => {
  it("indexes every node by sessionId", () => {
    const tree = buildTree(root, [
      child({ id: "a", parent_session_id: "root" }),
      child({ id: "b", parent_session_id: "a" }),
    ]);
    expect(Object.keys(flattenTree(tree)).sort()).toEqual(["a", "b", "root"]);
  });
});

import { deriveElapsed, formatElapsed } from "@/lib/agent-tree";

describe("deriveElapsed", () => {
  const now = Date.parse("2026-07-01T00:00:10.000Z");
  it("running-ish → now − created", () => {
    expect(
      deriveElapsed("2026-07-01T00:00:00.000Z", null, "running", now),
    ).toBe(10_000);
    expect(
      deriveElapsed("2026-07-01T00:00:00.000Z", null, "finishing", now),
    ).toBe(10_000);
  });
  it("terminal → updated − created", () => {
    expect(
      deriveElapsed(
        "2026-07-01T00:00:00.000Z",
        "2026-07-01T00:00:05.000Z",
        "completed",
        now,
      ),
    ).toBe(5_000);
    expect(
      deriveElapsed(
        "2026-07-01T00:00:00.000Z",
        "2026-07-01T00:00:03.000Z",
        "timed_out",
        now,
      ),
    ).toBe(3_000);
  });
  it("null / unparseable createdAt → null (never throws)", () => {
    expect(deriveElapsed(null, null, "running", now)).toBeNull();
    expect(deriveElapsed("not-a-date", null, "running", now)).toBeNull();
    expect(deriveElapsed("2026-07-01T00:00:00.000Z", null, "completed", now)).toBeNull();
  });
});

describe("formatElapsed", () => {
  it("formats null as em dash and durations compactly", () => {
    expect(formatElapsed(null)).toBe("—");
    expect(formatElapsed(3_000)).toBe("3s");
    expect(formatElapsed(65_000)).toBe("1m 5s");
    expect(formatElapsed(3_661_000)).toBe("1h 1m");
  });
});

import {
  assignAgentColors,
  countToolCalls,
  eventSeq,
  eventSortKey,
  mergeAgentTimelines,
  type AgentEventBundle,
} from "@/lib/agent-tree";
import type { SessionEventRecord } from "@/lib/event-normalize";

const rec = (event: string, data: Record<string, unknown>): SessionEventRecord => ({ event, data });

describe("eventSeq / eventSortKey", () => {
  it("eventSeq reads a numeric data.seq, else null", () => {
    expect(eventSeq(rec("message", { seq: 4 }))).toBe(4);
    expect(eventSeq(rec("message", {}))).toBeNull();
    expect(eventSeq(rec("message", { seq: "4" }))).toBeNull();
  });
  it("eventSortKey reads numeric data.created_at, else +Infinity", () => {
    expect(eventSortKey(rec("message", { created_at: 100 }))).toBe(100);
    expect(eventSortKey(rec("message", {}))).toBe(Number.POSITIVE_INFINITY);
  });
  it("eventSortKey parses ISO-string created_at into epoch SECONDS (unit-aligned with the numeric branch)", () => {
    // Synthetic compaction events carry ISO-string created_at (session-store.ts ~1678);
    // real session events carry epoch-SECONDS numbers. They must interleave chronologically.
    const isoInstant = "2026-05-03T00:00:00.000Z";
    const isoKey = Date.parse(isoInstant) / 1000; // ISO → epoch seconds
    const numericSeconds = Date.parse("2026-05-03T00:00:10.000Z") / 1000; // 10s LATER, in seconds
    const isoRec = rec("compaction", { created_at: isoInstant });
    const numericRec = rec("message", { created_at: numericSeconds });

    // Both keys are finite (NOT +Infinity) and the earlier ISO event sorts before the later numeric one.
    expect(eventSortKey(isoRec)).toBe(isoKey);
    expect(eventSortKey(numericRec)).toBe(numericSeconds);
    expect(Number.isFinite(eventSortKey(isoRec))).toBe(true);
    expect(Number.isFinite(eventSortKey(numericRec))).toBe(true);
    expect(eventSortKey(isoRec)).toBeLessThan(eventSortKey(numericRec));

    // mergeAgentTimelines must place the earlier ISO event BEFORE the later numeric event,
    // and an event with NO created_at still sorts LAST (+Infinity preserved).
    const noTsRec = rec("message", {});
    const merged = mergeAgentTimelines([
      {
        sessionId: "root",
        role: "root",
        color: "#000",
        events: [numericRec, isoRec, noTsRec],
      },
    ]);
    expect(merged.map((m) => m.event)).toEqual([isoRec, numericRec, noTsRec]);
    expect(eventSortKey(merged[merged.length - 1].event)).toBe(Number.POSITIVE_INFINITY);
  });
});

describe("countToolCalls", () => {
  it("counts distinct tool_call_id among tool events only", () => {
    expect(
      countToolCalls([
        rec("tool", { tool_call_id: "t1" }),
        rec("tool", { tool_call_id: "t1" }), // repeat update — not double-counted
        rec("tool", { tool_call_id: "t2" }),
        rec("message", { tool_call_id: "t3" }), // non-tool — ignored
      ]),
    ).toBe(2);
  });
});

describe("assignAgentColors", () => {
  it("assigns a stable color per id and cycles the palette", () => {
    const c = assignAgentColors(["a", "b"]);
    expect(c.a).toBeTruthy();
    expect(c.b).toBeTruthy();
    expect(c.a).not.toBe(c.b);
    expect(assignAgentColors(["a", "b"])).toEqual(c); // stable
  });
});

describe("mergeAgentTimelines", () => {
  const bundle = (sessionId: string, events: SessionEventRecord[]): AgentEventBundle => ({
    sessionId,
    role: "subagent",
    color: "#000",
    events,
  });
  it("orders cross-session by created_at, tiebreaks within a session by seq", () => {
    const merged = mergeAgentTimelines([
      bundle("A", [rec("message", { created_at: 10, seq: 1 }), rec("tool", { created_at: 10, seq: 2 })]),
      bundle("B", [rec("message", { created_at: 5, seq: 1 })]),
    ]);
    expect(merged.map((m) => m.sourceSessionId)).toEqual(["B", "A", "A"]);
  });
  it("sorts events with a missing created_at to the end", () => {
    const merged = mergeAgentTimelines([
      bundle("A", [rec("message", {}), rec("tool", { created_at: 1 })]),
    ]);
    expect(eventSortKey(merged[0].event)).toBe(1);
    expect(eventSortKey(merged[1].event)).toBe(Number.POSITIVE_INFINITY);
  });
});
