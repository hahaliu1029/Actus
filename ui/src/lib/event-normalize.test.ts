import { describe, expect, it } from "vitest";

import { normalizeSessionEvents, type SessionEventRecord } from "@/lib/event-normalize";

const ev = (event: string, data: Record<string, unknown>): SessionEventRecord => ({ event, data });

describe("normalizeSessionEvents (moved verbatim — full pipeline)", () => {
  it("dedups messages by stream_id (last wins) and skips title events", () => {
    const out = normalizeSessionEvents([
      ev("title", { title: "x" }),
      ev("message", { stream_id: "s", content: "a", event_id: "e1" }),
      ev("message", { stream_id: "s", content: "ab", event_id: "e2" }),
    ]);
    expect(out).toHaveLength(1);
    expect(out[0].data.content).toBe("ab");
  });

  it("dedups tools by tool_call_id (last wins)", () => {
    const out = normalizeSessionEvents([
      ev("tool", { tool_call_id: "t1", status: "running", event_id: "e1" }),
      ev("tool", { tool_call_id: "t1", status: "done", event_id: "e2" }),
    ]);
    expect(out).toHaveLength(1);
    expect(out[0].data.status).toBe("done");
  });

  it("syncs a plan step's status from a later step event", () => {
    const out = normalizeSessionEvents([
      ev("plan", { steps: [{ id: "s1", status: "pending", description: "do" }] }),
      ev("step", { id: "s1", status: "completed" }),
    ]);
    const plan = out.find((e) => e.event === "plan");
    const steps = plan?.data.steps as Array<{ id: string; status: string }>;
    expect(steps[0].status).toBe("completed");
  });

  it("prunes a recovered LLM error followed by an assistant message", () => {
    const out = normalizeSessionEvents([
      ev("error", { error: "调用语言模型失败: timeout", event_id: "e1" }),
      ev("message", { stream_id: "s", role: "assistant", content: "recovered", event_id: "e2" }),
    ]);
    expect(out.some((e) => e.event === "error")).toBe(false);
    expect(out).toHaveLength(1);
  });
});
