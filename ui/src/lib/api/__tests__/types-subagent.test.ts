import { describe, it, expect } from "vitest";
import type {
  ChildStartedEvent,
  ChildDoneEvent,
  JoinedSummaryEvent,
  ChildOutcome,
  ResearchSubagentRequest,
  ListSessionItem,
  SubagentEvent,
} from "@/lib/api/types";

describe("subagent types", () => {
  it("ChildStartedEvent has type field 'child_started'", () => {
    const ev: ChildStartedEvent = {
      id: "ev-1",
      type: "child_started",
      probe_run_id: "p-1",
      child_session_id: "c-1",
      prompt: "test",
    };
    expect(ev.type).toBe("child_started");
  });

  it("ChildDoneEvent outcome is one of allowed enum values", () => {
    const ev: ChildDoneEvent = {
      id: "ev-2",
      type: "child_done",
      probe_run_id: "p-1",
      child_session_id: "c-1",
      outcome: "completed",
      final_answer: "x",
      transcript_tokens: 100,
      error_summary: null,
    };
    const allowed: ChildOutcome[] = [
      "completed",
      "failed",
      "timed_out",
      "waiting",
      "cancelled",
    ];
    expect(allowed).toContain(ev.outcome);
  });

  it("JoinedSummaryEvent has summary + validation_warnings", () => {
    const ev: JoinedSummaryEvent = {
      id: "ev-3",
      type: "joined_summary",
      probe_run_id: "p-1",
      summary: "integrated",
      summary_tokens: 42,
      completed_children: ["c-1"],
      dropped_children: [],
      metrics: { total_tokens: 42 },
      validation_warnings: [],
    };
    expect(ev.summary).toBe("integrated");
    expect(ev.validation_warnings).toEqual([]);
  });

  it("SubagentEvent union narrows via type discriminator", () => {
    const events: SubagentEvent[] = [
      {
        id: "1",
        type: "child_started",
        probe_run_id: "p",
        child_session_id: "c",
        prompt: "x",
      },
    ];
    const first = events[0];
    if (first.type === "child_started") {
      expect(first.prompt).toBe("x");
    }
  });

  it("ResearchSubagentRequest has prompts + max_children", () => {
    const req: ResearchSubagentRequest = {
      prompts: ["a", "b"],
      max_children: 2,
    };
    expect(req.prompts).toHaveLength(2);
    expect(req.max_children).toBe(2);
  });

  it("ListSessionItem has sample_session_id field (nullable)", () => {
    const item: ListSessionItem = {
      session_id: "s-1",
      title: "test",
      sample_session_id: null,
      parent_session_id: null,
      worker_type: "root",
      latest_message: "",
      latest_message_at: new Date().toISOString(),
      status: "pending",
      unread_message_count: 0,
      supervisor_snapshot: null,
    };
    expect(item.sample_session_id).toBeNull();
  });
});
