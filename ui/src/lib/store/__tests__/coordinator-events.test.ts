import { describe, it, expect } from "vitest";
import { __test_applySSEToSession } from "../session-store";
import type { Session, SSEEventData } from "../../api/types";

function base(): Session {
  return { session_id: "s1", title: "t", status: "running", events: [] };
}

const SHAPES: SSEEventData[] = [
  {
    type: "coordinator_dispatch",
    data: {
      root_session_id: "r", parent_session_id: "p", child_session_id: null,
      coordinator_run_id: "run1", work_unit_id: null,
      event_id: "e1", step_id: "s1", work_unit_count: 2,
      work_unit_ids: ["wu1", "wu2"], phases: ["write", "exploration"],
    },
  },
  {
    type: "coordinator_worker_spawned",
    data: {
      root_session_id: "r", parent_session_id: "p", child_session_id: "c1",
      coordinator_run_id: "run1", work_unit_id: "wu1",
      event_id: "e2", objective: "do x", phase: "write",
      allowed_tools: ["file_read"], write_lease_count: 1,
    },
  },
  {
    type: "coordinator_reduce",
    data: {
      root_session_id: "r", parent_session_id: "p", child_session_id: null,
      coordinator_run_id: "run1", work_unit_id: null,
      event_id: "e3", group_outcome: "success", per_worker_outcomes: { wu1: "success" },
      diagnostics_summary: "ok", conflict_paths: [],
      cost_total: { total_input_tokens: 10, total_output_tokens: 5, total_usd: 0.01, tool_call_count: 3 },
    },
  },
  {
    type: "coordinator_apply",
    data: {
      root_session_id: "r", parent_session_id: "p", child_session_id: null,
      coordinator_run_id: "run1", work_unit_id: null,
      event_id: "e4", apply_status: "applied", file_count: 2, total_bytes: 100,
      failed_at_path: null, rollback_status: null,
    },
  },
  {
    type: "coordinator_sibling_cancel",
    data: {
      root_session_id: "r", parent_session_id: "p", child_session_id: null,
      coordinator_run_id: "run1", work_unit_id: "wu1",
      event_id: "e5", triggered_by_work_unit_id: "wu1", triggered_by_outcome: "failed",
      cancelled_work_unit_ids: ["wu2"], reason: "fail_fast",
    },
  },
];

describe("coordinator SSE events land in session.events", () => {
  it("stores each of the 5 coordinator event shapes keyed on .event", () => {
    let session = base();
    for (const ev of SHAPES) {
      session = __test_applySSEToSession(session, ev);
    }
    const stored = session.events.map((e) => (e as { event: string }).event);
    expect(stored).toEqual([
      "coordinator_dispatch",
      "coordinator_worker_spawned",
      "coordinator_reduce",
      "coordinator_apply",
      "coordinator_sibling_cancel",
    ]);
  });

  it("dedups a re-delivered coordinator event by event_id", () => {
    let session = base();
    session = __test_applySSEToSession(session, SHAPES[0]);
    session = __test_applySSEToSession(session, SHAPES[0]); // same event_id e1
    const dispatches = session.events.filter(
      (e) => (e as { event: string }).event === "coordinator_dispatch"
    );
    expect(dispatches.length).toBe(1);
  });
});
