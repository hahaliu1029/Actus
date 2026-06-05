import { describe, it, expect } from "vitest";
import {
  deriveLatestControlMode,
  deriveStatusFromEvents,
} from "../session-store";
import type { SessionEventRecord } from "../session-store";

const mode = (
  to: string,
  mode_revision?: number,
  from_mode?: string
): SessionEventRecord => ({
  event: "session_mode_changed",
  data: { to, mode_revision, from_mode, reason: "x" },
});

describe("deriveLatestControlMode", () => {
  it("returns the control mode of the only event", () => {
    expect(deriveLatestControlMode([mode("waiting", 1)])).toBe("waiting");
  });

  it("max mode_revision wins regardless of array order", () => {
    const events = [mode("takeover", 5), mode("running", 9), mode("waiting", 3)];
    expect(deriveLatestControlMode(events)).toBe("running");
  });

  it("falls back to latest-by-order when mode_revision is absent", () => {
    const events: SessionEventRecord[] = [
      { event: "session_mode_changed", data: { to: "takeover", reason: "x" } },
      { event: "session_mode_changed", data: { to: "running", reason: "x" } },
    ];
    expect(deriveLatestControlMode(events)).toBe("running");
  });

  it("a later revision-LESS event wins over an earlier revisioned one (read-fail fallback, R1#P1)", () => {
    const events: SessionEventRecord[] = [
      mode("takeover", 5),
      { event: "session_mode_changed", data: { to: "running", reason: "x" } },
    ];
    expect(deriveLatestControlMode(events)).toBe("running");
  });

  it("an earlier revision-less event does NOT beat a later revisioned one", () => {
    const events: SessionEventRecord[] = [
      { event: "session_mode_changed", data: { to: "takeover", reason: "x" } },
      mode("running", 9),
    ];
    expect(deriveLatestControlMode(events)).toBe("running");
  });

  it("explicit mode_revision: null is treated as missing (wire emits null, not omitted; R4#P2)", () => {
    const events: SessionEventRecord[] = [
      { event: "session_mode_changed", data: { to: "takeover", reason: "x", mode_revision: null } },
      { event: "session_mode_changed", data: { to: "running", reason: "x", mode_revision: null } },
    ];
    expect(deriveLatestControlMode(events)).toBe("running"); // later wins
  });

  it("ignores non-control `to` values and non-mode events", () => {
    const events: SessionEventRecord[] = [
      { event: "control", data: { action: "started" } },
      { event: "session_mode_changed", data: { to: "completed", reason: "x" } },
    ];
    expect(deriveLatestControlMode(events)).toBeNull();
  });
});

describe("deriveStatusFromEvents with session_mode_changed", () => {
  it("drives a BACKWARD takeover→running transition (pickMoreAdvancedStatus alone fails)", () => {
    const events = [mode("takeover", 4), mode("running", 7)];
    expect(deriveStatusFromEvents(events)).toBe("running");
  });

  it("regression: a session_mode_changed event does NOT reset status to running", () => {
    expect(deriveStatusFromEvents([mode("takeover_pending", 2)])).toBe("takeover_pending");
  });

  it("terminal precedence: a later done after a mode-changed yields completed", () => {
    const events: SessionEventRecord[] = [
      mode("takeover", 5),
      { event: "done", data: {} },
    ];
    expect(deriveStatusFromEvents(events)).toBe("completed");
  });

  it("terminal precedence: health terminated wins over a stale mode-changed", () => {
    const events: SessionEventRecord[] = [
      mode("takeover", 5),
      { event: "health", data: { status: "terminated" } },
    ];
    expect(deriveStatusFromEvents(events)).toBe("timed_out");
  });

  it("terminal precedence: a later finishing after a mode-changed yields completed (R1#P3, E1 normalization)", () => {
    const events: SessionEventRecord[] = [
      mode("takeover", 5),
      { event: "finishing", data: {} },
    ];
    expect(deriveStatusFromEvents(events)).toBe("completed");
  });

  it("existing inference still works as fallback when no mode event present", () => {
    const events: SessionEventRecord[] = [
      { event: "wait", data: {} },
    ];
    expect(deriveStatusFromEvents(events)).toBe("waiting");
  });
});

import { resolveMergedSessionStatus } from "../session-store";
import { __test_applySSEToSession } from "../session-store";
import type { Session } from "../../api/types";

describe("live reducer applySSEToSession with session_mode_changed", () => {
  it("appends the event to the session events list", () => {
    const base: Session = {
      session_id: "s1",
      title: "t",
      status: "takeover",
      events: [],
    };
    const next = __test_applySSEToSession(base, {
      type: "session_mode_changed",
      data: { to: "running", reason: "takeover_ended", mode_revision: 9 },
    });
    expect(
      next.events.some(
        (e) => (e as { event: string }).event === "session_mode_changed"
      )
    ).toBe(true);
  });
});

describe("resolveMergedSessionStatus", () => {
  // signature: (events, remoteStatus, monotonicFallback)
  it("control mode (end-takeover→running) wins over a stale monotonic takeover", () => {
    const merged: SessionEventRecord[] = [
      { event: "control", data: { action: "started", source: "user" } },
      mode("running", 12, "takeover"),
    ];
    // remote (DB) = running (not terminal); monotonic would keep "takeover".
    expect(resolveMergedSessionStatus(merged, "running", "takeover")).toBe("running");
  });

  it("reopen (completed→takeover_pending) wins over a stale monotonic completed", () => {
    // NOTE: `done` is NOT in the event list (applySSEToSession drops it); the
    // reopen control + mode events are. remote (DB) = takeover_pending (NOT terminal).
    const merged: SessionEventRecord[] = [
      { event: "control", data: { action: "reopened", source: "user" } },
      mode("takeover_pending", 20),
    ];
    expect(resolveMergedSessionStatus(merged, "takeover_pending", "completed")).toBe(
      "takeover_pending"
    );
  });

  it("terminal remote status keeps terminal over a stale mode event (R10#P1)", () => {
    // Real shape: a session that went takeover→completed has the mode event in
    // the log but NO `done` (dropped); the DB status is the terminal authority.
    const merged: SessionEventRecord[] = [mode("takeover", 5)];
    expect(resolveMergedSessionStatus(merged, "completed", "completed")).toBe("completed");
    expect(resolveMergedSessionStatus(merged, "timed_out", "timed_out")).toBe("timed_out");
  });

  it("no mode events → returns the monotonic fallback unchanged (R9#P1)", () => {
    const merged: SessionEventRecord[] = [{ event: "wait", data: {} }];
    expect(resolveMergedSessionStatus(merged, "running", "running")).toBe("running");
  });

  it("real paired end-takeover sequence (mode(running) + control(ended)) → running (R10#P2)", () => {
    // end_takeover ALWAYS emits mode(running, higher rev) paired with control(ended),
    // so the real log resolves to running (matching the live reducer); the
    // unpaired [mode(takeover), control(ended)] sequence is unreachable.
    const merged: SessionEventRecord[] = [
      mode("takeover", 10),
      mode("running", 15),
      { event: "control", data: { action: "ended", source: "user", handoff_mode: "continue" } },
    ];
    expect(resolveMergedSessionStatus(merged, "running", "takeover")).toBe("running");
  });
});
