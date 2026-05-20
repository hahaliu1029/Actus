import { describe, it, expect, beforeEach } from "vitest";
import { useSessionStore } from "@/lib/store/session-store";
import type { ListSessionItem } from "@/lib/api/types";

function makeItem(overrides: Partial<ListSessionItem> = {}): ListSessionItem {
  return {
    session_id: "s",
    title: "t",
    sample_session_id: null,
    parent_session_id: null,
    worker_type: "root",
    latest_message: "",
    latest_message_at: null,
    status: "pending",
    unread_message_count: 0,
    supervisor_snapshot: null,
    ...overrides,
  };
}

describe("useFilteredSessionsForList selector", () => {
  beforeEach(() => {
    useSessionStore.setState({ sessions: [] });
  });

  it("includes sessions where sample_session_id is null", () => {
    const main = makeItem({
      session_id: "main-1",
      title: "Main session",
      latest_message_at: new Date().toISOString(),
    });
    useSessionStore.setState({ sessions: [main] });
    const filtered = useSessionStore.getState().getFilteredSessionsForList();
    expect(filtered).toHaveLength(1);
    expect(filtered[0]?.session_id).toBe("main-1");
  });

  it("excludes probe child sessions (sample_session_id non-null)", () => {
    const main = makeItem({ session_id: "main-1", title: "main" });
    const child = makeItem({
      session_id: "child-1",
      title: "[probe] x",
      sample_session_id: "main-1",
      parent_session_id: "main-1",
      worker_type: "subagent",
    });
    useSessionStore.setState({ sessions: [main, child] });

    const filtered = useSessionStore.getState().getFilteredSessionsForList();
    expect(filtered).toHaveLength(1);
    expect(filtered.find((s) => s.session_id === "child-1")).toBeUndefined();
  });

  it("probeState actions: startProbe / updateChildByPrompt / updateChild / setProbeSummary / resetProbe", () => {
    const store = useSessionStore.getState();
    store.startProbe("probe-1", ["question A", "question B"]);
    let probe = useSessionStore.getState().probeState;
    expect(probe.running).toBe(true);
    expect(probe.probe_run_id).toBe("probe-1");
    expect(probe.children).toHaveLength(2);
    expect(probe.children[0]?.child_session_id).toBe("");

    useSessionStore.getState().updateChildByPrompt("question A", {
      child_session_id: "child-A",
      outcome: "pending",
    });
    probe = useSessionStore.getState().probeState;
    const childA = probe.children.find((c) => c.prompt === "question A");
    expect(childA?.child_session_id).toBe("child-A");
    expect(childA?.outcome).toBe("pending");
    const childB = probe.children.find((c) => c.prompt === "question B");
    expect(childB?.child_session_id).toBe("");

    useSessionStore.getState().updateChild("child-A", {
      outcome: "completed",
      final_answer: "ans A",
    });
    probe = useSessionStore.getState().probeState;
    expect(probe.children.find((c) => c.child_session_id === "child-A")?.outcome)
      .toBe("completed");

    useSessionStore.getState().setProbeSummary("integrated", ["warn 1"]);
    probe = useSessionStore.getState().probeState;
    expect(probe.running).toBe(false);
    expect(probe.summary).toBe("integrated");
    expect(probe.validation_warnings).toEqual(["warn 1"]);

    useSessionStore.getState().resetProbe();
    probe = useSessionStore.getState().probeState;
    expect(probe.running).toBe(false);
    expect(probe.probe_run_id).toBeNull();
    expect(probe.children).toEqual([]);
    expect(probe.summary).toBeNull();
    expect(probe.validation_warnings).toEqual([]);
  });

  it("updateChildByPrompt only updates the FIRST matching row without a session id", () => {
    const store = useSessionStore.getState();
    store.startProbe("probe-2", ["dup", "dup"]);
    useSessionStore.getState().updateChildByPrompt("dup", {
      child_session_id: "first",
      outcome: "pending",
    });
    const probe = useSessionStore.getState().probeState;
    expect(probe.children[0]?.child_session_id).toBe("first");
    expect(probe.children[1]?.child_session_id).toBe("");
  });

  it("setProbeError clears running and stores message (Codex R1 P1#2)", () => {
    const store = useSessionStore.getState();
    store.startProbe("probe-err", ["q"]);
    expect(useSessionStore.getState().probeState.running).toBe(true);
    useSessionStore.getState().setProbeError("http-error");
    const probe = useSessionStore.getState().probeState;
    expect(probe.running).toBe(false);
    expect(probe.error).toBe("http-error");
  });

  it("startProbe clears error from a previous run", () => {
    useSessionStore.getState().setProbeError("stale");
    useSessionStore.getState().startProbe("p2", ["x"]);
    expect(useSessionStore.getState().probeState.error).toBeNull();
  });
});
