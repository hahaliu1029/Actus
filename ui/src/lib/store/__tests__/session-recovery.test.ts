import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  __test_applySSEToSession,
  deriveStatusFromEvents,
  pickMoreAdvancedStatus,
  useSessionStore,
} from "../session-store";
import type { SessionEventRecord } from "../session-store";
import type { sessionApi } from "../../api/session";
import type {
  ChatParams,
  SSEEventData,
  SSEEventHandler,
  SupervisorSnapshot,
} from "../../api/types";

type SessionApi = typeof sessionApi;

vi.mock("../../api/session", () => ({
  sessionApi: {
    getSession: vi.fn(),
    getEventsSince: vi.fn(),
    getSessions: vi.fn(),
    createSession: vi.fn(),
    chat: vi.fn(),
    stopSession: vi.fn(),
    deleteSession: vi.fn(),
    clearUnreadMessageCount: vi.fn(),
    getSessionFiles: vi.fn(),
    streamSessions: vi.fn(),
    viewFile: vi.fn(),
    viewShell: vi.fn(),
    downloadSandboxFile: vi.fn(),
    getTakeover: vi.fn(),
    startTakeover: vi.fn(),
    renewTakeover: vi.fn(),
    rejectTakeover: vi.fn(),
    endTakeover: vi.fn(),
    reopenTakeover: vi.fn(),
  },
}));

vi.mock("@/lib/api/session-compaction", () => ({
  fetchCompactionList: vi.fn(async () => []),
  fetchCompactionDetail: vi.fn(),
  fetchCompactionOriginalContent: vi.fn(),
}));

describe("deriveStatusFromEvents", () => {
  it("returns timed_out for HealthEvent TERMINATED", () => {
    const events: SessionEventRecord[] = [
      { event: "health", data: { status: "terminated" } },
    ];
    expect(deriveStatusFromEvents(events)).toBe("timed_out");
  });

  it("returns timed_out for HealthEvent TERMINATING", () => {
    const events: SessionEventRecord[] = [
      { event: "health", data: { status: "terminating" } },
    ];
    expect(deriveStatusFromEvents(events)).toBe("timed_out");
  });

  it("returns waiting for ToolConfirmationEvent", () => {
    const events: SessionEventRecord[] = [
      { event: "tool_confirmation", data: { tool_call_id: "tc-1" } },
    ];
    expect(deriveStatusFromEvents(events)).toBe("waiting");
  });

  it("returns waiting for WaitEvent", () => {
    const events: SessionEventRecord[] = [
      { event: "wait", data: {} },
    ];
    expect(deriveStatusFromEvents(events)).toBe("waiting");
  });

  it("returns takeover_pending for ControlEvent REQUESTED", () => {
    const events: SessionEventRecord[] = [
      { event: "control", data: { action: "requested" } },
    ];
    expect(deriveStatusFromEvents(events)).toBe("takeover_pending");
  });

  it("returns takeover for ControlEvent STARTED", () => {
    const events: SessionEventRecord[] = [
      { event: "control", data: { action: "started" } },
    ];
    expect(deriveStatusFromEvents(events)).toBe("takeover");
  });

  it("returns running for ControlEvent ENDED with handoff_mode=continue", () => {
    const events: SessionEventRecord[] = [
      { event: "control", data: { action: "ended", handoff_mode: "continue" } },
    ];
    expect(deriveStatusFromEvents(events)).toBe("running");
  });

  it("returns completed for ControlEvent ENDED with handoff_mode=complete", () => {
    const events: SessionEventRecord[] = [
      { event: "control", data: { action: "ended", handoff_mode: "complete" } },
    ];
    expect(deriveStatusFromEvents(events)).toBe("completed");
  });

  it("returns completed for ControlEvent REJECTED with reason=terminate", () => {
    const events: SessionEventRecord[] = [
      { event: "control", data: { action: "rejected", reason: "terminate" } },
    ];
    expect(deriveStatusFromEvents(events)).toBe("completed");
  });

  it("returns completed for DoneEvent", () => {
    const events: SessionEventRecord[] = [
      { event: "done", data: {} },
    ];
    expect(deriveStatusFromEvents(events)).toBe("completed");
  });

  it("returns completed for FinishingEvent (E1 normalization)", () => {
    const events: SessionEventRecord[] = [
      { event: "finishing", data: {} },
    ];
    expect(deriveStatusFromEvents(events)).toBe("completed");
  });

  it("returns null when no status signals", () => {
    const events: SessionEventRecord[] = [
      { event: "message", data: { role: "assistant" } },
    ];
    expect(deriveStatusFromEvents(events)).toBeNull();
  });

  it("takes last signal event when multiple exist", () => {
    const events: SessionEventRecord[] = [
      { event: "wait", data: {} },
      { event: "health", data: { status: "terminated" } },
    ];
    expect(deriveStatusFromEvents(events)).toBe("timed_out");
  });

  it("timed_out is sticky — done/error after TERMINATED stays timed_out", () => {
    const events: SessionEventRecord[] = [
      { event: "health", data: { status: "terminated" } },
      { event: "done", data: {} },
    ];
    expect(deriveStatusFromEvents(events)).toBe("timed_out");
  });

  it("timed_out is sticky — error after TERMINATING stays timed_out", () => {
    const events: SessionEventRecord[] = [
      { event: "health", data: { status: "terminating" } },
      { event: "error", data: {} },
    ];
    expect(deriveStatusFromEvents(events)).toBe("timed_out");
  });

  it("returns takeover_pending for ControlEvent EXPIRED with reason=takeover_timeout", () => {
    const events: SessionEventRecord[] = [
      {
        event: "control",
        data: { action: "expired", reason: "takeover_timeout" },
      },
    ];
    expect(deriveStatusFromEvents(events)).toBe("takeover_pending");
  });

  it("returns completed for ControlEvent EXPIRED with reason=pending_timeout", () => {
    const events: SessionEventRecord[] = [
      {
        event: "control",
        data: { action: "expired", reason: "pending_timeout" },
      },
    ];
    expect(deriveStatusFromEvents(events)).toBe("completed");
  });

  it("returns null for DEGRADED health event (informational, no state change)", () => {
    const events: SessionEventRecord[] = [
      { event: "health", data: { status: "degraded" } },
    ];
    // DEGRADED keeps currentStatus (running) which is the default.
    // Since the only signal resolves to running and no "real" state change occurred,
    // resolveStatusFromEvent returns "running" but sawSignal is true → returns "running"
    expect(deriveStatusFromEvents(events)).toBe("running");
  });
});

describe("pickMoreAdvancedStatus", () => {
  it("timed_out wins over everything", () => {
    expect(pickMoreAdvancedStatus("completed", "timed_out")).toBe("timed_out");
    expect(pickMoreAdvancedStatus("timed_out", "running")).toBe("timed_out");
    expect(pickMoreAdvancedStatus("timed_out", "completed")).toBe("timed_out");
  });

  it("completed wins over running", () => {
    expect(pickMoreAdvancedStatus("running", "completed")).toBe("completed");
  });

  it("waiting wins over running", () => {
    expect(pickMoreAdvancedStatus("running", "waiting")).toBe("waiting");
  });

  it("ignores null candidates", () => {
    expect(pickMoreAdvancedStatus("running", null)).toBe("running");
    expect(pickMoreAdvancedStatus(null, "waiting")).toBe("waiting");
  });

  it("returns null when all null", () => {
    expect(pickMoreAdvancedStatus(null, null)).toBeNull();
  });

  it("handles three arguments", () => {
    expect(
      pickMoreAdvancedStatus("running", "waiting", "timed_out")
    ).toBe("timed_out");
  });
});

describe("recoverSession", () => {
  beforeEach(() => {
    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: {
        session_id: "s1",
        title: "test",
        status: "running",
        events: [
          { event: "message", data: { role: "user", event_id: "evt-1" } },
        ],
      },
      isChatting: false,
      chatSessionId: null,
      chatAbort: null,
    });
    vi.clearAllMocks();
  });

  it("merges recovered events with local using recovery direction", async () => {
    const { sessionApi } = await import("../../api/session");
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [
        { event: "message", data: { role: "assistant", stream_id: "s-1", event_id: "evt-2" } },
        { event: "health", data: { status: "terminated", event_id: "evt-3" } },
      ],
      session_status: "running",
      has_more: false,
    });

    await useSessionStore.getState().recoverSession("s1");

    const session = useSessionStore.getState().currentSession;
    expect(session).not.toBeNull();
    expect(session!.status).toBe("timed_out");
  });

  it("works without lastEventId (tab reopen, no local events)", async () => {
    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: {
        session_id: "s1",
        title: "test",
        status: "running",
        events: [],
      },
      isChatting: false,
      chatSessionId: null,
      chatAbort: null,
    });
    const { sessionApi } = await import("../../api/session");
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [
        { event: "message", data: { role: "assistant", stream_id: "s-1", event_id: "evt-1" } },
        { event: "done", data: { event_id: "evt-2" } },
      ],
      session_status: "completed",
      has_more: false,
    });

    await useSessionStore.getState().recoverSession("s1");

    expect(sessionApi.getEventsSince).toHaveBeenCalledWith("s1", undefined, undefined);
    const session = useSessionStore.getState().currentSession;
    expect(session!.status).toBe("completed");
    expect((session!.events as SessionEventRecord[]).length).toBeGreaterThan(0);
  });

  it("does nothing for completed session", async () => {
    useSessionStore.setState({
      currentSession: {
        session_id: "s1",
        title: "test",
        status: "completed",
        events: [],
      },
    });
    const { sessionApi } = await import("../../api/session");

    await useSessionStore.getState().recoverSession("s1");

    expect(sessionApi.getEventsSince).not.toHaveBeenCalled();
  });

  it("fetchSessionById does not regress local timed_out to remote running", async () => {
    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: {
        session_id: "s1",
        title: "test",
        status: "timed_out",
        events: [
          { event: "health", data: { status: "terminated", event_id: "evt-1" } },
        ],
      },
    });
    const { sessionApi } = await import("../../api/session");
    (sessionApi.getSession as ReturnType<typeof vi.fn>).mockResolvedValue({
      session_id: "s1",
      title: "test",
      status: "running",
      events: [
        { event: "health", data: { status: "terminated", event_id: "evt-1" } },
      ],
    });

    await useSessionStore.getState().fetchSessionById("s1", { silent: true });

    const session = useSessionStore.getState().currentSession;
    expect(session!.status).toBe("timed_out");
  });

  it("fetchSessionById derives last_seq from returned event seqs", async () => {
    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: null,
    });
    const { sessionApi } = await import("../../api/session");
    (sessionApi.getSession as ReturnType<typeof vi.fn>).mockResolvedValue({
      session_id: "s1",
      title: "test",
      status: "running",
      events: [
        { event: "message", data: { role: "assistant", event_id: "1000-3", seq: 3 } },
        { event: "message", data: { role: "assistant", event_id: "1000-9", seq: 9 } },
      ],
    });

    await useSessionStore.getState().fetchSessionById("s1", { silent: true });

    expect(useSessionStore.getState().currentSession?.last_seq).toBe(9);
  });

  it("fetchSessionById preserves a higher local last_seq when merging live events", async () => {
    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: {
        session_id: "s1",
        title: "test",
        status: "running",
        last_seq: 7,
        events: [
          { event: "message", data: { role: "assistant", event_id: "1000-7", seq: 7 } },
        ],
      },
    });
    const { sessionApi } = await import("../../api/session");
    (sessionApi.getSession as ReturnType<typeof vi.fn>).mockResolvedValue({
      session_id: "s1",
      title: "test",
      status: "running",
      last_seq: 3,
      events: [
        { event: "message", data: { role: "assistant", event_id: "1000-3", seq: 3 } },
      ],
    });

    await useSessionStore.getState().fetchSessionById("s1", { silent: true });

    expect(useSessionStore.getState().currentSession?.last_seq).toBe(7);
  });

    it("fetchSessionById keeps local cursor but trusts remote supervisor snapshot", async () => {
    const localSnapshot: SupervisorSnapshot = {
      execution_revision: 0,
      execution_mode: "background",
      execution_phase: "running",
      background_reason: "auto_degrade",
      expires_at: "2026-05-11T08:30:00Z",
      retry_budget_remaining: 2,
      suspended_reason: null,
      terminal_reason: null,
      last_progress_at: "2026-05-11T08:29:00Z",
      is_alive: true,
      cancellation_state: "none",
    };
    const staleRemoteSnapshot: SupervisorSnapshot = {
      execution_revision: 0,
      execution_mode: "foreground",
      execution_phase: "running",
      background_reason: null,
      expires_at: null,
      retry_budget_remaining: 3,
      suspended_reason: null,
      terminal_reason: null,
      last_progress_at: "2026-05-11T08:00:00Z",
      is_alive: true,
      cancellation_state: "none",
    };
    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: {
        session_id: "s1",
        title: "test",
        status: "running",
        last_seq: 12,
        supervisor_snapshot: localSnapshot,
        events: [
          { event: "message", data: { role: "assistant", event_id: "1000-12", seq: 12 } },
        ],
      },
    });
    const { sessionApi } = await import("../../api/session");
    (sessionApi.getSession as ReturnType<typeof vi.fn>).mockResolvedValue({
      session_id: "s1",
      title: "test",
      status: "running",
      last_seq: 7,
      supervisor_snapshot: staleRemoteSnapshot,
      events: [
        { event: "message", data: { role: "assistant", event_id: "1000-7", seq: 7 } },
      ],
    });

    await useSessionStore.getState().fetchSessionById("s1", { silent: true });

      const session = useSessionStore.getState().currentSession;
      expect(session?.last_seq).toBe(12);
      expect(session?.supervisor_snapshot).toEqual(staleRemoteSnapshot);
    });

  it("fetchSessionById stores remote last_seq when only the cursor advances", async () => {
    const events: SessionEventRecord[] = [
      { event: "message", data: { role: "assistant", event_id: "1000-3", seq: 3 } },
    ];
    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: {
        session_id: "s1",
        title: "test",
        status: "running",
        last_seq: 3,
        events,
      },
    });
    const { sessionApi } = await import("../../api/session");
    (sessionApi.getSession as ReturnType<typeof vi.fn>).mockResolvedValue({
      session_id: "s1",
      title: "test",
      status: "running",
      last_seq: 9,
      events,
    });

    await useSessionStore.getState().fetchSessionById("s1", { silent: true });

    expect(useSessionStore.getState().currentSession?.last_seq).toBe(9);
  });

  it("does not regress status on empty events", async () => {
    useSessionStore.setState({
      currentSession: {
        session_id: "s1",
        title: "test",
        status: "timed_out",
        events: [
          { event: "health", data: { status: "terminated", event_id: "evt-1" } },
        ],
      },
    });
    const { sessionApi } = await import("../../api/session");
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [],
      session_status: "running",
      has_more: false,
    });

    await useSessionStore.getState().recoverSession("s1");

    const session = useSessionStore.getState().currentSession;
    expect(session!.status).toBe("timed_out");
  });

  it("live + recovery merge dedups overlapping event_ids", async () => {
    // 本地已有 3 帧 live events (来自 SSE), event_id "1000-0" / "1000-1" / "1000-2"
    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: {
        session_id: "s1",
        title: "test",
        status: "running",
        events: [
          { event: "message", data: { role: "assistant", event_id: "1000-0" } },
          { event: "message", data: { role: "assistant", event_id: "1000-1" } },
          { event: "message", data: { role: "assistant", event_id: "1000-2" } },
        ],
      },
      isChatting: false,
      chatSessionId: null,
      chatAbort: null,
    });
    vi.clearAllMocks();

    const { sessionApi } = await import("../../api/session");
    // Recovery 返回 5 帧, 前 3 帧 id 与 live 重叠, 后 2 帧是新 gap
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [
        { event: "message", data: { role: "assistant", event_id: "1000-0" } },
        { event: "message", data: { role: "assistant", event_id: "1000-1" } },
        { event: "message", data: { role: "assistant", event_id: "1000-2" } },
        { event: "message", data: { role: "assistant", event_id: "1000-3" } },
        { event: "message", data: { role: "assistant", event_id: "1000-4" } },
      ],
      session_status: "running",
      has_more: false,
    });

    await useSessionStore.getState().recoverSession("s1");

    const merged = useSessionStore.getState().currentSession?.events ?? [];
    const ids = merged.map((e) => e.data?.event_id as string | undefined);
    // 断言: 合并后 exactly 5 条, 无重复, 顺序正确
    expect(ids).toEqual(["1000-0", "1000-1", "1000-2", "1000-3", "1000-4"]);
    expect(new Set(ids).size).toBe(5);
  });

  it("uses live SSE seq as the next recovery cursor", async () => {
    const liveSession = __test_applySSEToSession(
      {
        session_id: "s1",
        title: "test",
        status: "running",
        events: [],
      },
      {
        type: "message",
        data: {
          role: "assistant",
          message: "live",
          event_id: "1000-7",
          seq: 7,
          attachments: [],
        },
      },
    );
    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: liveSession,
      isChatting: false,
      chatSessionId: null,
      chatAbort: null,
    });

    const { sessionApi } = await import("../../api/session");
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [],
      session_status: "running",
      has_more: false,
      last_seq: 7,
      supervisor_snapshot: null,
    });

    await useSessionStore.getState().recoverSession("s1");

    expect(sessionApi.getEventsSince).toHaveBeenCalledWith("s1", "1000-7", 7);
  });

  it("updates cursor and supervisor snapshot on zero-event reconnect", async () => {
    const supervisorSnapshot: SupervisorSnapshot = {
      execution_revision: 0,
      execution_mode: "background",
      execution_phase: "running",
      background_reason: "explicit",
      expires_at: "2026-05-11T08:00:00Z",
      retry_budget_remaining: 2,
      suspended_reason: null,
      terminal_reason: null,
      last_progress_at: "2026-05-11T07:59:00Z",
      is_alive: true,
      cancellation_state: "none",
    };
    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: {
        session_id: "s1",
        title: "test",
        status: "running",
        last_seq: 7,
        supervisor_snapshot: null,
        events: [
          { event: "message", data: { role: "assistant", event_id: "1000-7", seq: 7 } },
        ],
      },
      isChatting: false,
      chatSessionId: null,
      chatAbort: null,
    });

    const { sessionApi } = await import("../../api/session");
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [],
      session_status: "running",
      has_more: false,
      last_seq: 12,
      supervisor_snapshot: supervisorSnapshot,
    });

    await useSessionStore.getState().recoverSession("s1");

    const session = useSessionStore.getState().currentSession;
    expect(session?.status).toBe("running");
    expect(session?.last_seq).toBe(12);
    expect(session?.supervisor_snapshot).toEqual(supervisorSnapshot);
  });

  it("keeps newer local supervisor snapshot when zero-event reconnect returns a stale cursor", async () => {
    const localSnapshot: SupervisorSnapshot = {
      execution_revision: 0,
      execution_mode: "background",
      execution_phase: "running",
      background_reason: "auto_degrade",
      expires_at: "2026-05-11T08:30:00Z",
      retry_budget_remaining: 2,
      suspended_reason: null,
      terminal_reason: null,
      last_progress_at: "2026-05-11T08:29:00Z",
      is_alive: true,
      cancellation_state: "none",
    };
    const staleRemoteSnapshot: SupervisorSnapshot = {
      execution_revision: 0,
      execution_mode: "foreground",
      execution_phase: "running",
      background_reason: null,
      expires_at: null,
      retry_budget_remaining: 3,
      suspended_reason: null,
      terminal_reason: null,
      last_progress_at: "2026-05-11T08:00:00Z",
      is_alive: true,
      cancellation_state: "none",
    };

    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: {
        session_id: "s1",
        title: "test",
        status: "running",
        last_seq: 12,
        supervisor_snapshot: localSnapshot,
        events: [
          { event: "message", data: { role: "assistant", event_id: "1000-12", seq: 12 } },
        ],
      },
      isChatting: false,
      chatSessionId: null,
      chatAbort: null,
    });

    const { sessionApi } = await import("../../api/session");
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [],
      session_status: "running",
      has_more: false,
      last_seq: 7,
      supervisor_snapshot: staleRemoteSnapshot,
    });

    await useSessionStore.getState().recoverSession("s1");

    const session = useSessionStore.getState().currentSession;
    expect(session?.last_seq).toBe(12);
    expect(session?.supervisor_snapshot).toEqual(localSnapshot);
  });

  it("updates supervisor snapshot from live execution_state_changed event", () => {
    const session = __test_applySSEToSession(
      {
        session_id: "s1",
        title: "test",
        status: "running",
        last_seq: 7,
        supervisor_snapshot: null,
        events: [],
      },
      {
        type: "execution_state_changed",
        data: {
          event_id: "1000-12",
          created_at: "2026-05-11T08:00:00Z",
          seq: 12,
          payload: {
            execution_revision: 1,
            execution_mode: "background",
            execution_phase: "running",
            background_reason: "auto_degrade",
            transition_reason: "auto_degrade_sse_disconnect",
            expires_at: "2026-05-11T08:30:00Z",
            retry_budget_remaining: 2,
            suspended_reason: null,
            terminal_reason: null,
          },
        },
      },
    );

    expect(session.status).toBe("running");
    expect(session.last_seq).toBe(12);
    expect(session.supervisor_snapshot).toMatchObject({
      execution_revision: 1,
      execution_mode: "background",
      execution_phase: "running",
      background_reason: "auto_degrade",
      expires_at: "2026-05-11T08:30:00Z",
      retry_budget_remaining: 2,
      suspended_reason: null,
      terminal_reason: null,
    });
  });

  it("ignores an older background execution event after a newer foreground transition", () => {
    const foregroundSnapshot: SupervisorSnapshot = {
      execution_revision: 2,
      execution_mode: "foreground",
      execution_phase: "running",
      background_reason: null,
      expires_at: null,
      retry_budget_remaining: 2,
      suspended_reason: null,
      terminal_reason: null,
      last_progress_at: "2026-05-11T08:01:00Z",
      is_alive: true,
      cancellation_state: "none",
    };

    const session = __test_applySSEToSession(
      {
        session_id: "s1",
        title: "test",
        status: "running",
        last_seq: 13,
        supervisor_snapshot: foregroundSnapshot,
        events: [],
      },
      {
        type: "execution_state_changed",
        data: {
          event_id: "1000-14",
          created_at: "2026-05-11T08:02:00Z",
          seq: 14,
          payload: {
            execution_revision: 1,
            execution_mode: "background",
            execution_phase: "running",
            background_reason: "auto_degrade",
            transition_reason: "auto_degrade_sse_disconnect",
            expires_at: "2026-05-11T08:30:00Z",
            retry_budget_remaining: 2,
            suspended_reason: null,
            terminal_reason: null,
          },
        },
      },
    );

    expect(session.supervisor_snapshot).toEqual(foregroundSnapshot);
    expect(session.last_seq).toBe(14);
  });

  it("keeps an authoritative suspended snapshot when a stale event has the same revision", () => {
    const suspendedSnapshot: SupervisorSnapshot = {
      execution_revision: 1,
      execution_mode: "background",
      execution_phase: "suspended",
      background_reason: "auto_degrade",
      expires_at: null,
      retry_budget_remaining: 0,
      suspended_reason: "retry_budget_exhausted",
      terminal_reason: null,
      last_progress_at: "2026-05-11T08:01:00Z",
      is_alive: false,
      cancellation_state: "none",
    };

    const session = __test_applySSEToSession(
      {
        session_id: "s1",
        title: "test",
        status: "running",
        last_seq: 13,
        supervisor_snapshot: suspendedSnapshot,
        events: [],
      },
      {
        type: "execution_state_changed",
        data: {
          event_id: "1000-14",
          created_at: "2026-05-11T08:02:00Z",
          seq: 14,
          payload: {
            execution_revision: 1,
            execution_mode: "background",
            execution_phase: "running",
            background_reason: "auto_degrade",
            transition_reason: "auto_degrade_sse_disconnect",
            expires_at: "2026-05-11T08:30:00Z",
            retry_budget_remaining: 2,
            suspended_reason: null,
            terminal_reason: null,
          },
        },
      },
    );

    expect(session.supervisor_snapshot).toEqual(suspendedSnapshot);
    expect(session.last_seq).toBe(14);
  });

  it("folds a legacy execution event when no supervisor snapshot exists", () => {
    const session = __test_applySSEToSession(
      {
        session_id: "s1",
        title: "test",
        status: "running",
        last_seq: 7,
        supervisor_snapshot: null,
        events: [],
      },
      {
        type: "execution_state_changed",
        data: {
          event_id: "1000-8",
          created_at: "2026-05-11T08:00:00Z",
          seq: 8,
          payload: {
            execution_mode: "background",
            execution_phase: "running",
            background_reason: "explicit",
            expires_at: null,
            retry_budget_remaining: 3,
            suspended_reason: null,
            terminal_reason: null,
          },
        },
      },
    );

    expect(session.supervisor_snapshot).toMatchObject({
      execution_revision: 0,
      execution_mode: "background",
      execution_phase: "running",
      background_reason: "explicit",
    });
  });

  it("lets a higher-revision execution event replace an authoritative snapshot", () => {
    const suspendedSnapshot: SupervisorSnapshot = {
      execution_revision: 1,
      execution_mode: "background",
      execution_phase: "suspended",
      background_reason: "auto_degrade",
      expires_at: null,
      retry_budget_remaining: 0,
      suspended_reason: "retry_budget_exhausted",
      terminal_reason: null,
      last_progress_at: "2026-05-11T08:01:00Z",
      is_alive: false,
      cancellation_state: "none",
    };

    const session = __test_applySSEToSession(
      {
        session_id: "s1",
        title: "test",
        status: "running",
        last_seq: 13,
        supervisor_snapshot: suspendedSnapshot,
        events: [],
      },
      {
        type: "execution_state_changed",
        data: {
          event_id: "1000-14",
          created_at: "2026-05-11T08:02:00Z",
          seq: 14,
          payload: {
            execution_revision: 2,
            execution_mode: "background",
            execution_phase: "running",
            background_reason: "explicit",
            transition_reason: "explicit_retry",
            expires_at: null,
            retry_budget_remaining: 2,
            suspended_reason: null,
            terminal_reason: null,
          },
        },
      },
    );

    expect(session.supervisor_snapshot).toMatchObject({
      execution_revision: 2,
      execution_mode: "background",
      execution_phase: "running",
      background_reason: "explicit",
      retry_budget_remaining: 2,
      suspended_reason: null,
    });
  });

  it("updates supervisor snapshot from recovered execution_state_changed event", async () => {
    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: {
        session_id: "s1",
        title: "test",
        status: "running",
        last_seq: 7,
        supervisor_snapshot: null,
        events: [
          { event: "message", data: { role: "assistant", event_id: "1000-7", seq: 7 } },
        ],
      },
      isChatting: false,
      chatSessionId: null,
      chatAbort: null,
    });
    const { sessionApi } = await import("../../api/session");
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [
        {
          event: "execution_state_changed",
          data: {
            event_id: "1000-12",
            created_at: "2026-05-11T08:00:00Z",
            seq: 12,
            payload: {
              execution_revision: 1,
              execution_mode: "background",
              execution_phase: "running",
              background_reason: "auto_degrade",
              transition_reason: "auto_degrade_sse_disconnect",
              expires_at: "2026-05-11T08:30:00Z",
              retry_budget_remaining: 1,
              suspended_reason: null,
              terminal_reason: null,
            },
          },
        },
      ],
      session_status: "running",
      has_more: false,
      last_seq: 12,
      supervisor_snapshot: null,
    });

    await useSessionStore.getState().recoverSession("s1");

    const session = useSessionStore.getState().currentSession;
    expect(session?.status).toBe("running");
    expect(session?.last_seq).toBe(12);
    expect(session?.supervisor_snapshot).toMatchObject({
      execution_revision: 1,
      execution_mode: "background",
      execution_phase: "running",
      background_reason: "auto_degrade",
      expires_at: "2026-05-11T08:30:00Z",
      retry_budget_remaining: 1,
      suspended_reason: null,
      terminal_reason: null,
    });
  });

  it("treats the remote supervisor snapshot revision as a lower bound for recovered events", async () => {
    const foregroundSnapshot: SupervisorSnapshot = {
      execution_revision: 4,
      execution_mode: "foreground",
      execution_phase: "running",
      background_reason: null,
      expires_at: null,
      retry_budget_remaining: 1,
      suspended_reason: null,
      terminal_reason: null,
      last_progress_at: "2026-05-11T08:04:00Z",
      is_alive: true,
      cancellation_state: "none",
    };
    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: {
        session_id: "s1",
        title: "test",
        status: "running",
        last_seq: 7,
        supervisor_snapshot: null,
        events: [
          { event: "message", data: { role: "assistant", event_id: "1000-7", seq: 7 } },
        ],
      },
      isChatting: false,
      chatSessionId: null,
      chatAbort: null,
    });
    const { sessionApi } = await import("../../api/session");
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [
        {
          event: "execution_state_changed",
          data: {
            event_id: "1000-12",
            created_at: "2026-05-11T08:03:00Z",
            seq: 12,
            payload: {
              execution_revision: 3,
              execution_mode: "background",
              execution_phase: "running",
              background_reason: "auto_degrade",
              transition_reason: "auto_degrade_sse_disconnect",
              expires_at: "2026-05-11T08:30:00Z",
              retry_budget_remaining: 1,
              suspended_reason: null,
              terminal_reason: null,
            },
          },
        },
      ],
      session_status: "running",
      has_more: false,
      last_seq: 12,
      supervisor_snapshot: foregroundSnapshot,
    });

    await useSessionStore.getState().recoverSession("s1");

    expect(useSessionStore.getState().currentSession?.supervisor_snapshot).toEqual(
      foregroundSnapshot,
    );
  });

  it("keeps an authoritative snapshot over a same-revision recovered event", async () => {
    const suspendedSnapshot: SupervisorSnapshot = {
      execution_revision: 1,
      execution_mode: "background",
      execution_phase: "suspended",
      background_reason: "auto_degrade",
      expires_at: null,
      retry_budget_remaining: 0,
      suspended_reason: "retry_budget_exhausted",
      terminal_reason: null,
      last_progress_at: "2026-05-11T08:04:00Z",
      is_alive: false,
      cancellation_state: "none",
    };
    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: {
        session_id: "s1",
        title: "test",
        status: "running",
        last_seq: 7,
        supervisor_snapshot: null,
        events: [
          { event: "message", data: { role: "assistant", event_id: "1000-7", seq: 7 } },
        ],
      },
      isChatting: false,
      chatSessionId: null,
      chatAbort: null,
    });
    const { sessionApi } = await import("../../api/session");
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [
        {
          event: "execution_state_changed",
          data: {
            event_id: "1000-12",
            created_at: "2026-05-11T08:03:00Z",
            seq: 12,
            payload: {
              execution_revision: 1,
              execution_mode: "background",
              execution_phase: "running",
              background_reason: "auto_degrade",
              transition_reason: "auto_degrade_sse_disconnect",
              expires_at: "2026-05-11T08:30:00Z",
              retry_budget_remaining: 1,
              suspended_reason: null,
              terminal_reason: null,
            },
          },
        },
      ],
      session_status: "running",
      has_more: false,
      last_seq: 12,
      supervisor_snapshot: suspendedSnapshot,
    });

    await useSessionStore.getState().recoverSession("s1");

    const session = useSessionStore.getState().currentSession;
    expect(session?.supervisor_snapshot).toEqual(suspendedSnapshot);
    expect(session?.last_seq).toBe(12);
    const eventIds = session?.events.map((event) => event.data?.event_id);
    expect(eventIds).toHaveLength(2);
    expect(eventIds).toEqual(expect.arrayContaining(["1000-7", "1000-12"]));
  });

  it("keeps newer local supervisor snapshot when recovered events do not advance cursor", async () => {
    const localSnapshot: SupervisorSnapshot = {
      execution_revision: 0,
      execution_mode: "background",
      execution_phase: "running",
      background_reason: "auto_degrade",
      expires_at: "2026-05-11T08:30:00Z",
      retry_budget_remaining: 2,
      suspended_reason: null,
      terminal_reason: null,
      last_progress_at: "2026-05-11T08:29:00Z",
      is_alive: true,
      cancellation_state: "none",
    };
    const staleRemoteSnapshot: SupervisorSnapshot = {
      execution_revision: 0,
      execution_mode: "foreground",
      execution_phase: "running",
      background_reason: null,
      expires_at: null,
      retry_budget_remaining: 3,
      suspended_reason: null,
      terminal_reason: null,
      last_progress_at: "2026-05-11T08:00:00Z",
      is_alive: true,
      cancellation_state: "none",
    };

    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: {
        session_id: "s1",
        title: "test",
        status: "running",
        last_seq: 12,
        supervisor_snapshot: localSnapshot,
        events: [
          { event: "message", data: { role: "assistant", event_id: "1000-12", seq: 12 } },
        ],
      },
      isChatting: false,
      chatSessionId: null,
      chatAbort: null,
    });

    const { sessionApi } = await import("../../api/session");
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [
        { event: "message", data: { role: "assistant", event_id: "1000-8", seq: 8 } },
      ],
      session_status: "running",
      has_more: false,
      last_seq: 8,
      supervisor_snapshot: staleRemoteSnapshot,
    });

    await useSessionStore.getState().recoverSession("s1");

    const session = useSessionStore.getState().currentSession;
    expect(session?.last_seq).toBe(12);
    expect(session?.supervisor_snapshot).toEqual(localSnapshot);
  });

  it("[Codex holistic R3+R4+R5] fires fetchCompactionList on zero-event reconnect (regression test)", async () => {
    // Locks the placement of the IIFE BEFORE the recoveredEvents.length===0
    // early-return so a future refactor that moves it back below the early
    // return is caught. Without this assertion, the 'keeps local status'
    // test still passes because mergeCompactionList only mutates events,
    // never status — so behavior is silent.
    const { sessionApi } = await import("../../api/session");
    const { fetchCompactionList } = await import("@/lib/api/session-compaction");
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [],
      session_status: null,
      has_more: false,
    });
    (fetchCompactionList as ReturnType<typeof vi.fn>).mockResolvedValue([]);

    await useSessionStore.getState().recoverSession("s1");
    // Wait for the fire-and-forget IIFE to complete its microtasks.
    await Promise.resolve();
    await Promise.resolve();

    expect(fetchCompactionList).toHaveBeenCalledWith("s1");
  });
});

describe("stream disconnect recovery with streamConnected + sawTerminalEvent", () => {
  // Helper: mock sessionApi.chat capturing all callbacks including onConnected (6th arg)
  type ChatCallbacks = {
    onEvent: SSEEventHandler;
    onError: (error: Error) => void;
    onClose: () => void;
    onConnected: () => void;
  };

  function mockChat(api: SessionApi, simulateConnected: boolean): ChatCallbacks {
    const cbs = {} as ChatCallbacks;
    (api.chat as ReturnType<typeof vi.fn>).mockImplementation(
      (
        _sid: string,
        _params: ChatParams,
        onEvent: SSEEventHandler,
        onError: (error: Error) => void,
        onClose: () => void,
        onConnected: () => void
      ) => {
        cbs.onEvent = onEvent;
        cbs.onError = onError;
        cbs.onClose = onClose;
        cbs.onConnected = onConnected;
        // Simulate createSSEStream() success/failure
        if (simulateConnected && onConnected) onConnected();
        return () => {};
      }
    );
    return cbs;
  }

  beforeEach(() => {
    vi.useFakeTimers();
    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: {
        session_id: "s1",
        title: "test",
        status: "running",
        events: [],
      },
      isChatting: false,
      chatSessionId: null,
      chatAbort: null,
      _isRecovering: false,
    });
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("keeps live streaming state after execution_state_changed event", async () => {
    const { sessionApi } = await import("../../api/session");
    const cbs = mockChat(sessionApi, true);

    await useSessionStore.getState().sendChat("s1", {});
    cbs.onEvent({
      type: "execution_state_changed",
      data: {
        event_id: "1000-12",
        created_at: "2026-05-11T08:00:00Z",
        seq: 12,
        payload: {
          execution_revision: 1,
          execution_mode: "background",
          execution_phase: "running",
          background_reason: "auto_degrade",
          transition_reason: "auto_degrade_sse_disconnect",
          expires_at: "2026-05-11T08:30:00Z",
          retry_budget_remaining: 2,
          suspended_reason: null,
          terminal_reason: null,
        },
      },
    });

    const state = useSessionStore.getState();
    expect(state.currentSession?.supervisor_snapshot).toMatchObject({
      execution_revision: 1,
      execution_mode: "background",
      execution_phase: "running",
      background_reason: "auto_degrade",
      expires_at: "2026-05-11T08:30:00Z",
      retry_budget_remaining: 2,
    });
    expect(state.currentSession?.last_seq).toBe(12);
    expect(state.isChatting).toBe(true);
    expect(state.chatSessionId).toBe("s1");
  });

  it("does NOT trigger recovery when stream ends after DoneEvent", async () => {
    const { sessionApi } = await import("../../api/session");
    const cbs = mockChat(sessionApi, true);
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [], session_status: "completed", has_more: false,
    });

    await useSessionStore.getState().sendChat("s1", {});
    cbs.onEvent({ type: "done", data: {} });
    cbs.onClose();

    await vi.advanceTimersByTimeAsync(3000);
    expect(sessionApi.getEventsSince).not.toHaveBeenCalled();
  });

  it("DOES trigger recovery when connected stream ends without terminal event", async () => {
    const { sessionApi } = await import("../../api/session");
    const cbs = mockChat(sessionApi, true);
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [], session_status: "running", has_more: false,
    });

    await useSessionStore.getState().sendChat("s1", {});
    cbs.onEvent({
      type: "message",
      data: { role: "assistant", stream_id: "s-1", message: "", attachments: [] },
    });
    cbs.onClose();

    await vi.advanceTimersByTimeAsync(3000);
    expect(sessionApi.getEventsSince).toHaveBeenCalled();
  });

  it("DOES trigger recovery when connected stream drops before first event", async () => {
    const { sessionApi } = await import("../../api/session");
    // Stream connected (onConnected called) but no events arrive before disconnect
    const cbs = mockChat(sessionApi, true);
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [], session_status: "running", has_more: false,
    });

    await useSessionStore.getState().sendChat("s1", {});
    // No onEvent calls — disconnect before first event
    cbs.onClose();

    await vi.advanceTimersByTimeAsync(3000);
    // streamConnected=true, sawTerminalEvent=false → recovery SHOULD fire
    expect(sessionApi.getEventsSince).toHaveBeenCalled();
  });

  it("suppresses error toast on recoverable disconnect (connected, no terminal)", async () => {
    const { sessionApi } = await import("../../api/session");
    const cbs = mockChat(sessionApi, true);
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [], session_status: "running", has_more: false,
    });

    const { useUIStore } = await import("../../store/ui-store");
    const setMessageSpy = vi.spyOn(useUIStore.getState(), "setMessage");

    await useSessionStore.getState().sendChat("s1", {});
    cbs.onEvent({
      type: "message",
      data: { role: "assistant", stream_id: "s-1", message: "", attachments: [] },
    });
    cbs.onError(new Error("network disconnect"));
    cbs.onClose();

    // Error toast should NOT have been shown
    expect(setMessageSpy).not.toHaveBeenCalledWith(
      expect.objectContaining({ type: "error" })
    );
    await vi.advanceTimersByTimeAsync(3000);
    expect(sessionApi.getEventsSince).toHaveBeenCalled();

    setMessageSpy.mockRestore();
  });

  it("does NOT trigger recovery when stream ends after ToolConfirmationEvent", async () => {
    const { sessionApi } = await import("../../api/session");
    const cbs = mockChat(sessionApi, true);

    await useSessionStore.getState().sendChat("s1", {});
    cbs.onEvent({
      type: "tool_confirmation",
      data: {
        tool_call_id: "tc-1",
        tool_name: "shell_execute",
        tool_args: {},
        risk_level: "high",
        risk_reason: "",
        matched_patterns: [],
        suggested_alternative: null,
        approval_options: [],
        timeout_seconds: 60,
      },
    });
    cbs.onClose();

    await vi.advanceTimersByTimeAsync(3000);
    expect(sessionApi.getEventsSince).not.toHaveBeenCalled();
  });

  it("does NOT trigger recovery when stream ends after OwnerConflictEvent", async () => {
    const { sessionApi } = await import("../../api/session");
    const cbs = mockChat(sessionApi, true);
    useSessionStore.setState({
      currentSession: {
        session_id: "s1",
        title: "test",
        status: "waiting",
        events: [],
      },
    });

    await useSessionStore.getState().sendChat("s1", {});
    cbs.onEvent({
      type: "owner_conflict",
      data: {
        event_id: "evt-owner-conflict",
        payload: {
          current_owner_connection_id: "user-1:tab-1",
          conflicting_connection_id: "user-1:tab-2",
          session_id: "s1",
          suggested_action: "wait_lease_expire",
        },
      },
    } as SSEEventData);
    cbs.onClose();

    await vi.advanceTimersByTimeAsync(3000);
    expect(sessionApi.getEventsSince).not.toHaveBeenCalled();
    expect(useSessionStore.getState().currentSession?.status).toBe("waiting");
    expect(useSessionStore.getState().currentSession?.events.at(-1)).toMatchObject({
      event: "owner_conflict",
      data: {
        payload: {
          current_owner_connection_id: "user-1:tab-1",
          conflicting_connection_id: "user-1:tab-2",
          session_id: "s1",
        },
      },
    });
  });

  it("shows error and does NOT trigger recovery when stream never connected", async () => {
    const { sessionApi } = await import("../../api/session");
    // simulateConnected=false → onConnected never called
    const cbs = mockChat(sessionApi, false);
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [], session_status: "running", has_more: false,
    });

    const { useUIStore } = await import("../../store/ui-store");
    const setMessageSpy = vi.spyOn(useUIStore.getState(), "setMessage");

    await useSessionStore.getState().sendChat("s1", {});
    // Stream never connected → onError with 5xx
    cbs.onError(new Error("HTTP 500: Internal Server Error"));
    cbs.onClose();

    await vi.advanceTimersByTimeAsync(3000);

    // Recovery should NOT have fired (stream never connected)
    expect(sessionApi.getEventsSince).not.toHaveBeenCalled();
    // Error toast SHOULD have been shown
    expect(setMessageSpy).toHaveBeenCalledWith(
      expect.objectContaining({ type: "error" })
    );

    setMessageSpy.mockRestore();
  });

  // R5b-5: tool_confirmation 提交收到 HTTP 409 → 自动走 /events?since=... 复播
  it("auto-reconnects SSE via getEventsSince when tool_confirmation submit returns 409", async () => {
    const { sessionApi } = await import("../../api/session");
    const cbs = mockChat(sessionApi, false);
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [], session_status: "running", has_more: false,
    });

    const { useUIStore } = await import("../../store/ui-store");
    const setMessageSpy = vi.spyOn(useUIStore.getState(), "setMessage");
    const { ApiError } = await import("../../api/auth-utils");

    await useSessionStore.getState().sendChat("s1", {
      tool_confirmation: {
        action: "approve",
        scope: "session",
        tool_call_id: "tc-r5b5",
      },
    });

    // winner 已赢 claim / late-duplicate → ApiError(409)
    cbs.onError(
      new ApiError({
        code: 409,
        httpStatus: 409,
        msg: "工具确认[tc-r5b5]已被处理完成，请通过 /events?since=... 重连 SSE",
      }),
    );

    // recoverSession 是 setTimeout(0) 调度的，fake timer advance 让它跑
    await vi.advanceTimersByTimeAsync(10);

    // auto-reconnect 必达：getEventsSince 被调，不给用户弹错误 toast
    expect(sessionApi.getEventsSince).toHaveBeenCalledWith("s1", undefined, undefined);
    expect(setMessageSpy).not.toHaveBeenCalledWith(
      expect.objectContaining({ type: "error" }),
    );

    setMessageSpy.mockRestore();
  });

  // R5b-5 Codex round-8 HIGH: 409 恢复后要脱离 waiting
  it("transitions out of waiting after 409 recovery when remote=running with zero events", async () => {
    const { sessionApi } = await import("../../api/session");
    const cbs = mockChat(sessionApi, false);
    // winner 已 resume 但还没产出新事件 → 后端 running，events=[]
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [],
      session_status: "running",
      has_more: false,
    });

    // 本地 session 停在 waiting（confirmation card 尚未消失）
    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: {
        session_id: "s1",
        title: "test",
        status: "waiting",
        events: [],
      },
      isChatting: false,
      chatSessionId: null,
      chatAbort: null,
      _isRecovering: false,
    });

    const { ApiError } = await import("../../api/auth-utils");
    await useSessionStore.getState().sendChat("s1", {
      tool_confirmation: {
        action: "approve",
        scope: "session",
        tool_call_id: "tc-r5b5-waiting",
      },
    });
    cbs.onError(
      new ApiError({ code: 409, httpStatus: 409, msg: "已被处理" }),
    );

    await vi.advanceTimersByTimeAsync(10);

    // Codex round-8 锁死：recoverSession 的 zero-event 分支必须让 remote=running
    // 覆盖 local=waiting，否则用户看到的卡片停在 waiting 永不退出。
    const status = useSessionStore.getState().currentSession?.status;
    expect(status).toBe("running");
  });

  it("keeps local status when remote is unknown and events=[]", async () => {
    // 防退：remoteStatus=null 时不能误踢 local waiting；只有 remote 明确非 waiting 才覆盖
    const { sessionApi } = await import("../../api/session");
    const cbs = mockChat(sessionApi, false);
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [],
      session_status: null,
      has_more: false,
    });

    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: {
        session_id: "s1",
        title: "test",
        status: "waiting",
        events: [],
      },
      isChatting: false,
      chatSessionId: null,
      chatAbort: null,
      _isRecovering: false,
    });

    const { ApiError } = await import("../../api/auth-utils");
    await useSessionStore.getState().sendChat("s1", {
      tool_confirmation: {
        action: "approve",
        scope: "session",
        tool_call_id: "tc-r5b5-waiting",
      },
    });
    cbs.onError(
      new ApiError({ code: 409, httpStatus: 409, msg: "已被处理" }),
    );
    await vi.advanceTimersByTimeAsync(10);

    // remote=null 时 zero-event 分支不动状态，local waiting 保留
    const status = useSessionStore.getState().currentSession?.status;
    expect(status).toBe("waiting");
  });

  it("does NOT auto-reconnect for 409 on non-tool_confirmation chat", async () => {
    // 验证 409 guard 只对 tool_confirmation 生效——普通 chat 的 409 走普通错误路径
    const { sessionApi } = await import("../../api/session");
    const cbs = mockChat(sessionApi, false);
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [], session_status: "running", has_more: false,
    });

    const { useUIStore } = await import("../../store/ui-store");
    const setMessageSpy = vi.spyOn(useUIStore.getState(), "setMessage");
    const { ApiError } = await import("../../api/auth-utils");

    await useSessionStore.getState().sendChat("s1", { message: "hi" });

    cbs.onError(
      new ApiError({ code: 409, httpStatus: 409, msg: "会话冲突" }),
    );

    await vi.advanceTimersByTimeAsync(10);

    // 非 tool_confirmation 的 409 不触发 auto-reconnect
    expect(sessionApi.getEventsSince).not.toHaveBeenCalled();
    expect(setMessageSpy).toHaveBeenCalledWith(
      expect.objectContaining({ type: "error" }),
    );

    setMessageSpy.mockRestore();
  });
});

// A4-0 Task 11 — live reducer status path for session_mode_changed.
// Mirrors the mockChat-driven live-event suite above (870-1000): it captures
// onEvent via mockChat(api, true) and starts the chat through sendChat so
// chatSessionId/isChatting are set, then fires the new event and asserts the
// reducer's status authority — WITHOUT ending the stream.
describe("live reducer honours session_mode_changed (status authority, no stream reset)", () => {
  type ChatCallbacks = {
    onEvent: SSEEventHandler;
    onError: (error: Error) => void;
    onClose: () => void;
    onConnected: () => void;
  };

  function mockChat(api: SessionApi, simulateConnected: boolean): ChatCallbacks {
    const cbs = {} as ChatCallbacks;
    (api.chat as ReturnType<typeof vi.fn>).mockImplementation(
      (
        _sid: string,
        _params: ChatParams,
        onEvent: SSEEventHandler,
        onError: (error: Error) => void,
        onClose: () => void,
        onConnected: () => void
      ) => {
        cbs.onEvent = onEvent;
        cbs.onError = onError;
        cbs.onClose = onClose;
        cbs.onConnected = onConnected;
        if (simulateConnected && onConnected) onConnected();
        return () => {};
      }
    );
    return cbs;
  }

  beforeEach(() => {
    vi.useFakeTimers();
    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: {
        session_id: "s1",
        title: "test",
        status: "running",
        events: [],
      },
      isChatting: false,
      chatSessionId: null,
      chatAbort: null,
      _isRecovering: false,
    });
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("(a) updates status from the mode event and does NOT reset streaming", async () => {
    const { sessionApi } = await import("../../api/session");
    const cbs = mockChat(sessionApi, true);

    await useSessionStore.getState().sendChat("s1", {});
    cbs.onEvent({
      type: "session_mode_changed",
      data: { to: "takeover_pending", reason: "takeover_requested", mode_revision: 1 },
    });

    const st = useSessionStore.getState();
    expect(st.currentSession?.status).toBe("takeover_pending");
    // session_mode_changed must NOT end the stream.
    expect(st.isChatting).toBe(true);
    expect(st.chatSessionId).toBe("s1");
  });

  it("(b) live mode_revision LWW: a later lower-revision event does NOT regress", async () => {
    const { sessionApi } = await import("../../api/session");
    const cbs = mockChat(sessionApi, true);

    await useSessionStore.getState().sendChat("s1", {});
    cbs.onEvent({
      type: "session_mode_changed",
      data: { to: "takeover", reason: "takeover_started", mode_revision: 10 },
    });
    cbs.onEvent({
      type: "session_mode_changed",
      data: { to: "running", reason: "takeover_ended", mode_revision: 9 },
    });

    // rev 10 wins over the later rev 9 (max mode_revision LWW).
    expect(useSessionStore.getState().currentSession?.status).toBe("takeover");
  });

  it("(c) terminal precedence: a stale mode event does NOT resurrect completed", async () => {
    const { sessionApi } = await import("../../api/session");
    const cbs = mockChat(sessionApi, true);

    await useSessionStore.getState().sendChat("s1", {});
    // Pin the current session to a terminal status before firing the mode event.
    const stBefore = useSessionStore.getState();
    useSessionStore.setState({
      currentSession: { ...stBefore.currentSession!, status: "completed" },
    });
    cbs.onEvent({
      type: "session_mode_changed",
      data: { to: "takeover", reason: "takeover_started", mode_revision: 99 },
    });

    // terminal kept — a stale control mode must not resurrect a terminal session.
    expect(useSessionStore.getState().currentSession?.status).toBe("completed");
  });
});

// A4-2 Debt 1 — live reducer must NOT regress a takeover / takeover_pending
// session to running on a bare content event. The bug lives ONLY in the live
// reducer's content-event fallback (resolveStatusFromEvent :272-280 →
// nextStatus); deriveStatusFromEvents filters bare content via
// SIGNAL_EVENT_TYPES and cannot reproduce it.
describe("live reducer keeps control mode sticky across a bare content event (A4-2 Debt 1)", () => {
  type ChatCallbacks = {
    onEvent: SSEEventHandler;
    onError: (error: Error) => void;
    onClose: () => void;
    onConnected: () => void;
  };

  function mockChat(api: SessionApi, simulateConnected: boolean): ChatCallbacks {
    const cbs = {} as ChatCallbacks;
    (api.chat as ReturnType<typeof vi.fn>).mockImplementation(
      (
        _sid: string,
        _params: ChatParams,
        onEvent: SSEEventHandler,
        onError: (error: Error) => void,
        onClose: () => void,
        onConnected: () => void
      ) => {
        cbs.onEvent = onEvent;
        cbs.onError = onError;
        cbs.onClose = onClose;
        cbs.onConnected = onConnected;
        if (simulateConnected && onConnected) onConnected();
        return () => {};
      }
    );
    return cbs;
  }

  beforeEach(() => {
    vi.useFakeTimers();
    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: {
        session_id: "s1",
        title: "test",
        status: "running",
        events: [],
      },
      // R2-FE-001: seed the LIST row too. The live reducer regresses BOTH the
      // open session AND the `sessions` list row (nextSessions =
      // updateSessionListStatus(..., nextStatus)), so the split-brain assertion
      // must cover the list row. Shape = ListSessionItem (no `events`).
      sessions: [
        {
          session_id: "s1",
          title: "test",
          parent_session_id: null,
          worker_type: "root",
          latest_message: "",
          latest_message_at: null,
          status: "running",
          unread_message_count: 0,
          supervisor_snapshot: null,
        },
      ],
      isChatting: false,
      chatSessionId: null,
      chatAbort: null,
      _isRecovering: false,
    });
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  function rowStatus(): string | undefined {
    return useSessionStore
      .getState()
      .sessions.find((s) => s.session_id === "s1")?.status;
  }

  it("stays in takeover when a bare message event arrives mid-takeover", async () => {
    const { sessionApi } = await import("../../api/session");
    const cbs = mockChat(sessionApi, true);

    await useSessionStore.getState().sendChat("s1", {});
    cbs.onEvent({
      type: "session_mode_changed",
      data: { to: "takeover", reason: "takeover_started", mode_revision: 5 },
    });
    expect(useSessionStore.getState().currentSession?.status).toBe("takeover");
    expect(rowStatus()).toBe("takeover");

    // Bare content event (NOT a SIGNAL_EVENT_TYPE) — must NOT regress to running.
    cbs.onEvent({
      type: "message",
      data: { role: "assistant", stream_id: "s-1", message: "", attachments: [] },
    });

    // Both the open session AND the list row must stay takeover (no split-brain).
    expect(useSessionStore.getState().currentSession?.status).toBe("takeover");
    expect(rowStatus()).toBe("takeover");
  });

  it("stays in takeover_pending when a bare message event arrives", async () => {
    const { sessionApi } = await import("../../api/session");
    const cbs = mockChat(sessionApi, true);

    await useSessionStore.getState().sendChat("s1", {});
    cbs.onEvent({
      type: "session_mode_changed",
      data: {
        to: "takeover_pending",
        reason: "takeover_requested",
        mode_revision: 5,
      },
    });
    expect(useSessionStore.getState().currentSession?.status).toBe(
      "takeover_pending"
    );
    expect(rowStatus()).toBe("takeover_pending");

    cbs.onEvent({
      type: "message",
      data: { role: "assistant", stream_id: "s-1", message: "", attachments: [] },
    });

    expect(useSessionStore.getState().currentSession?.status).toBe(
      "takeover_pending"
    );
    expect(rowStatus()).toBe("takeover_pending");
  });
});

describe("recoverSession honours session_mode_changed (backward override)", () => {
  beforeEach(() => {
    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: {
        session_id: "s1",
        title: "t",
        status: "takeover",
        events: [],
      },
      isChatting: false,
      chatSessionId: null,
      chatAbort: null,
    });
    vi.clearAllMocks();
  });

  it("a backward takeover→running mode event wins over the monotonic merge", async () => {
    const { sessionApi } = await import("../../api/session");
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [
        {
          event: "session_mode_changed",
          data: {
            to: "running",
            reason: "takeover_ended",
            mode_revision: 12,
            event_id: "e1",
          },
        },
      ],
      session_status: "takeover",
      has_more: false,
    });

    await useSessionStore.getState().recoverSession("s1");

    expect(useSessionStore.getState().currentSession!.status).toBe("running");
  });

  it("zero-event recover: terminal remote keeps completed despite a stale local mode event (R10#P1)", async () => {
    // local has a stale takeover mode event but the session actually completed
    // (the `done` event is NOT in local.events — applySSEToSession drops it).
    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: {
        session_id: "s1",
        title: "t",
        status: "completed",
        events: [
          { event: "session_mode_changed", data: { to: "takeover", reason: "takeover_started", mode_revision: 5 } },
        ],
      },
      isChatting: false,
      chatSessionId: null,
      chatAbort: null,
    });
    const { sessionApi } = await import("../../api/session");
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [],                 // zero new events → zero-event branch
      session_status: "completed",
      has_more: false,
    });

    await useSessionStore.getState().recoverSession("s1");

    expect(useSessionStore.getState().currentSession!.status).toBe("completed");
  });
});

describe("fetchSessionById honours session_mode_changed control transitions (R7)", () => {
  it("end-takeover: local takeover + remote running + merged mode(running) → running", async () => {
    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: {
        session_id: "s1", title: "t", status: "takeover",
        events: [
          { event: "control", data: { action: "started", source: "user" } },
        ],
      },
    });
    const { sessionApi } = await import("../../api/session");
    (sessionApi.getSession as ReturnType<typeof vi.fn>).mockResolvedValue({
      session_id: "s1", title: "t", status: "running",
      events: [
        { event: "session_mode_changed",
          data: { to: "running", reason: "takeover_ended", mode_revision: 12, event_id: "e1" } },
      ],
    });

    await useSessionStore.getState().fetchSessionById("s1", { silent: true });

    expect(useSessionStore.getState().currentSession!.status).toBe("running");
  });

  it("reopen: local completed + remote takeover_pending + merged mode → takeover_pending", async () => {
    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: { session_id: "s1", title: "t", status: "completed", events: [] },
    });
    const { sessionApi } = await import("../../api/session");
    (sessionApi.getSession as ReturnType<typeof vi.fn>).mockResolvedValue({
      session_id: "s1", title: "t", status: "takeover_pending",
      events: [
        { event: "control", data: { action: "reopened", source: "user" } },
        { event: "session_mode_changed",
          data: { to: "takeover_pending", reason: "takeover_reopened", mode_revision: 20, event_id: "e2" } },
      ],
    });

    await useSessionStore.getState().fetchSessionById("s1", { silent: true });

    expect(useSessionStore.getState().currentSession!.status).toBe("takeover_pending");
  });
});

describe("live finishing→running clobber (A4-0 follow-up b)", () => {
  type ChatCbs = {
    onEvent: SSEEventHandler;
    onError: (error: Error) => void;
    onClose: () => void;
    onConnected: () => void;
  };

  function captureChat(api: SessionApi): ChatCbs {
    const cbs = {} as ChatCbs;
    (api.chat as ReturnType<typeof vi.fn>).mockImplementation(
      (
        _sid: string,
        _params: ChatParams,
        onEvent: SSEEventHandler,
        onError: (error: Error) => void,
        onClose: () => void,
        onConnected: () => void
      ) => {
        cbs.onEvent = onEvent;
        cbs.onError = onError;
        cbs.onClose = onClose;
        cbs.onConnected = onConnected;
        if (onConnected) onConnected();
        return () => {};
      }
    );
    return cbs;
  }

  const compactionData = {
    compaction_id: "c1",
    level: 2,
    tokens_before: 1000,
    tokens_after: 200,
    messages_removed: 5,
  };

  beforeEach(() => {
    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: {
        session_id: "s1",
        title: "t",
        status: "running",
        events: [],
      },
      // session LIST row is a ListSessionItem (no `events` field). Seeding it
      // lets the split-brain assertion compare the list row status against the
      // open session's status after a content event.
      sessions: [
        {
          session_id: "s1",
          title: "t",
          parent_session_id: null,
          worker_type: "root",
          latest_message: "",
          latest_message_at: null,
          status: "running",
          unread_message_count: 0,
          supervisor_snapshot: null,
        },
      ],
      isChatting: false,
      chatSessionId: null,
      chatAbort: null,
      _isRecovering: false,
    });
    vi.clearAllMocks();
  });

  it("finishing 后到 compaction：开放会话仍为 finishing(核心回归)", async () => {
    const { sessionApi } = await import("../../api/session");
    const cbs = captureChat(sessionApi);
    await useSessionStore.getState().sendChat("s1", {});
    cbs.onEvent({ type: "finishing", data: {} });
    cbs.onEvent({ type: "compaction", data: compactionData });
    expect(useSessionStore.getState().currentSession?.status).toBe("finishing");
  });

  it("finishing 后到任意非信号内容事件(step)：仍为 finishing(通用回归)", async () => {
    const { sessionApi } = await import("../../api/session");
    const cbs = captureChat(sessionApi);
    await useSessionStore.getState().sendChat("s1", {});
    cbs.onEvent({ type: "finishing", data: {} });
    cbs.onEvent({
      type: "step",
      data: { id: "step-1", status: "running", description: "tail step" },
    });
    expect(useSessionStore.getState().currentSession?.status).toBe("finishing");
  });

  it("finishing 后到 compaction：列表行与开放会话一致(无 split-brain, INV-B)", async () => {
    const { sessionApi } = await import("../../api/session");
    const cbs = captureChat(sessionApi);
    await useSessionStore.getState().sendChat("s1", {});
    cbs.onEvent({ type: "finishing", data: {} });
    cbs.onEvent({ type: "compaction", data: compactionData });
    const state = useSessionStore.getState();
    const row = state.sessions.find((s) => s.session_id === "s1");
    expect(row?.status).toBe("finishing");
    expect(state.currentSession?.status).toBe("finishing");
  });

  it("timed_out 经内容事件仍保留(INV-C)", async () => {
    const { sessionApi } = await import("../../api/session");
    const cbs = captureChat(sessionApi);
    await useSessionStore.getState().sendChat("s1", {});
    cbs.onEvent({
      type: "health",
      data: { status: "terminated", reason: "watchdog terminated" },
    });
    cbs.onEvent({ type: "compaction", data: compactionData });
    expect(useSessionStore.getState().currentSession?.status).toBe("timed_out");
  });

  it("finishing→compaction→done：仍以 done 终态 completed 收敛(R1 终态优先)", async () => {
    const { sessionApi } = await import("../../api/session");
    const cbs = captureChat(sessionApi);
    await useSessionStore.getState().sendChat("s1", {});
    cbs.onEvent({ type: "finishing", data: {} });
    cbs.onEvent({ type: "compaction", data: compactionData });
    cbs.onEvent({ type: "done", data: {} });
    expect(useSessionStore.getState().currentSession?.status).toBe("completed");
  });

  it("无 finishing 的 happy path：running 经 compaction 仍 running(D1 对非 finishing 为 no-op)", async () => {
    const { sessionApi } = await import("../../api/session");
    const cbs = captureChat(sessionApi);
    await useSessionStore.getState().sendChat("s1", {});
    cbs.onEvent({ type: "compaction", data: compactionData });
    expect(useSessionStore.getState().currentSession?.status).toBe("running");
  });
});

// B1-2 (Task 13) — provisional CALLING lifecycle through the real store paths:
// live SSE reducer (applySSEToSession), reconnect recovery merge, and
// fetchSessionById full-refresh merge. Anchors the seq-watermark anti-replay
// state machine end-to-end (spec §5.3). Reuses the fake-SSE/store harness above.
describe("B1-2 provisional CALLING through store paths (Task 13)", () => {
  type ChatCbs = {
    onEvent: SSEEventHandler;
    onError: (error: Error) => void;
    onClose: () => void;
    onConnected: () => void;
  };

  function captureChat(api: SessionApi): ChatCbs {
    const cbs = {} as ChatCbs;
    (api.chat as ReturnType<typeof vi.fn>).mockImplementation(
      (
        _sid: string,
        _params: ChatParams,
        onEvent: SSEEventHandler,
        onError: (error: Error) => void,
        onClose: () => void,
        onConnected: () => void
      ) => {
        cbs.onEvent = onEvent;
        cbs.onError = onError;
        cbs.onClose = onClose;
        cbs.onConnected = onConnected;
        if (onConnected) onConnected();
        return () => {};
      }
    );
    return cbs;
  }

  function toolCallData(id: string, status: string, seq: number) {
    return {
      envelope_version: 1 as const,
      tool_call_id: id,
      status: status as "calling" | "running" | "called",
      seq,
      name: "file",
      function: "file_write",
      args: {},
      activity_description: "",
      event_id: `${id}-${status}-${seq}`,
    };
  }

  function callingIds(): string[] {
    const events =
      (useSessionStore.getState().currentSession?.events as SessionEventRecord[]) ??
      [];
    return events
      .filter((e) => e.event === "tool" && e.data?.status === "calling")
      .map((e) => e.data?.tool_call_id as string);
  }

  beforeEach(() => {
    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: {
        session_id: "s1",
        title: "test",
        status: "running",
        events: [],
      },
      isChatting: false,
      chatSessionId: null,
      chatAbort: null,
      _isRecovering: false,
    });
    vi.clearAllMocks();
  });

  it("防复活 (recovery): a replayed CALLING below the watermark cannot resurrect a pruned card", async () => {
    const { sessionApi } = await import("../../api/session");
    const cbs = captureChat(sessionApi);
    await useSessionStore.getState().sendChat("s1", {});

    // live: calling(10) → provisional card; message(20) closes the turn & prunes it.
    cbs.onEvent({ type: "tool", data: toolCallData("a", "calling", 10) });
    expect(callingIds()).toEqual(["a"]);
    cbs.onEvent({
      type: "message",
      data: { role: "assistant", message: "text", event_id: "m-20", seq: 20, attachments: [] },
    });
    expect(callingIds()).toEqual([]);

    // reconnect: recovery stream replays the same calling(10). Must NOT resurrect.
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [{ event: "tool", data: toolCallData("a", "calling", 10) }],
      session_status: "running",
      has_more: false,
      last_seq: 20,
    });
    await useSessionStore.getState().recoverSession("s1");

    expect(callingIds()).toEqual([]);
  });

  it("recovered done prune-all: recovery stream ending in done drops residual provisional", async () => {
    const { sessionApi } = await import("../../api/session");
    const cbs = captureChat(sessionApi);
    await useSessionStore.getState().sendChat("s1", {});

    // live: calling(10) — still provisional (no close signal seen live).
    cbs.onEvent({ type: "tool", data: toolCallData("a", "calling", 10) });
    expect(callingIds()).toEqual(["a"]);

    // reconnect: recovery = [calling(10), done(30)] → done prunes all provisional.
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [
        { event: "tool", data: toolCallData("a", "calling", 10) },
        { event: "done", data: { event_id: "d-30", seq: 30 } },
      ],
      session_status: "completed",
      has_more: false,
      last_seq: 30,
    });
    await useSessionStore.getState().recoverSession("s1");

    expect(callingIds()).toEqual([]);
  });

  it("防复活 (refetch): full refresh cannot resurrect a pruned card and preserves the local watermark", async () => {
    const { sessionApi } = await import("../../api/session");
    const cbs = captureChat(sessionApi);
    await useSessionStore.getState().sendChat("s1", {});

    // live: calling(10) → message(20) closes & prunes, local watermark = 20.
    cbs.onEvent({ type: "tool", data: toolCallData("a", "calling", 10) });
    cbs.onEvent({
      type: "message",
      data: { role: "assistant", message: "text", event_id: "m-20", seq: 20, attachments: [] },
    });
    expect(callingIds()).toEqual([]);
    expect(
      useSessionStore.getState().currentSession?.provisional_prune_watermark
    ).toBe(20);

    // fetchSessionById full refresh: remote replays the stale calling(10).
    (sessionApi.getSession as ReturnType<typeof vi.fn>).mockResolvedValue({
      session_id: "s1",
      title: "test",
      status: "running",
      events: [
        { event: "tool", data: toolCallData("a", "calling", 10) },
        { event: "message", data: { role: "assistant", message: "text", event_id: "m-20", seq: 20 } },
      ],
    });
    await useSessionStore.getState().fetchSessionById("s1", { silent: true });

    expect(callingIds()).toEqual([]);
    // Local watermark survives the refresh (remote payload has no watermark keys).
    expect(
      useSessionStore.getState().currentSession?.provisional_prune_watermark
    ).toBe(20);
  });

  it("live error 边界 (scenario vi): error marks the boundary so the retry's new CALLING prunes the orphan", async () => {
    const { sessionApi } = await import("../../api/session");
    const cbs = captureChat(sessionApi);
    await useSessionStore.getState().sendChat("s1", {});

    cbs.onEvent({ type: "tool", data: toolCallData("a", "calling", 10) });
    // seq lives on BaseEvent end-to-end; ErrorEvent's wire type doesn't declare
    // it (out of this task's types.ts scope), so cast for the test literal.
    cbs.onEvent({
      type: "error",
      data: { error: "boom", event_id: "e-15", seq: 15 },
    } as unknown as SSEEventData);
    // error does not prune (Done backstop owns that) — but advances the watermark.
    expect(callingIds()).toEqual(["a"]);
    cbs.onEvent({ type: "tool", data: toolCallData("a2", "calling", 20) });

    expect(callingIds()).toEqual(["a2"]);
    expect(
      useSessionStore.getState().currentSession?.provisional_prune_watermark ?? 0
    ).toBeGreaterThanOrEqual(15);
  });

  it("recovery 升级边界防复活 (scenario v-c): a stale different-id CALLING below the upgrade watermark is dropped", async () => {
    const { sessionApi } = await import("../../api/session");
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [
        { event: "tool", data: toolCallData("a", "calling", 10) },
        { event: "tool", data: toolCallData("a", "called", 11) },
        { event: "tool", data: toolCallData("z", "calling", 9) },
      ],
      session_status: "running",
      has_more: false,
      last_seq: 11,
    });

    await useSessionStore.getState().recoverSession("s1");

    const events =
      (useSessionStore.getState().currentSession?.events as SessionEventRecord[]) ??
      [];
    expect(events.some((e) => e.data?.tool_call_id === "z")).toBe(false);
    const aCard = events.find((e) => e.data?.tool_call_id === "a");
    expect(aCard?.data?.status).toBe("called");
  });

  it("R3-FIX (recovery cursor race): a stale same-id CALLING in the recovery stream does NOT erase a local CALLED (recovery≡live)", async () => {
    const { sessionApi } = await import("../../api/session");
    const cbs = captureChat(sessionApi);
    await useSessionStore.getState().sendChat("s1", {});

    // live: calling(a, 10) then called(a, 11) — the local card is CALLED and the
    // watermark has advanced to 11 (the tool called branch pushes it).
    cbs.onEvent({ type: "tool", data: toolCallData("a", "calling", 10) });
    cbs.onEvent({ type: "tool", data: toolCallData("a", "called", 11) });
    const cardStatus = () => {
      const events =
        (useSessionStore.getState().currentSession
          ?.events as SessionEventRecord[]) ?? [];
      return events.find((e) => e.data?.tool_call_id === "a")?.data?.status;
    };
    expect(cardStatus()).toBe("called");
    expect(
      useSessionStore.getState().currentSession?.provisional_prune_watermark
    ).toBe(11);

    // reconnect: recovery response was built before the FE's CALLED(11) landed,
    // so it replays the stale CALLING(a, 10). Recovery events are the SECOND
    // merge argument (recoverSession passes (local, recovered)) — pre-fix this
    // overwrote the local CALLED and the fold then dropped the CALLING as an
    // anti-replay orphan, making card `a` vanish. The seq-compare guard keeps the
    // higher-seq local CALLED.
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [{ event: "tool", data: toolCallData("a", "calling", 10) }],
      session_status: "running",
      has_more: false,
      last_seq: 11,
    });
    await useSessionStore.getState().recoverSession("s1");

    // The `a` card REMAINS with status "called" and the watermark stays 11.
    expect(cardStatus()).toBe("called");
    expect(
      useSessionStore.getState().currentSession?.provisional_prune_watermark
    ).toBe(11);
  });

  it("R3-FIX (refetch direction twin): a stale remote CALLING cannot downgrade a newer local CALLED", async () => {
    const { sessionApi } = await import("../../api/session");
    const cbs = captureChat(sessionApi);
    await useSessionStore.getState().sendChat("s1", {});

    // live: calling(a, 10) then called(a, 11) — local card CALLED, watermark 11.
    cbs.onEvent({ type: "tool", data: toolCallData("a", "calling", 10) });
    cbs.onEvent({ type: "tool", data: toolCallData("a", "called", 11) });
    const cardStatus = () => {
      const events =
        (useSessionStore.getState().currentSession
          ?.events as SessionEventRecord[]) ?? [];
      return events.find((e) => e.data?.tool_call_id === "a")?.data?.status;
    };
    expect(cardStatus()).toBe("called");

    // fetchSessionById full refresh: the remote payload (FIRST merge argument at
    // this site) carries a stale CALLING(a, 10). Here LOCAL normally wins by
    // arg-order, but the guard is direction-agnostic: even in the direction where
    // remote is the first arg, the higher-seq CALLED must survive.
    (sessionApi.getSession as ReturnType<typeof vi.fn>).mockResolvedValue({
      session_id: "s1",
      title: "test",
      status: "running",
      events: [{ event: "tool", data: toolCallData("a", "calling", 10) }],
    });
    await useSessionStore.getState().fetchSessionById("s1", { silent: true });

    expect(cardStatus()).toBe("called");
  });

  it("refetch 保留开放 provisional (scenario vii): an open turn's own CALLING is not anti-replay pruned", async () => {
    const { sessionApi } = await import("../../api/session");
    const cbs = captureChat(sessionApi);
    await useSessionStore.getState().sendChat("s1", {});

    // live: calling(a, 10) with NO close signal → open turn, watermark = 10.
    cbs.onEvent({ type: "tool", data: toolCallData("a", "calling", 10) });
    expect(callingIds()).toEqual(["a"]);
    expect(
      useSessionStore.getState().currentSession?.provisional_turn_closed
    ).toBe(false);

    // fetchSessionById full refresh: remote = [calling(a, 10)] → fold must keep a.
    (sessionApi.getSession as ReturnType<typeof vi.fn>).mockResolvedValue({
      session_id: "s1",
      title: "test",
      status: "running",
      events: [{ event: "tool", data: toolCallData("a", "calling", 10) }],
    });
    await useSessionStore.getState().fetchSessionById("s1", { silent: true });

    expect(callingIds()).toEqual(["a"]);
  });
});
