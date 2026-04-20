import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  deriveStatusFromEvents,
  pickMoreAdvancedStatus,
  useSessionStore,
} from "../session-store";
import type { SessionEventRecord } from "../session-store";
import type { sessionApi } from "../../api/session";
import type { ChatParams, SSEEventHandler } from "../../api/types";

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

    expect(sessionApi.getEventsSince).toHaveBeenCalledWith("s1", undefined);
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
});
