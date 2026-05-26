import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/api/session", () => ({
  sessionApi: {
    getSessions: vi.fn(),
    streamSessions: vi.fn(),
    createSession: vi.fn(),
    getSession: vi.fn(),
    getSessionFiles: vi.fn(),
    chat: vi.fn(),
    retryFromSuspend: vi.fn(),
    stopSession: vi.fn(),
    deleteSession: vi.fn(),
    clearUnreadMessageCount: vi.fn(),
    viewFile: vi.fn(),
    viewShell: vi.fn(),
    downloadSandboxFile: vi.fn(),
  },
}));

vi.mock("@/lib/api/file", () => ({
  fileApi: {
    uploadFile: vi.fn(),
    downloadFile: vi.fn(),
  },
}));

vi.mock("@/lib/api/session-compaction", () => ({
  fetchCompactionList: vi.fn(async () => []),
}));

import { fileApi } from "@/lib/api/file";
import { sessionApi } from "@/lib/api/session";
import type { ListSessionItem, Session, SupervisorSnapshot } from "@/lib/api/types";
import {
  useFilteredSessionsForList,
  useSessionStore,
} from "@/lib/store/session-store";
import { useUIStore } from "@/lib/store/ui-store";

const mockedSessionApi = vi.mocked(sessionApi, { deep: true });
const mockedFileApi = vi.mocked(fileApi, { deep: true });

function buildSession(overrides?: Partial<Session>): Session {
  return {
    session_id: "s1",
    title: "会话",
    status: "running",
    events: [],
    ...overrides,
  };
}

const backgroundSnapshot: SupervisorSnapshot = {
  execution_mode: "background",
  execution_phase: "running",
  background_reason: "explicit",
  expires_at: null,
  retry_budget_remaining: 2,
  suspended_reason: null,
  terminal_reason: null,
  last_progress_at: null,
  is_alive: true,
  cancellation_state: "none",
};

function buildListSession(overrides?: Partial<ListSessionItem>): ListSessionItem {
  return {
    session_id: "s-list",
    title: "列表会话",
    parent_session_id: null,
    worker_type: "root",
    latest_message: "处理中",
    latest_message_at: "2026-05-12T00:00:00.000Z",
    status: "running",
    unread_message_count: 0,
    ...overrides,
  };
}

describe("session-store", () => {
  beforeEach(() => {
    useSessionStore.getState().reset();
    useUIStore.getState().reset();
    vi.clearAllMocks();

    mockedSessionApi.getSessions.mockResolvedValue([]);
    mockedSessionApi.streamSessions.mockReturnValue(() => {});
    mockedSessionApi.createSession.mockResolvedValue({ session_id: "s1" });
    mockedSessionApi.getSession.mockResolvedValue(buildSession());
    mockedSessionApi.getSessionFiles.mockResolvedValue({ files: [] });
    mockedSessionApi.chat.mockReturnValue(() => {});
    mockedSessionApi.retryFromSuspend.mockResolvedValue({
      status: "running",
      request_status: "resumed",
      retry_budget_remaining: 1,
      expires_at: null,
    });
    mockedSessionApi.stopSession.mockResolvedValue();
    mockedSessionApi.deleteSession.mockResolvedValue();
    mockedSessionApi.clearUnreadMessageCount.mockResolvedValue();
  });

  it("createSession 成功后会显示成功提示", async () => {
    await useSessionStore.getState().createSession();

    expect(useUIStore.getState().message).toEqual({
      type: "success",
      text: "新任务已创建",
    });
  });

  it("fetchSessions 保留列表项 supervisor_snapshot", async () => {
    mockedSessionApi.getSessions.mockResolvedValue([
      buildListSession({ supervisor_snapshot: backgroundSnapshot }),
    ]);

    await useSessionStore.getState().fetchSessions();

    expect(useSessionStore.getState().sessions[0]?.supervisor_snapshot).toEqual(
      backgroundSnapshot
    );
  });

  it("useFilteredSessionsForList 在 sessions 未变化时保持引用稳定", () => {
    const rootSession = buildListSession({ session_id: "root" });
    const childSession = buildListSession({
      session_id: "child",
      parent_session_id: "root",
      worker_type: "subagent",
    });
    useSessionStore.setState({
      sessions: [rootSession, childSession],
      isLoadingSessions: false,
    });

    const { result } = renderHook(() => useFilteredSessionsForList());
    const first = result.current;

    useSessionStore.setState({ isLoadingSessions: true });

    expect(result.current).toBe(first);
    expect(result.current).toEqual([rootSession]);
  });

  it("streamSessions 保留列表项 supervisor_snapshot", () => {
    mockedSessionApi.streamSessions.mockImplementation((onEvent) => {
      onEvent({
        type: "sessions",
        data: {
          sessions: [buildListSession({ supervisor_snapshot: backgroundSnapshot })],
        },
      });
      return () => {};
    });

    useSessionStore.getState().streamSessions();

    expect(useSessionStore.getState().sessions[0]?.supervisor_snapshot).toEqual(
      backgroundSnapshot
    );
  });

  it("fetchSessionById 不会覆盖本地已流式追加的事件", async () => {
    useSessionStore.setState({
      currentSession: buildSession({
        session_id: "s1",
        events: [
          {
            event: "message",
            data: { event_id: "evt-local", role: "assistant", message: "local" },
          },
        ],
      }),
    });

    mockedSessionApi.getSession.mockResolvedValue(
      buildSession({
        session_id: "s1",
        events: [],
      })
    );

    await useSessionStore.getState().fetchSessionById("s1");

    const events = useSessionStore.getState().currentSession?.events ?? [];
    expect(events).toHaveLength(1);
    expect(events[0]?.data?.event_id).toBe("evt-local");
  });

  it("fetchSessionById 对非当前激活会话的过期响应不应覆盖 currentSession", async () => {
    useSessionStore.setState({
      currentSession: buildSession({
        session_id: "s-active",
        events: [
          {
            event: "message",
            data: { event_id: "evt-active", role: "assistant", message: "active" },
          },
        ],
      }),
      activeSessionId: "s-active",
    });

    mockedSessionApi.getSession.mockResolvedValue(
      buildSession({
        session_id: "s-stale",
        events: [
          {
            event: "message",
            data: { event_id: "evt-stale", role: "assistant", message: "stale" },
          },
        ],
      })
    );

    await useSessionStore.getState().fetchSessionById("s-stale");

    const current = useSessionStore.getState().currentSession;
    expect(current?.session_id).toBe("s-active");
    expect(current?.events?.[0]?.data?.event_id).toBe("evt-active");
  });

  it("sendChat 在 currentSession 为空时也能接收首条流式事件", async () => {
    mockedSessionApi.chat.mockImplementation((_sessionId, _params, onEvent) => {
      onEvent({
        type: "message",
        data: {
          event_id: "evt-1",
          created_at: Math.floor(Date.now() / 1000),
          role: "assistant",
          message: "hello",
          attachments: [],
        },
      });
      onEvent({
        type: "done",
        data: {
          event_id: "evt-done",
          created_at: Math.floor(Date.now() / 1000),
        },
      });
      return () => {};
    });

    await useSessionStore.getState().sendChat("s-new", { message: "hi" });

    const current = useSessionStore.getState().currentSession;
    expect(current?.session_id).toBe("s-new");
    expect(current?.events).toHaveLength(1);
    expect(current?.events[0]?.event).toBe("message");
  });

  it("title 事件应更新会话标题但不渲染为会话事件", async () => {
    mockedSessionApi.chat.mockImplementation((_sessionId, _params, onEvent) => {
      onEvent({
        type: "title",
        data: {
          event_id: "evt-title",
          created_at: Math.floor(Date.now() / 1000),
          title: "新标题",
        },
      });
      onEvent({
        type: "message",
        data: {
          event_id: "evt-msg",
          created_at: Math.floor(Date.now() / 1000),
          role: "assistant",
          message: "ok",
          attachments: [],
        },
      });
      onEvent({
        type: "done",
        data: { event_id: "evt-done", created_at: Math.floor(Date.now() / 1000) },
      });
      return () => {};
    });

    await useSessionStore.getState().sendChat("s-title", { message: "hi" });

    const current = useSessionStore.getState().currentSession;
    expect(current?.title).toBe("新标题");
    expect(current?.events).toHaveLength(1);
    expect(current?.events[0]?.event).toBe("message");
  });

  it("同一 tool_call_id 的 calling/called 事件应进行替换更新", async () => {
    mockedSessionApi.chat.mockImplementation((_sessionId, _params, onEvent) => {
      onEvent({
        type: "tool",
        data: {
          envelope_version: 1,
          event_id: "evt-tool-1",
          created_at: Math.floor(Date.now() / 1000),
          tool_call_id: "tool-123",
          name: "file",
          function: "write_file",
          args: { filepath: "/home/ubuntu/a.txt" },
          status: "calling",
          activity_description: "Writing file",
        },
      });
      onEvent({
        type: "tool",
        data: {
          envelope_version: 1,
          event_id: "evt-tool-2",
          created_at: Math.floor(Date.now() / 1000),
          tool_call_id: "tool-123",
          name: "file",
          function: "write_file",
          args: { filepath: "/home/ubuntu/a.txt" },
          status: "called",
          activity_description: "Writing file",
        },
      });
      onEvent({
        type: "done",
        data: { event_id: "evt-done", created_at: Math.floor(Date.now() / 1000) },
      });
      return () => {};
    });

    await useSessionStore.getState().sendChat("s-tool", { message: "hi" });

    const events = useSessionStore.getState().currentSession?.events ?? [];
    const toolEvents = events.filter((item) => item.event === "tool");
    expect(toolEvents).toHaveLength(1);
    expect(toolEvents[0]?.data?.status).toBe("called");
  });

  it("同一 stream_id 的消息分片应持续覆盖更新为最新内容", async () => {
    mockedSessionApi.chat.mockImplementation((_sessionId, _params, onEvent) => {
      onEvent({
        type: "message",
        data: {
          event_id: "evt-msg-1",
          created_at: Math.floor(Date.now() / 1000),
          role: "assistant",
          message: "流式",
          stream_id: "stream-1",
          partial: true,
          attachments: [],
        },
      });
      onEvent({
        type: "message",
        data: {
          event_id: "evt-msg-2",
          created_at: Math.floor(Date.now() / 1000),
          role: "assistant",
          message: "流式输出完成",
          stream_id: "stream-1",
          partial: false,
          attachments: [],
        },
      });
      onEvent({
        type: "done",
        data: { event_id: "evt-done", created_at: Math.floor(Date.now() / 1000) },
      });
      return () => {};
    });

    await useSessionStore.getState().sendChat("s-stream", { message: "hi" });

    const events = useSessionStore.getState().currentSession?.events ?? [];
    const messageEvents = events.filter((item) => item.event === "message");
    expect(messageEvents).toHaveLength(1);
    expect(messageEvents[0]?.data?.message).toBe("流式输出完成");
  });

  it("step 事件应实时同步更新 plan 中对应步骤状态", async () => {
    mockedSessionApi.chat.mockImplementation((_sessionId, _params, onEvent) => {
      onEvent({
        type: "plan",
        data: {
          event_id: "evt-plan",
          created_at: Math.floor(Date.now() / 1000),
          steps: [
            { id: "step-1", description: "第一步", status: "pending" },
            { id: "step-2", description: "第二步", status: "pending" },
          ],
        },
      });
      onEvent({
        type: "step",
        data: {
          event_id: "evt-step",
          created_at: Math.floor(Date.now() / 1000),
          id: "step-1",
          description: "第一步",
          status: "completed",
        },
      });
      onEvent({
        type: "done",
        data: { event_id: "evt-done", created_at: Math.floor(Date.now() / 1000) },
      });
      return () => {};
    });

    await useSessionStore.getState().sendChat("s-plan", { message: "hi" });

    const events = useSessionStore.getState().currentSession?.events ?? [];
    const planEvent = events.find((item) => item.event === "plan");
    const steps = (planEvent?.data?.steps as Array<Record<string, unknown>>) || [];
    const step1 = steps.find((step) => step.id === "step-1");
    expect(step1?.status).toBe("completed");
  });

  it("LLM 暂时失败后继续产出时，应移除可恢复错误事件", async () => {
    mockedSessionApi.chat.mockImplementation((_sessionId, _params, onEvent) => {
      onEvent({
        type: "error",
        data: {
          event_id: "evt-error",
          created_at: Math.floor(Date.now() / 1000),
          error:
            "调用语言模型失败: 调用OpenAI客户端向LLM发起请求出错",
        },
      });
      onEvent({
        type: "message",
        data: {
          event_id: "evt-msg",
          created_at: Math.floor(Date.now() / 1000),
          role: "assistant",
          message: "任务继续执行",
          attachments: [],
        },
      });
      onEvent({
        type: "done",
        data: { event_id: "evt-done", created_at: Math.floor(Date.now() / 1000) },
      });
      return () => {};
    });

    await useSessionStore.getState().sendChat("s-recover", { message: "go" });

    const events = useSessionStore.getState().currentSession?.events ?? [];
    expect(events.some((item) => item.event === "error")).toBe(false);
    expect(events.some((item) => item.event === "message")).toBe(true);
  });

  it("fetchSessionById 对 running 会话应自动续流并携带最新 event_id", async () => {
    mockedSessionApi.getSession.mockResolvedValue(
      buildSession({
        session_id: "s-running",
        status: "running",
        events: [
          {
            event: "message",
            data: { event_id: "evt-100", role: "assistant", message: "old" },
          },
        ],
      })
    );

    await useSessionStore.getState().fetchSessionById("s-running");

    expect(mockedSessionApi.chat).toHaveBeenCalledTimes(1);
    expect(mockedSessionApi.chat.mock.calls[0]?.[0]).toBe("s-running");
    expect(mockedSessionApi.chat.mock.calls[0]?.[1]).toMatchObject({
      event_id: "evt-100",
    });
  });

  it("fetchSessionById 对挂起后台会话不应自动续流", async () => {
    mockedSessionApi.getSession.mockResolvedValue(
      buildSession({
        session_id: "s-bg-suspended",
        status: "running",
        supervisor_snapshot: {
          ...backgroundSnapshot,
          execution_phase: "suspended",
          is_alive: false,
        },
        events: [
          {
            event: "message",
            data: { event_id: "evt-100", role: "assistant", message: "old" },
          },
        ],
      })
    );

    await useSessionStore.getState().fetchSessionById("s-bg-suspended");

    expect(mockedSessionApi.chat).not.toHaveBeenCalled();
  });

  it("fetchSessionById 应信任详情接口返回的当前 supervisor_snapshot", async () => {
    mockedSessionApi.getSession.mockResolvedValue(
      buildSession({
        session_id: "s-bg-suspended",
        status: "running",
        supervisor_snapshot: {
          ...backgroundSnapshot,
          execution_phase: "suspended",
          suspended_reason: "bg_idle_timeout",
          is_alive: false,
        },
        events: [
          {
            event: "execution_state_changed",
            data: {
              event_id: "evt-auto-degrade",
              seq: 10,
              payload: {
                execution_mode: "background",
                execution_phase: "running",
                background_reason: "auto_degrade",
                expires_at: "2026-05-11T08:30:00Z",
                retry_budget_remaining: 2,
                suspended_reason: null,
                terminal_reason: null,
              },
            },
          },
        ],
      })
    );

    await useSessionStore.getState().fetchSessionById("s-bg-suspended");

    expect(
      useSessionStore.getState().currentSession?.supervisor_snapshot
        ?.execution_phase
    ).toBe("suspended");
    expect(
      useSessionStore.getState().currentSession?.supervisor_snapshot
        ?.suspended_reason
    ).toBe("bg_idle_timeout");
    expect(mockedSessionApi.chat).not.toHaveBeenCalled();
  });

  it("fetchSessionById 应让详情 snapshot 越过本地更高事件游标", async () => {
    useSessionStore.setState({
      activeSessionId: "s-bg-suspended",
      currentSession: buildSession({
        session_id: "s-bg-suspended",
        status: "running",
        last_seq: 99,
        supervisor_snapshot: backgroundSnapshot,
        events: [
          {
            event: "message",
            data: {
              event_id: "evt-local-partial",
              seq: 99,
              role: "assistant",
            },
          },
        ],
      }),
    });
    mockedSessionApi.getSession.mockResolvedValue(
      buildSession({
        session_id: "s-bg-suspended",
        status: "running",
        last_seq: 98,
        supervisor_snapshot: {
          ...backgroundSnapshot,
          execution_phase: "suspended",
          suspended_reason: "bg_idle_timeout",
          is_alive: false,
        },
        events: [
          {
            event: "message",
            data: { event_id: "evt-persisted", seq: 98, role: "assistant" },
          },
        ],
      })
    );

    await useSessionStore.getState().fetchSessionById("s-bg-suspended");

    expect(useSessionStore.getState().currentSession?.last_seq).toBe(99);
    expect(
      useSessionStore.getState().currentSession?.supervisor_snapshot
        ?.execution_phase
    ).toBe("suspended");
    expect(mockedSessionApi.chat).not.toHaveBeenCalled();
  });

  it("retryFromSuspend 成功后刷新列表和当前详情", async () => {
    useSessionStore.setState({
      activeSessionId: "s-bg-suspended",
      currentSession: buildSession({
        session_id: "s-bg-suspended",
        status: "running",
        supervisor_snapshot: {
          ...backgroundSnapshot,
          execution_phase: "suspended",
          is_alive: false,
        },
      }),
      isChatting: true,
      chatSessionId: "other-session",
    });
    mockedSessionApi.getSessions.mockResolvedValue([
      buildListSession({ session_id: "s-bg-suspended" }),
    ]);
    mockedSessionApi.getSession.mockResolvedValue(
      buildSession({
        session_id: "s-bg-suspended",
        status: "running",
        supervisor_snapshot: backgroundSnapshot,
      })
    );

    await useSessionStore.getState().retryFromSuspend("s-bg-suspended");

    expect(mockedSessionApi.retryFromSuspend).toHaveBeenCalledWith(
      "s-bg-suspended"
    );
    expect(mockedSessionApi.getSessions).toHaveBeenCalled();
    expect(mockedSessionApi.getSession).toHaveBeenCalledWith("s-bg-suspended");
  });

  it("retryFromSuspend 成功后立即用接口结果恢复当前详情 snapshot", async () => {
    useSessionStore.setState({
      activeSessionId: "s-bg-suspended",
      currentSession: buildSession({
        session_id: "s-bg-suspended",
        status: "running",
        last_seq: 99,
        supervisor_snapshot: {
          ...backgroundSnapshot,
          execution_phase: "suspended",
          is_alive: false,
        },
      }),
    });
    mockedSessionApi.getSessions.mockResolvedValue([
      buildListSession({ session_id: "s-bg-suspended" }),
    ]);
    mockedSessionApi.getSession.mockResolvedValue(
      buildSession({
        session_id: "s-bg-suspended",
        status: "running",
        last_seq: 1,
        supervisor_snapshot: backgroundSnapshot,
      })
    );

    await useSessionStore.getState().retryFromSuspend("s-bg-suspended");

    expect(
      useSessionStore.getState().currentSession?.supervisor_snapshot
        ?.execution_phase
    ).toBe("running");
    expect(
      useSessionStore.getState().currentSession?.supervisor_snapshot?.is_alive
    ).toBe(true);
  });

  it("fetchSessionById 对 takeover 会话不应自动续流", async () => {
    mockedSessionApi.getSession.mockResolvedValue(
      buildSession({
        session_id: "s-takeover",
        status: "takeover",
        events: [
          {
            event: "control",
            data: {
              event_id: "evt-control",
              action: "started",
            },
          },
        ],
      })
    );

    await useSessionStore.getState().fetchSessionById("s-takeover");

    expect(mockedSessionApi.chat).not.toHaveBeenCalled();
  });

  it("sendChat 收到 wait 事件后应结束流并将状态置为 waiting", async () => {
    mockedSessionApi.chat.mockImplementation((_sessionId, _params, onEvent, _onError, onClose) => {
      onEvent({
        type: "wait",
        data: {
          event_id: "evt-wait",
          created_at: Math.floor(Date.now() / 1000),
        },
      });
      onClose?.();
      return () => {};
    });

    await useSessionStore.getState().sendChat("s-wait", { message: "继续" });

    const state = useSessionStore.getState();
    expect(state.isChatting).toBe(false);
    expect(state.chatAbort).toBeNull();
    expect(state.currentSession?.status).toBe("waiting");
  });

  it("sendChat 收到 control.started 事件后应进入 takeover 状态并结束流", async () => {
    mockedSessionApi.chat.mockImplementation((_sessionId, _params, onEvent, _onError, onClose) => {
      onEvent({
        type: "control",
        data: {
          event_id: "evt-control-start",
          created_at: Math.floor(Date.now() / 1000),
          action: "started",
          source: "system",
          request_status: "started",
          takeover_id: "tk_001",
        },
      });
      onClose?.();
      return () => {};
    });

    await useSessionStore.getState().sendChat("s-control-start", { message: "接管" });

    const state = useSessionStore.getState();
    expect(state.isChatting).toBe(false);
    expect(state.chatAbort).toBeNull();
    expect(state.currentSession?.status).toBe("takeover");
  });

  it("sendChat 收到 control.rejected(cancel_timeout) 事件后应回到 running 状态", async () => {
    mockedSessionApi.chat.mockImplementation((_sessionId, _params, onEvent, _onError, onClose) => {
      onEvent({
        type: "control",
        data: {
          event_id: "evt-control-rejected",
          created_at: Math.floor(Date.now() / 1000),
          action: "rejected",
          source: "system",
          reason: "cancel_timeout",
          request_status: "rejected",
          takeover_id: "tk_002",
        },
      });
      onClose?.();
      return () => {};
    });

    await useSessionStore.getState().sendChat("s-control-rejected", { message: "接管" });

    const state = useSessionStore.getState();
    expect(state.currentSession?.status).toBe("running");
  });

  it("sendChat 收到未知 control.action 时应保持状态并输出告警", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    mockedSessionApi.chat.mockImplementation((_sessionId, _params, onEvent, _onError, onClose) => {
      onEvent({
        type: "control",
        data: {
          event_id: "evt-control-unknown",
          created_at: Math.floor(Date.now() / 1000),
          action: "unknown_action",
          source: "system",
        },
      });
      onClose?.();
      return () => {};
    });

    await useSessionStore.getState().sendChat("s-control-unknown", { message: "接管" });

    const state = useSessionStore.getState();
    expect(state.currentSession?.status).toBe("running");
    expect(warnSpy).toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it("sendChat 终态事件应同步更新 sessions 列表状态", async () => {
    useSessionStore.setState({
      sessions: [
        {
          session_id: "s-done",
          title: "会话",
          parent_session_id: null,
          worker_type: "root",
          latest_message: "",
          latest_message_at: null,
          status: "running",
          unread_message_count: 0,
        },
      ],
    });

    mockedSessionApi.chat.mockImplementation((_sessionId, _params, onEvent, _onError, onClose) => {
      onEvent({
        type: "done",
        data: { event_id: "evt-done", created_at: Math.floor(Date.now() / 1000) },
      });
      onClose?.();
      return () => {};
    });

    await useSessionStore.getState().sendChat("s-done", { message: "完成" });

    expect(useSessionStore.getState().sessions[0]?.status).toBe("completed");
  });

  it("stopSession 后应乐观更新会话状态并停止当前流", async () => {
    const abortMock = vi.fn();
    useSessionStore.setState({
      sessions: [
        {
          session_id: "s-stop",
          title: "会话",
          parent_session_id: null,
          worker_type: "root",
          latest_message: "",
          latest_message_at: null,
          status: "running",
          unread_message_count: 0,
        },
      ],
      currentSession: buildSession({ session_id: "s-stop", status: "running" }),
      isChatting: true,
      chatSessionId: "s-stop",
      chatAbort: abortMock,
    });

    await useSessionStore.getState().stopSession("s-stop");

    const state = useSessionStore.getState();
    expect(mockedSessionApi.stopSession).toHaveBeenCalledWith("s-stop");
    expect(abortMock).toHaveBeenCalledTimes(1);
    expect(state.isChatting).toBe(false);
    expect(state.chatSessionId).toBeNull();
    expect(state.currentSession?.status).toBe("completed");
    expect(state.sessions[0]?.status).toBe("completed");
  });

  it("isSessionStreaming 仅对当前流会话返回 true", () => {
    useSessionStore.setState({
      isChatting: true,
      chatSessionId: "s-active",
    });

    expect(useSessionStore.getState().isSessionStreaming("s-active")).toBe(true);
    expect(useSessionStore.getState().isSessionStreaming("s-other")).toBe(false);
  });

  it("fetchSessionById 在 silent 模式下不应切换加载态", async () => {
    let resolveSession: ((session: Session) => void) | null = null;
    mockedSessionApi.getSession.mockImplementation(
      () =>
        new Promise<Session>((resolve) => {
          resolveSession = resolve;
        })
    );

    const request = useSessionStore
      .getState()
      .fetchSessionById("s1", { silent: true });

    expect(useSessionStore.getState().isLoadingCurrentSession).toBe(false);

    resolveSession?.(
      buildSession({
        session_id: "s1",
        status: "running",
        events: [],
      })
    );
    await request;

    expect(useSessionStore.getState().isLoadingCurrentSession).toBe(false);
  });

  it("fetchSessionById 远端无变化时不应替换 currentSession 引用", async () => {
    const current = buildSession({
      session_id: "s1",
      title: "same",
      status: "completed",
      events: [
        {
          event: "message",
          data: {
            event_id: "evt-1",
            role: "assistant",
            message: "hello",
          },
        },
      ],
    });

    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: current,
    });

    mockedSessionApi.getSession.mockResolvedValue(
      buildSession({
        session_id: "s1",
        title: "same",
        status: "completed",
        events: [
          {
            event: "message",
            data: {
              event_id: "evt-1",
              role: "assistant",
              message: "hello",
            },
          },
        ],
      })
    );

    await useSessionStore.getState().fetchSessionById("s1", { silent: true });

    expect(useSessionStore.getState().currentSession).toBe(current);
  });

  describe("uploadFile with transfer options", () => {
    it("forwards onProgress and signal to fileApi", async () => {
      const mockResult = { id: "f1", filename: "a.txt", filepath: "", key: "", extension: ".txt", mime_type: "text/plain", size: 10 };
      mockedFileApi.uploadFile.mockResolvedValue(mockResult);
      const onProgress = vi.fn();
      const controller = new AbortController();
      const file = new File(["x"], "a.txt");

      await useSessionStore.getState().uploadFile(file, "s1", { onProgress, signal: controller.signal });

      expect(mockedFileApi.uploadFile).toHaveBeenCalledWith(
        expect.objectContaining({ file, session_id: "s1", onProgress, signal: controller.signal })
      );
    });
  });

  describe("downloadFile with transfer options", () => {
    it("forwards onProgress and signal to fileApi", async () => {
      mockedFileApi.downloadFile.mockResolvedValue(new Blob(["data"]));
      const onProgress = vi.fn();
      const controller = new AbortController();

      await useSessionStore.getState().downloadFile("f1", { onProgress, signal: controller.signal });

      expect(mockedFileApi.downloadFile).toHaveBeenCalledWith("f1", { onProgress, signal: controller.signal });
    });
  });

  describe("downloadSandboxFile", () => {
    it("forwards to sessionApi with options", async () => {
      mockedSessionApi.downloadSandboxFile.mockResolvedValue(new Blob(["data"]));
      const onProgress = vi.fn();

      await useSessionStore.getState().downloadSandboxFile("s1", "/path/file.txt", { onProgress });

      expect(mockedSessionApi.downloadSandboxFile).toHaveBeenCalledWith("s1", "/path/file.txt", { onProgress });
    });
  });

  describe("D5: health event reducer", () => {
    it("TERMINATED health event pins session status to timed_out across subsequent done", async () => {
      mockedSessionApi.chat.mockImplementation((_sessionId, _params, onEvent) => {
        onEvent({
          type: "health",
          data: {
            event_id: "evt-health",
            created_at: Math.floor(Date.now() / 1000),
            status: "terminated",
            reason: "执行已超时终止",
            action: "terminated",
            metrics: {
              tool_calls_total: 5,
              tool_success_rate: 0.8,
            },
          },
        });
        onEvent({
          type: "done",
          data: {
            event_id: "evt-done",
            created_at: Math.floor(Date.now() / 1000),
          },
        });
        return () => {};
      });

      await useSessionStore.getState().sendChat("s-timeout", { message: "hi" });

      const current = useSessionStore.getState().currentSession;
      // done must NOT revert timed_out to completed
      expect(current?.status).toBe("timed_out");
    });

    it("TERMINATING health event sets timed_out immediately", async () => {
      mockedSessionApi.chat.mockImplementation((_sessionId, _params, onEvent) => {
        onEvent({
          type: "health",
          data: {
            event_id: "evt-terminating",
            created_at: Math.floor(Date.now() / 1000),
            status: "terminating",
            reason: "执行即将超时终止",
            action: "hard_terminate",
          },
        });
        return () => {};
      });

      await useSessionStore.getState().sendChat("s-terminating", { message: "hi" });

      const current = useSessionStore.getState().currentSession;
      expect(current?.status).toBe("timed_out");
    });

    it("DEGRADED health event does not change running status", async () => {
      mockedSessionApi.chat.mockImplementation((_sessionId, _params, onEvent) => {
        onEvent({
          type: "message",
          data: {
            event_id: "evt-msg",
            created_at: Math.floor(Date.now() / 1000),
            role: "assistant",
            message: "working",
            attachments: [],
          },
        });
        onEvent({
          type: "health",
          data: {
            event_id: "evt-degraded",
            created_at: Math.floor(Date.now() / 1000),
            status: "degraded",
            reason: "Agent 似乎遇到了困难，正在尝试恢复...",
            action: "soft_recovery",
            idle_seconds: 120.0,
          },
        });
        return () => {};
      });

      await useSessionStore.getState().sendChat("s-degraded", { message: "hi" });

      const current = useSessionStore.getState().currentSession;
      // DEGRADED is informational — session stays running (no done yet)
      expect(current?.status).toBe("running");
    });

    it("content events after TERMINATED do not revert timed_out", async () => {
      mockedSessionApi.chat.mockImplementation((_sessionId, _params, onEvent) => {
        onEvent({
          type: "health",
          data: {
            event_id: "evt-term",
            created_at: Math.floor(Date.now() / 1000),
            status: "terminated",
            reason: "执行已超时终止",
            action: "terminated",
          },
        });
        // A late MessageEvent arriving after TERMINATED must not flip status
        // back to running.
        onEvent({
          type: "message",
          data: {
            event_id: "evt-late",
            created_at: Math.floor(Date.now() / 1000),
            role: "assistant",
            message: "late partial",
            partial: true,
            attachments: [],
          },
        });
        return () => {};
      });

      await useSessionStore.getState().sendChat("s-late", { message: "hi" });

      const current = useSessionStore.getState().currentSession;
      expect(current?.status).toBe("timed_out");
    });

    it("health event appends to events array for rendering", async () => {
      mockedSessionApi.chat.mockImplementation((_sessionId, _params, onEvent) => {
        onEvent({
          type: "health",
          data: {
            event_id: "evt-health-render",
            created_at: Math.floor(Date.now() / 1000),
            status: "degraded",
            reason: "Agent 似乎遇到了困难",
            action: "soft_recovery",
          },
        });
        return () => {};
      });

      await useSessionStore.getState().sendChat("s-append", { message: "hi" });

      const events = useSessionStore.getState().currentSession?.events ?? [];
      const healthEvent = events.find((e) => e.event === "health");
      expect(healthEvent).toBeDefined();
      expect(healthEvent?.data?.status).toBe("degraded");
    });

    it("sandbox_state_changed 事件在 destroyed 时保持 timed_out 语义", async () => {
      mockedSessionApi.chat.mockImplementation((_sessionId, _params, onEvent) => {
        onEvent({
          type: "health",
          data: {
            event_id: "evt-term",
            created_at: Math.floor(Date.now() / 1000),
            status: "terminated",
            reason: "执行已超时终止",
            action: "terminated",
          },
        });
        onEvent({
          type: "sandbox_state_changed",
          data: {
            event_id: "evt-destroyed",
            created_at: Math.floor(Date.now() / 1000),
            old_state: "destroying",
            new_state: "destroyed",
          },
        });
        return () => {};
      });

      await useSessionStore.getState().sendChat("s-destroyed", { message: "hi" });

      const current = useSessionStore.getState().currentSession;
      expect(current?.status).toBe("timed_out");
      expect(current?.events?.some((e) => e.event === "sandbox_state_changed")).toBe(true);
    });
  });
});
