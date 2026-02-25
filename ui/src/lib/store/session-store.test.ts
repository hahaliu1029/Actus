import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/api/session", () => ({
  sessionApi: {
    getSessions: vi.fn(),
    streamSessions: vi.fn(),
    createSession: vi.fn(),
    getSession: vi.fn(),
    getSessionFiles: vi.fn(),
    chat: vi.fn(),
    stopSession: vi.fn(),
    deleteSession: vi.fn(),
    clearUnreadMessageCount: vi.fn(),
    viewFile: vi.fn(),
    viewShell: vi.fn(),
  },
}));

vi.mock("@/lib/api/file", () => ({
  fileApi: {
    uploadFile: vi.fn(),
    downloadFile: vi.fn(),
  },
}));

import { sessionApi } from "@/lib/api/session";
import type { Session } from "@/lib/api/types";
import { useSessionStore } from "@/lib/store/session-store";
import { useUIStore } from "@/lib/store/ui-store";

const mockedSessionApi = vi.mocked(sessionApi, { deep: true });

function buildSession(overrides?: Partial<Session>): Session {
  return {
    session_id: "s1",
    title: "会话",
    status: "running",
    events: [],
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
          event_id: "evt-tool-1",
          created_at: Math.floor(Date.now() / 1000),
          tool_call_id: "tool-123",
          name: "file",
          function: "write_file",
          args: { filepath: "/home/ubuntu/a.txt" },
          status: "calling",
        },
      });
      onEvent({
        type: "tool",
        data: {
          event_id: "evt-tool-2",
          created_at: Math.floor(Date.now() / 1000),
          tool_call_id: "tool-123",
          name: "file",
          function: "write_file",
          args: { filepath: "/home/ubuntu/a.txt" },
          status: "called",
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

  it("sendChat 终态事件应同步更新 sessions 列表状态", async () => {
    useSessionStore.setState({
      sessions: [
        {
          session_id: "s-done",
          title: "会话",
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
      status: "running",
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
        status: "running",
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
});
