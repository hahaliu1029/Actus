"use client";

import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";

import { fileApi } from "@/lib/api/file";
import { sessionApi } from "@/lib/api/session";
import type {
  ChatParams,
  FileInfo,
  GetSessionFilesResponse,
  ListSessionItem,
  Session,
  SSEEventData,
} from "@/lib/api/types";
import { registerStoreResetter } from "@/lib/store/reset";
import { useUIStore } from "@/lib/store/ui-store";

type SessionState = {
  sessions: ListSessionItem[];
  activeSessionId: string | null;
  currentSession: Session | null;
  currentSessionFiles: FileInfo[];
  isLoadingSessions: boolean;
  isLoadingCurrentSession: boolean;
  isChatting: boolean;
  chatSessionId: string | null;
  chatAbort: (() => void) | null;
  sessionsAbort: (() => void) | null;
};

type SessionActions = {
  reset: () => void;
  setActiveSession: (sessionId: string | null) => void;
  isSessionStreaming: (sessionId: string) => boolean;
  getSessionStatus: (sessionId: string) => Session["status"] | null;
  fetchSessions: () => Promise<void>;
  streamSessions: () => void;
  stopStreamSessions: () => void;
  createSession: () => Promise<string>;
  fetchSessionById: (
    sessionId: string,
    options?: { silent?: boolean }
  ) => Promise<void>;
  fetchSessionFiles: (
    sessionId: string,
    options?: { silent?: boolean }
  ) => Promise<void>;
  sendChat: (sessionId: string, params: ChatParams) => Promise<void>;
  stopChat: () => void;
  stopSession: (sessionId: string) => Promise<void>;
  deleteSession: (sessionId: string) => Promise<void>;
  clearUnread: (sessionId: string) => Promise<void>;
  uploadFile: (file: File, sessionId?: string) => Promise<FileInfo>;
  downloadFile: (fileId: string) => Promise<Blob>;
};

type SessionStore = SessionState & SessionActions;

type SessionEventRecord = {
  event: string;
  data: Record<string, unknown>;
};

const initialState: SessionState = {
  sessions: [],
  activeSessionId: null,
  currentSession: null,
  currentSessionFiles: [],
  isLoadingSessions: false,
  isLoadingCurrentSession: false,
  isChatting: false,
  chatSessionId: null,
  chatAbort: null,
  sessionsAbort: null,
};

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null
    ? (value as Record<string, unknown>)
    : {};
}

function applySSEToSession(session: Session, event: SSEEventData): Session {
  if (event.type === "done") {
    return session;
  }

  if (event.type === "title") {
    const nextTitle =
      typeof event.data.title === "string" ? event.data.title : session.title;
    return {
      ...session,
      title: nextTitle,
    };
  }

  const nextEvent: SessionEventRecord = {
    event: event.type,
    data: event.data as Record<string, unknown>,
  };

  const events = upsertSessionEvent(
    session.events as SessionEventRecord[],
    nextEvent
  );
  const withPlanStepSynced =
    event.type === "step"
      ? syncPlanStepsByStepEvent(events, nextEvent)
      : events;
  const withRecoveredErrorsPruned =
    event.type !== "error"
      ? pruneRecoveredLLMErrors(withPlanStepSynced)
      : withPlanStepSynced;

  return {
    ...session,
    events: withRecoveredErrorsPruned,
  };
}

function eventSemanticKey(event: SessionEventRecord): string | null {
  if (event.event === "message") {
    const streamId = event.data?.stream_id;
    if (typeof streamId === "string" && streamId.trim()) {
      return `message:${streamId}`;
    }
  }

  if (event.event === "plan") {
    return "plan:latest";
  }

  if (event.event === "tool") {
    const toolCallId = event.data?.tool_call_id;
    if (typeof toolCallId === "string" && toolCallId.trim()) {
      return `tool:${toolCallId}`;
    }
  }

  if (event.event === "step") {
    const stepId = event.data?.id;
    if (typeof stepId === "string" && stepId.trim()) {
      return `step:${stepId}`;
    }
  }

  const eventId = eventIdOf(event);
  if (eventId) {
    return `event:${eventId}`;
  }

  return null;
}

function upsertSessionEvent(
  events: SessionEventRecord[],
  nextEvent: SessionEventRecord
): SessionEventRecord[] {
  const nextKey = eventSemanticKey(nextEvent);
  if (!nextKey) {
    return [...events, nextEvent];
  }

  const existingIndex = events.findIndex(
    (item) => eventSemanticKey(item) === nextKey
  );
  if (existingIndex < 0) {
    return [...events, nextEvent];
  }

  const updated = [...events];
  updated[existingIndex] = nextEvent;
  return updated;
}

function normalizeSessionEvents(
  events: SessionEventRecord[]
): SessionEventRecord[] {
  let normalized: SessionEventRecord[] = [];
  events.forEach((event) => {
    if (event.event === "title") {
      return;
    }
    normalized = upsertSessionEvent(normalized, event);
    if (event.event === "step") {
      normalized = syncPlanStepsByStepEvent(normalized, event);
    }
  });
  return pruneRecoveredLLMErrors(normalized);
}

function pickTitle(session: Session): string | null {
  if (session.title) {
    return session.title;
  }
  const titleEvent = [...(session.events as SessionEventRecord[])]
    .reverse()
    .find((item) => item.event === "title");
  const title = titleEvent?.data?.title;
  return typeof title === "string" ? title : null;
}

function syncPlanStepsByStepEvent(
  events: SessionEventRecord[],
  stepEvent: SessionEventRecord
): SessionEventRecord[] {
  const stepId = stepEvent.data?.id;
  if (typeof stepId !== "string" || !stepId) {
    return events;
  }

  const planIndex = [...events]
    .map((item, index) => ({ item, index }))
    .reverse()
    .find(({ item }) => item.event === "plan")?.index;

  if (planIndex === undefined) {
    return events;
  }

  const planEvent = events[planIndex];
  const rawSteps = planEvent.data?.steps;
  if (!Array.isArray(rawSteps)) {
    return events;
  }

  const nextSteps = rawSteps.map((rawStep) => {
    const step = asRecord(rawStep);
    if (String(step.id || "") !== stepId) {
      return step;
    }
    return {
      ...step,
      status: stepEvent.data.status || step.status,
      description: stepEvent.data.description || step.description,
    };
  });

  const nextEvents = [...events];
  nextEvents[planIndex] = {
    ...planEvent,
    data: {
      ...planEvent.data,
      steps: nextSteps,
    },
  };
  return nextEvents;
}

function eventIdOf(event: SessionEventRecord): string | null {
  const eventId = event.data?.event_id;
  if (typeof eventId === "string" && eventId.trim()) {
    return eventId;
  }
  return null;
}

function getLatestEventId(events: SessionEventRecord[]): string | undefined {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (!event) {
      continue;
    }
    const id = eventIdOf(event);
    if (id) {
      return id;
    }
  }
  return undefined;
}

function mergeSessionEvents(
  remoteEvents: SessionEventRecord[],
  localEvents: SessionEventRecord[]
): SessionEventRecord[] {
  const merged = [...remoteEvents];
  const indexByKey = new Map<string, number>();

  merged.forEach((event, index) => {
    const key = eventSemanticKey(event);
    if (key) {
      indexByKey.set(key, index);
      return;
    }
    indexByKey.set(`remote:${index}:${event.event}`, index);
  });

  localEvents.forEach((event, index) => {
    const semanticKey = eventSemanticKey(event);
    const key = semanticKey || `local:${index}:${event.event}`;
    const existingIndex = indexByKey.get(key);
    if (existingIndex !== undefined) {
      merged[existingIndex] = event;
      return;
    }
    indexByKey.set(key, merged.length);
    merged.push(event);
  });

  return merged;
}

function showMessage(type: "success" | "error" | "info", text: string) {
  useUIStore.getState().setMessage({ type, text });
}

function stringifyEventData(data: Record<string, unknown>): string {
  try {
    return JSON.stringify(data);
  } catch {
    return "";
  }
}

function isSameEvents(
  left: SessionEventRecord[],
  right: SessionEventRecord[]
): boolean {
  if (left.length !== right.length) {
    return false;
  }

  for (let index = 0; index < left.length; index += 1) {
    const leftEvent = left[index];
    const rightEvent = right[index];
    if (!leftEvent || !rightEvent) {
      return false;
    }
    if (leftEvent.event !== rightEvent.event) {
      return false;
    }
    if (stringifyEventData(leftEvent.data) !== stringifyEventData(rightEvent.data)) {
      return false;
    }
  }

  return true;
}

function isSameSessionSnapshot(left: Session, right: Session): boolean {
  return (
    left.session_id === right.session_id &&
    left.status === right.status &&
    left.title === right.title &&
    isSameEvents(
      (left.events || []) as SessionEventRecord[],
      (right.events || []) as SessionEventRecord[]
    )
  );
}

function isSameFileList(left: FileInfo[], right: FileInfo[]): boolean {
  if (left.length !== right.length) {
    return false;
  }
  for (let index = 0; index < left.length; index += 1) {
    const leftFile = left[index];
    const rightFile = right[index];
    if (!leftFile || !rightFile) {
      return false;
    }
    if (
      leftFile.id !== rightFile.id ||
      leftFile.filename !== rightFile.filename ||
      leftFile.filepath !== rightFile.filepath ||
      leftFile.key !== rightFile.key ||
      leftFile.extension !== rightFile.extension ||
      leftFile.mime_type !== rightFile.mime_type ||
      leftFile.size !== rightFile.size
    ) {
      return false;
    }
  }
  return true;
}

function updateSessionListStatus(
  sessions: ListSessionItem[],
  sessionId: string,
  status: Session["status"]
): ListSessionItem[] {
  let changed = false;
  const next = sessions.map((item) => {
    if (item.session_id !== sessionId || item.status === status) {
      return item;
    }
    changed = true;
    return { ...item, status };
  });
  return changed ? next : sessions;
}

function isRecoverableLLMErrorEvent(event: SessionEventRecord): boolean {
  if (event.event !== "error") {
    return false;
  }
  const text = String(event.data?.error || "");
  if (!text) {
    return false;
  }
  return (
    text.includes("调用语言模型失败") ||
    text.includes("调用OpenAI客户端向LLM发起请求出错")
  );
}

function hasFollowingRecoveryEvent(
  events: SessionEventRecord[],
  fromIndex: number
): boolean {
  for (let index = fromIndex + 1; index < events.length; index += 1) {
    const event = events[index];
    if (!event) {
      continue;
    }
    if (event.event === "error" || event.event === "done" || event.event === "wait") {
      continue;
    }
    if (event.event === "message") {
      const role = String(event.data?.role || "assistant");
      if (role !== "assistant") {
        continue;
      }
    }
    return true;
  }
  return false;
}

function pruneRecoveredLLMErrors(
  events: SessionEventRecord[]
): SessionEventRecord[] {
  return events.filter((event, index) => {
    if (!isRecoverableLLMErrorEvent(event)) {
      return true;
    }
    return !hasFollowingRecoveryEvent(events, index);
  });
}

export const useSessionStore = create<SessionStore>()(
  subscribeWithSelector((set, get) => ({
    ...initialState,

    reset: () => {
      get().stopChat();
      get().stopStreamSessions();
      set(initialState);
    },

    setActiveSession: (sessionId: string | null) => {
      set((state) => {
        if (state.activeSessionId === sessionId) {
          return {};
        }
        return { activeSessionId: sessionId };
      });
    },

    isSessionStreaming: (sessionId: string) => {
      const state = get();
      return state.isChatting && state.chatSessionId === sessionId;
    },

    getSessionStatus: (sessionId: string) => {
      const state = get();
      if (state.currentSession?.session_id === sessionId) {
        return state.currentSession.status;
      }
      return state.sessions.find((item) => item.session_id === sessionId)?.status ?? null;
    },

    fetchSessions: async () => {
      set({ isLoadingSessions: true });
      try {
        const sessions = await sessionApi.getSessions();
        set({ sessions });
      } catch (error) {
        showMessage(
          "error",
          error instanceof Error ? error.message : "加载会话失败"
        );
      } finally {
        set({ isLoadingSessions: false });
      }
    },

    streamSessions: () => {
      const previousAbort = get().sessionsAbort;
      if (previousAbort) {
        previousAbort();
      }

      const abort = sessionApi.streamSessions(
        (event) => {
          if (event.type !== "sessions") {
            return;
          }
          set({ sessions: event.data.sessions });
        },
        (error) => {
          showMessage("error", error.message || "会话流连接异常");
        }
      );

      set({ sessionsAbort: abort });
    },

    stopStreamSessions: () => {
      const abort = get().sessionsAbort;
      if (abort) {
        abort();
      }
      set({ sessionsAbort: null });
    },

    createSession: async () => {
      const created = await sessionApi.createSession();
      await get().fetchSessions();
      showMessage("success", "新任务已创建");
      return created.session_id;
    },

    fetchSessionById: async (sessionId: string, options = {}) => {
      const silent = options.silent ?? false;
      if (!silent) {
        set({ isLoadingCurrentSession: true });
      }
      try {
        const session = await sessionApi.getSession(sessionId);
        const normalizedRemote: Session = {
          ...session,
          title: pickTitle(session),
          events: normalizeSessionEvents(session.events as SessionEventRecord[]),
        };

        set((state) => {
          if (state.activeSessionId && state.activeSessionId !== sessionId) {
            return {};
          }

          const localSession = state.currentSession;
          if (!localSession || localSession.session_id !== sessionId) {
            return { currentSession: normalizedRemote };
          }

          const mergedEvents = mergeSessionEvents(
            normalizedRemote.events as SessionEventRecord[],
            localSession.events as SessionEventRecord[]
          );
          const nextSession: Session = {
            ...normalizedRemote,
            title: normalizedRemote.title || localSession.title,
            events: mergedEvents,
          };

          if (isSameSessionSnapshot(localSession, nextSession)) {
            return {};
          }

          return {
            currentSession: nextSession,
          };
        });

        const stateAfterFetch = get();
        const fetchedSession =
          stateAfterFetch.currentSession &&
          stateAfterFetch.currentSession.session_id === sessionId
            ? stateAfterFetch.currentSession
            : null;
        const shouldResumeStream =
          fetchedSession?.status === "running" &&
          !stateAfterFetch.isChatting &&
          !stateAfterFetch.chatAbort;

        if (shouldResumeStream) {
          const latestEventId = getLatestEventId(
            (fetchedSession?.events || []) as SessionEventRecord[]
          );
          void get().sendChat(sessionId, { event_id: latestEventId });
        }
      } catch (error) {
        if (!silent) {
          showMessage(
            "error",
            error instanceof Error ? error.message : "加载会话详情失败"
          );
        }
      } finally {
        if (!silent) {
          const activeSessionId = get().activeSessionId;
          if (!activeSessionId || activeSessionId === sessionId) {
            set({ isLoadingCurrentSession: false });
          }
        }
      }
    },

    fetchSessionFiles: async (sessionId: string, options = {}) => {
      const silent = options.silent ?? false;
      try {
        const result: GetSessionFilesResponse =
          await sessionApi.getSessionFiles(sessionId);
        set((state) => {
          if (state.activeSessionId && state.activeSessionId !== sessionId) {
            return {};
          }
          if (isSameFileList(state.currentSessionFiles, result.files)) {
            return {};
          }
          return { currentSessionFiles: result.files };
        });
      } catch (error) {
        if (!silent) {
          showMessage(
            "error",
            error instanceof Error ? error.message : "加载会话文件失败"
          );
        }
      }
    },

    sendChat: async (sessionId, params) => {
      get().stopChat();

      set({ isChatting: true, chatSessionId: sessionId });

      const current = get().currentSession;
      const fallbackEventId =
        params.event_id ??
        (current?.session_id === sessionId
          ? getLatestEventId((current.events || []) as SessionEventRecord[])
          : undefined);
      const requestParams = fallbackEventId
        ? {
            ...params,
            event_id: fallbackEventId,
          }
        : params;

      let abortRef: (() => void) | null = null;
      let shouldClearAbortAfterBind = false;

      const clearChatState = () => {
        if (!abortRef) {
          shouldClearAbortAfterBind = true;
          set({ isChatting: false, chatSessionId: null });
          return;
        }
        set((state) => {
          if (!abortRef || state.chatAbort !== abortRef) {
            return {};
          }
          return { isChatting: false, chatSessionId: null, chatAbort: null };
        });
      };

      const abort = sessionApi.chat(
        sessionId,
        requestParams,
        (event) => {
          if (
            event.type === "tool" &&
            event.data.name === "file" &&
            event.data.status === "called"
          ) {
            void get().fetchSessionFiles(sessionId);
          }

          set((state) => {
            if (process.env.NODE_ENV === "development") {
              console.debug("[session-store] chat-event", {
                session_id: sessionId,
                event_type: event.type,
                chat_session_id: state.chatSessionId,
                is_chatting: state.isChatting,
              });
            }

            const nextStatus: Session["status"] =
              event.type === "wait"
                ? "waiting"
                : event.type === "done" || event.type === "error"
                  ? "completed"
                  : "running";
            const nextSessions = updateSessionListStatus(
              state.sessions,
              sessionId,
              nextStatus
            );

            if (state.activeSessionId && state.activeSessionId !== sessionId) {
              return nextSessions === state.sessions ? {} : { sessions: nextSessions };
            }

            const current =
              state.currentSession && state.currentSession.session_id === sessionId
                ? state.currentSession
                : ({
                    session_id: sessionId,
                    title: null,
                    status: "running",
                    events: [],
                  } as Session);

            const next = applySSEToSession(current, event);
            const shouldResetStreaming = state.chatSessionId === sessionId;

            if (event.type === "done") {
              shouldClearAbortAfterBind = true;
              return {
                currentSession: {
                  ...next,
                  status: "completed",
                },
                sessions: nextSessions,
                ...(shouldResetStreaming
                  ? { isChatting: false, chatSessionId: null }
                  : {}),
              };
            }

            if (event.type === "wait") {
              shouldClearAbortAfterBind = true;
              return {
                currentSession: {
                  ...next,
                  status: "waiting",
                },
                sessions: nextSessions,
                ...(shouldResetStreaming
                  ? { isChatting: false, chatSessionId: null }
                  : {}),
              };
            }

            if (event.type === "error") {
              shouldClearAbortAfterBind = true;
              return {
                currentSession: {
                  ...next,
                  status: "completed",
                },
                sessions: nextSessions,
                ...(shouldResetStreaming
                  ? { isChatting: false, chatSessionId: null }
                  : {}),
              };
            }

            return {
              currentSession: {
                ...next,
                status: "running",
              },
              sessions: nextSessions,
            };
          });
        },
        (error) => {
          showMessage("error", error.message || "聊天流中断");
          clearChatState();
        },
        () => {
          clearChatState();
        }
      );

      abortRef = abort;
      set({ chatAbort: abort });
      if (shouldClearAbortAfterBind) {
        clearChatState();
      }
    },

    stopChat: () => {
      const chatAbort = get().chatAbort;
      if (chatAbort) {
        chatAbort();
      }
      set({ chatAbort: null, isChatting: false, chatSessionId: null });
    },

    stopSession: async (sessionId: string) => {
      await sessionApi.stopSession(sessionId);
      if (get().chatSessionId === sessionId) {
        get().stopChat();
      }
      set((state) => ({
        sessions: updateSessionListStatus(state.sessions, sessionId, "completed"),
        currentSession:
          state.currentSession?.session_id === sessionId
            ? { ...state.currentSession, status: "completed" }
            : state.currentSession,
      }));
      showMessage("success", "任务已停止");
    },

    deleteSession: async (sessionId: string) => {
      await sessionApi.deleteSession(sessionId);
      const sessions = get().sessions.filter(
        (item) => item.session_id !== sessionId
      );
      set({ sessions });
      if (get().currentSession?.session_id === sessionId) {
        set({ currentSession: null, currentSessionFiles: [] });
      }
      showMessage("success", "任务已删除");
    },

    clearUnread: async (sessionId: string) => {
      await sessionApi.clearUnreadMessageCount(sessionId);
      set({
        sessions: get().sessions.map((item) =>
          item.session_id === sessionId
            ? { ...item, unread_message_count: 0 }
            : item
        ),
      });
    },

    uploadFile: async (file: File, sessionId?: string) => {
      const uploaded = await fileApi.uploadFile({ file, session_id: sessionId });
      set({ currentSessionFiles: [...get().currentSessionFiles, uploaded] });
      showMessage("success", `已上传文件：${uploaded.filename}`);
      return uploaded;
    },

    downloadFile: async (fileId: string) => {
      return fileApi.downloadFile(fileId);
    },
  }))
);

registerStoreResetter("session", () => {
  useSessionStore.getState().reset();
});
