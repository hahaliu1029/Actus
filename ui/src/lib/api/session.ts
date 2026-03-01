import { createSSEStream, get, parseSSEStream, post } from "./fetch";
import type {
  ChatParams,
  CreateSessionResponse,
  EndTakeoverParams,
  EndTakeoverResponse,
  FileReadResponse,
  GetTakeoverResponse,
  GetSessionFilesResponse,
  ListSessionItem,
  ListSessionResponse,
  RejectTakeoverParams,
  RejectTakeoverResponse,
  RenewTakeoverParams,
  RenewTakeoverResponse,
  ReopenTakeoverResponse,
  Session,
  ShellReadResponse,
  SSEEventData,
  SSEEventHandler,
  StartTakeoverParams,
  StartTakeoverResponse,
  ViewFileParams,
  ViewShellParams,
} from "./types";

export const sessionApi = {
  getSessions: async (): Promise<ListSessionItem[]> => {
    const data = await get<ListSessionResponse>("/sessions");
    return data.sessions;
  },

  createSession: (): Promise<CreateSessionResponse> => {
    return post<CreateSessionResponse>("/sessions", {});
  },

  streamSessions: (
    onEvent: SSEEventHandler,
    onError?: (error: Error) => void
  ): (() => void) => {
    let aborted = false;
    let stream: ReadableStream<Uint8Array> | null = null;

    const startStream = async () => {
      try {
        stream = await createSSEStream("/sessions/stream", {});

        await parseSSEStream(
          stream,
          (messageEvent) => {
            if (aborted) {
              return;
            }

            const parsed = messageEvent.data as ListSessionResponse;
            onEvent({
              type: "sessions",
              data: parsed,
            });
          },
          (error) => {
            if (!aborted) {
              onError?.(error);
            }
          }
        );
      } catch (error) {
        if (!aborted) {
          onError?.(error instanceof Error ? error : new Error("流式会话连接失败"));
        }
      }
    };

    void startStream();

    return () => {
      aborted = true;
      if (stream) {
        void stream.cancel();
      }
    };
  },

  getSession: (sessionId: string): Promise<Session> => {
    return get<Session>(`/sessions/${sessionId}`);
  },

  chat: (
    sessionId: string,
    params: ChatParams,
    onEvent: SSEEventHandler,
    onError?: (error: Error) => void,
    onClose?: () => void
  ): (() => void) => {
    let aborted = false;
    let stream: ReadableStream<Uint8Array> | null = null;

    const startChat = async () => {
      try {
        stream = await createSSEStream(`/sessions/${sessionId}/chat`, params);

        await parseSSEStream(
          stream,
          (messageEvent) => {
            if (aborted) {
              return;
            }

            const eventType = messageEvent.type as SSEEventData["type"];
            const data = messageEvent.data as SSEEventData["data"];

            onEvent({
              type: eventType,
              data,
            } as SSEEventData);
          },
          (error) => {
            if (!aborted) {
              onError?.(error);
            }
          }
        );
      } catch (error) {
        if (!aborted) {
          onError?.(error instanceof Error ? error : new Error("聊天流启动失败"));
        }
      } finally {
        if (!aborted) {
          onClose?.();
        }
      }
    };

    void startChat();

    return () => {
      aborted = true;
      if (stream) {
        void stream.cancel();
      }
    };
  },

  stopSession: (sessionId: string): Promise<void> => {
    return post<void>(`/sessions/${sessionId}/stop`, {});
  },

  deleteSession: (sessionId: string): Promise<void> => {
    return post<void>(`/sessions/${sessionId}/delete`, {});
  },

  clearUnreadMessageCount: (sessionId: string): Promise<void> => {
    return post<void>(`/sessions/${sessionId}/clear-unread-message-count`, {});
  },

  getSessionFiles: (sessionId: string): Promise<GetSessionFilesResponse> => {
    return get<GetSessionFilesResponse>(`/sessions/${sessionId}/files`);
  },

  viewFile: (sessionId: string, params: ViewFileParams): Promise<FileReadResponse> => {
    return post<FileReadResponse>(`/sessions/${sessionId}/file`, params);
  },

  viewShell: (
    sessionId: string,
    params: ViewShellParams
  ): Promise<ShellReadResponse> => {
    return post<ShellReadResponse>(`/sessions/${sessionId}/shell`, params);
  },

  getTakeover: (sessionId: string): Promise<GetTakeoverResponse> => {
    return get<GetTakeoverResponse>(`/sessions/${sessionId}/takeover`);
  },

  startTakeover: (
    sessionId: string,
    params: StartTakeoverParams = {}
  ): Promise<StartTakeoverResponse> => {
    return post<StartTakeoverResponse>(`/sessions/${sessionId}/takeover/start`, {
      scope: params.scope || "shell",
    });
  },

  renewTakeover: (
    sessionId: string,
    params: RenewTakeoverParams
  ): Promise<RenewTakeoverResponse> => {
    return post<RenewTakeoverResponse>(`/sessions/${sessionId}/takeover/renew`, params);
  },

  rejectTakeover: (
    sessionId: string,
    params: RejectTakeoverParams
  ): Promise<RejectTakeoverResponse> => {
    return post<RejectTakeoverResponse>(`/sessions/${sessionId}/takeover/reject`, params);
  },

  endTakeover: (
    sessionId: string,
    params: EndTakeoverParams = {}
  ): Promise<EndTakeoverResponse> => {
    return post<EndTakeoverResponse>(`/sessions/${sessionId}/takeover/end`, {
      handoff_mode: params.handoff_mode || "continue",
    });
  },

  reopenTakeover: (sessionId: string): Promise<ReopenTakeoverResponse> => {
    return post<ReopenTakeoverResponse>(`/sessions/${sessionId}/takeover/reopen`, {});
  },
};
