import { createSSEStream, get, parseSSEStream, post } from "./fetch";
import { fileTransferClient } from "./axios-client";
import type {
  ChatParams,
  CostAggregateResponse,
  CreateSessionResponse,
  EndTakeoverParams,
  EndTakeoverResponse,
  EventsSinceResponse,
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

  getEventsSince: (
    sessionId: string,
    sinceEventId?: string,
    sinceSeq?: number, // B3-core PR-1 §3.3 — preferred monotonic cursor
  ): Promise<EventsSinceResponse> => {
    const params: Record<string, string> = {};
    if (sinceEventId) {
      params.since = sinceEventId;
    }
    if (sinceSeq !== undefined && sinceSeq !== null) {
      params.since_seq = String(sinceSeq);
    }
    return get<EventsSinceResponse>(`/sessions/${sessionId}/events`, params);
  },

  chat: (
    sessionId: string,
    params: ChatParams,
    onEvent: SSEEventHandler,
    onError?: (error: Error) => void,
    onClose?: () => void,
    onConnected?: () => void
  ): (() => void) => {
    let aborted = false;
    let stream: ReadableStream<Uint8Array> | null = null;

    const startChat = async () => {
      try {
        stream = await createSSEStream(`/sessions/${sessionId}/chat`, params);
        // E2: SSE 连接已建立（HTTP 200 + response.body），通知调用方
        if (!aborted) onConnected?.();

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

  downloadSandboxFile: async (
    sessionId: string,
    filepath: string,
    options?: {
      onProgress?: (loaded: number, total: number) => void;
      signal?: AbortSignal;
    }
  ): Promise<Blob> => {
    const response = await fileTransferClient.get(
      `/sessions/${sessionId}/file/download`,
      {
        params: { filepath },
        responseType: "blob",
        signal: options?.signal,
        onDownloadProgress: options?.onProgress
          ? (event) => {
              options.onProgress!(event.loaded, event.total ?? 0);
            }
          : undefined,
      }
    );
    return response.data as Blob;
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

  /**
   * B4 M0: per-session LLM cost rollup.
   *
   * Returns aggregate `total_usd` + breakdowns + `cost_status` so the UI
   * can show "partial / unknown" badges when the ledger is degraded.
   * Decimal fields are strings — render verbatim.
   */
  getSessionCost: (sessionId: string): Promise<CostAggregateResponse> => {
    return get<CostAggregateResponse>(`/sessions/${sessionId}/cost`);
  },
};
