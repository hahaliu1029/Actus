import { createSSEStream, get, parseSSEStream, post } from "./fetch";
import { fileTransferClient } from "./axios-client";
import {
  API_BASE_URL,
  getAccessToken,
  handleLogout,
  maybeRefreshToken,
} from "@/lib/api/auth-utils";
import type {
  BackgroundQuotaResponse,
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
  ResearchSubagentRequest,
  RetryFromSuspendResponse,
  Session,
  ShellReadResponse,
  SSEEventData,
  SSEEventHandler,
  StartTakeoverParams,
  StartTakeoverResponse,
  SubagentEvent,
  ViewFileParams,
  ViewShellParams,
} from "./types";

export const sessionApi = {
  getSessions: async (): Promise<ListSessionItem[]> => {
    const data = await get<ListSessionResponse>("/sessions");
    return data.sessions;
  },

  getBackgroundQuota: (): Promise<BackgroundQuotaResponse> => {
    return get<BackgroundQuotaResponse>("/sessions/background-quota");
  },

  retryFromSuspend: (sessionId: string): Promise<RetryFromSuspendResponse> => {
    return post<RetryFromSuspendResponse>(
      `/sessions/${sessionId}/retry-from-suspend`,
      {}
    );
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
    return post<void>(`/sessions/${sessionId}/cancel`, {});
  },

  cancelSession: (sessionId: string): Promise<void> => {
    return post<void>(`/sessions/${sessionId}/cancel`, {});
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

// ==================== Subagent Research (Phase 1 minimal) ====================

/**
 * Open a POST-SSE stream against the subagent research endpoint.
 *
 * Unlike `sessionApi.chat`, this uses raw `fetch()` instead of the shared
 * `createSSEStream` helper because the backend SSE wire here uses
 * single-line `data:` JSON payloads (no `event:` field) and the helper
 * pre-parses them differently. Returning `{ close }` matches the panel
 * component's cleanup contract on unmount / user dismiss.
 */
export function openSubagentResearchStream(
  parentSessionId: string,
  request: ResearchSubagentRequest,
  onEvent: (event: SubagentEvent) => void,
  onError: (error: Event) => void,
  onClose: () => void,
): { close: () => void } {
  const url = `${API_BASE_URL}/sessions/${parentSessionId}/subagents/research`;
  const controller = new AbortController();

  // Initial auth token check — refresh-and-retry on 401 happens below.
  const initialToken = getAccessToken();
  if (!initialToken) {
    onError(new Event("no-auth-token"));
    return { close: () => {} };
  }

  // Codex R3 P2: mirror requestResponse() 401-refresh-and-retry behaviour
  // (ui/src/lib/api/fetch.ts:126) so an expired access token gets transparently
  // refreshed instead of failing the panel.
  const doFetch = (authToken: string): Promise<Response> =>
    fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${authToken}`,
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(request),
      signal: controller.signal,
    });

  void (async () => {
    try {
      let response = await doFetch(initialToken);

      if (response.status === 401) {
        const refreshed = await maybeRefreshToken();
        if (!refreshed) {
          handleLogout();
          onError(new Event("http-401"));
          return;
        }
        const fresh = getAccessToken();
        if (!fresh) {
          onError(new Event("no-auth-token"));
          return;
        }
        response = await doFetch(fresh);
      }

      if (!response.ok) {
        // Codex R2 P2: propagate status so the panel can map 400/409/404 to
        // user-readable copy (classifier reject / quota / parent not found).
        onError(new Event(`http-${response.status}`));
        return;
      }
      if (!response.body) {
        onError(new Event("no-body"));
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      // Codex R1 P1#1: sse-starlette emits CRLF (`\r\n...\r\n\r\n`), so we must
      // split blocks on `\r?\n\r?\n` rather than `\n\n` — otherwise the buffer
      // grows forever and no event fires. Verified via
      // `ServerSentEvent(...).encode()` empirically.
      const SSE_BLOCK_SEP = /\r?\n\r?\n/;
      const SSE_LINE_SEP = /\r?\n/;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const blocks = buf.split(SSE_BLOCK_SEP);
        buf = blocks.pop() ?? "";
        for (const block of blocks) {
          let dataLine = "";
          let eventLine = "";
          let idLine = "";
          for (const line of block.split(SSE_LINE_SEP)) {
            if (line.startsWith("data:")) {
              dataLine = line.slice(5).trim();
            } else if (line.startsWith("event:")) {
              eventLine = line.slice(6).trim();
            } else if (line.startsWith("id:")) {
              idLine = line.slice(3).trim();
            }
          }
          if (dataLine && eventLine) {
            try {
              // Cross-PR P1 (final review): backend EventMapper →
              // CommonEventData.from_event excludes `id`/`type` from the
              // `data:` JSON (api/app/interfaces/schemas/event.py:60); the
              // discriminator lives on the SSE `event:` line and the frame id
              // on the `id:` line. Reassemble them back onto the event object
              // so `SubagentEvent` discriminated-union narrowing actually
              // matches at runtime.
              const payload = JSON.parse(dataLine) as Record<string, unknown>;
              const ev = {
                ...payload,
                type: eventLine,
                id: idLine,
              } as unknown as SubagentEvent;
              onEvent(ev);
            } catch (parseErr) {
              console.warn("subagent SSE parse error:", parseErr);
            }
          }
        }
      }
      onClose();
    } catch (err: unknown) {
      if (controller.signal.aborted) return;
      onError(err instanceof Event ? err : new Event("fetch-error"));
    }
  })();

  return {
    close: () => controller.abort(),
  };
}
