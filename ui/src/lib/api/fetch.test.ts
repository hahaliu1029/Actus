import { beforeEach, describe, expect, it, vi } from "vitest";

import { createSSEStream, parseSSEStream, request } from "@/lib/api/fetch";
import { useAuthStore } from "@/lib/store/auth-store";

function jsonResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json",
    },
  });
}

describe("api/fetch", () => {
  beforeEach(() => {
    useAuthStore.setState({
      accessToken: "expired-token",
      refreshToken: "refresh-token",
      user: {
        id: "u1",
        username: "demo",
        email: "demo@example.com",
        nickname: "Demo",
        avatar: null,
        role: "user",
        status: "active",
        created_at: new Date().toISOString(),
      },
      isHydrated: true,
    });
  });

  it("401 后自动 refresh 并重试成功", async () => {
    const fetchMock = vi.fn();
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ code: 401, msg: "未授权", data: {} }, 401))
      .mockResolvedValueOnce(
        jsonResponse({
          code: 200,
          msg: "ok",
          data: {
            access_token: "new-access-token",
            refresh_token: "new-refresh-token",
            token_type: "bearer",
          },
        })
      )
      .mockResolvedValueOnce(jsonResponse({ code: 200, msg: "ok", data: { value: 1 } }));

    vi.stubGlobal("fetch", fetchMock);

    const result = await request<{ value: number }>("/sessions", { method: "GET" });

    expect(result.value).toBe(1);
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(useAuthStore.getState().accessToken).toBe("new-access-token");
  });

  it("SSE 解析兼容 CRLF 并按事件顺序输出", async () => {
    const encoder = new TextEncoder();
    const events: Array<{ type: string; data: unknown }> = [];
    const errors: string[] = [];

    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(
          encoder.encode(
            "id: 1\r\nevent: message\r\ndata: {\"role\":\"assistant\",\"message\":\"hello\"}\r\n\r\n"
          )
        );
        controller.enqueue(
          encoder.encode("id: 2\r\nevent: done\r\ndata: {}\r\n\r\n")
        );
        controller.close();
      },
    });

    await parseSSEStream(
      stream,
      (event) => {
        events.push({ type: event.type, data: event.data });
      },
      (error) => {
        errors.push(error.message);
      }
    );

    expect(errors).toEqual([]);
    expect(events).toHaveLength(2);
    expect(events[0]?.type).toBe("message");
    expect(events[1]?.type).toBe("done");
  });

  it("createSSEStream 默认不应启用30秒请求超时", async () => {
    const timeoutSpy = vi.spyOn(globalThis, "setTimeout");
    const body = new ReadableStream<Uint8Array>({
      start() {},
    });

    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(body, {
          status: 200,
          headers: {
            "Content-Type": "text/event-stream",
          },
        })
      )
    );

    await createSSEStream("/sessions/stream", {});

    const timeoutCalls = timeoutSpy.mock.calls.filter(
      (call) => typeof call[1] === "number" && call[1] === 30000
    );
    expect(timeoutCalls).toHaveLength(0);
  });

  it("非统一错误响应包含 detail 时应提取后端明细文案", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(
          {
            detail: [
              {
                msg: "在sse或streamable_http模式下必须传递url",
              },
            ],
          },
          422
        )
      )
    );

    await expect(
      request("/app-config/mcp-servers", {
        method: "POST",
        body: "{}",
      })
    ).rejects.toMatchObject({
      message: "在sse或streamable_http模式下必须传递url",
    });
  });
});
