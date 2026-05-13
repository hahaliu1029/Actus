import { beforeEach, describe, expect, it, vi } from "vitest";

const { mockGet, mockPost } = vi.hoisted(() => ({
  mockGet: vi.fn(),
  mockPost: vi.fn(),
}));

vi.mock("./fetch", () => ({
  createSSEStream: vi.fn(),
  parseSSEStream: vi.fn(),
  get: mockGet,
  post: mockPost,
}));

import { sessionApi } from "./session";

describe("sessionApi takeover", () => {
  beforeEach(() => {
    mockGet.mockReset();
    mockPost.mockReset();
  });

  it("getEventsSince 应序列化 since 与 since_seq 参数", async () => {
    mockGet.mockResolvedValue({
      events: [],
      session_status: "running",
      has_more: false,
      last_seq: 7,
      supervisor_snapshot: null,
    });

    await sessionApi.getEventsSince("sid", "evt-1", 7);

    expect(mockGet).toHaveBeenCalledWith("/sessions/sid/events", {
      since: "evt-1",
      since_seq: "7",
    });
  });

  it("cancelSession 应走 PR-3c cancel endpoint", async () => {
    mockPost.mockResolvedValue(undefined);

    await sessionApi.cancelSession("sid-cancel");

    expect(mockPost).toHaveBeenCalledWith("/sessions/sid-cancel/cancel", {});
  });

  it("stopSession 应走 PR-3c cancel endpoint", async () => {
    mockPost.mockResolvedValue(undefined);

    await sessionApi.stopSession("sid-cancel");

    expect(mockPost).toHaveBeenCalledWith("/sessions/sid-cancel/cancel", {});
  });

  it("retryFromSuspend 应调用后台挂起重试 endpoint", async () => {
    mockPost.mockResolvedValue({
      status: "running",
      request_status: "resumed",
      retry_budget_remaining: 1,
      expires_at: null,
    });

    await sessionApi.retryFromSuspend("sid-bg");

    expect(mockPost).toHaveBeenCalledWith(
      "/sessions/sid-bg/retry-from-suspend",
      {}
    );
  });

  it("endTakeover 默认 handoff_mode 应为 continue", async () => {
    mockPost.mockResolvedValue({
      status: "running",
      handoff_mode: "continue",
    });

    await sessionApi.endTakeover("sid-1");

    expect(mockPost).toHaveBeenCalledWith("/sessions/sid-1/takeover/end", {
      handoff_mode: "continue",
    });
  });

  it("reopenTakeover 发送 POST 且无请求体参数", async () => {
    mockPost.mockResolvedValue({
      status: "takeover_pending",
      request_status: "reopened",
      reason: null,
      remaining_seconds: 240,
    });

    await sessionApi.reopenTakeover("sid-2");

    expect(mockPost).toHaveBeenCalledWith("/sessions/sid-2/takeover/reopen", {});
  });
});
