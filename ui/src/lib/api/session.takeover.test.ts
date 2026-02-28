import { describe, expect, it, vi } from "vitest";

const { mockPost } = vi.hoisted(() => ({
  mockPost: vi.fn(),
}));

vi.mock("./fetch", () => ({
  createSSEStream: vi.fn(),
  parseSSEStream: vi.fn(),
  get: vi.fn(),
  post: mockPost,
}));

import { sessionApi } from "./session";

describe("sessionApi takeover", () => {
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
});
