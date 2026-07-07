import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/api/fetch", () => ({
  get: vi.fn(),
  put: vi.fn(),
  del: vi.fn(),
  post: vi.fn(),
  ApiError: class ApiError extends Error {}, // session-compaction.ts imports it
}));

import { del, get, post, put } from "@/lib/api/fetch";
import { userToolPolicyApi } from "@/lib/api/config";
import { requestManualCompaction } from "@/lib/api/session-compaction";

const getMock = vi.mocked(get);
const putMock = vi.mocked(put);
const delMock = vi.mocked(del);
const postMock = vi.mocked(post);

afterEach(() => vi.clearAllMocks());

describe("userToolPolicyApi", () => {
  it("list unwraps { policies } envelope-data", async () => {
    getMock.mockResolvedValue({
      policies: [{ tool_name: "shell", policy: "deny" }],
    });
    const out = await userToolPolicyApi.list();
    expect(getMock).toHaveBeenCalledWith("/v2/user/tool-policies");
    expect(out).toEqual([{ tool_name: "shell", policy: "deny" }]);
  });

  it("set PUTs { policy } to encoded tool path", async () => {
    putMock.mockResolvedValue({ tool_name: "sh ell", policy: "auto" });
    await userToolPolicyApi.set("sh ell", "auto");
    expect(putMock).toHaveBeenCalledWith("/v2/user/tool-policies/sh%20ell", {
      policy: "auto",
    });
  });

  it("clear DELETEs encoded tool path", async () => {
    delMock.mockResolvedValue(undefined);
    await userToolPolicyApi.clear("sh/ell");
    expect(delMock).toHaveBeenCalledWith("/v2/user/tool-policies/sh%2Fell");
  });
});

describe("requestManualCompaction", () => {
  it("POSTs {} to the encoded compactions collection, returns queued (bare-JSON unwrap)", async () => {
    postMock.mockResolvedValue({ request_status: "queued" });
    const out = await requestManualCompaction("s 1");
    expect(postMock).toHaveBeenCalledWith("/sessions/s%201/compactions", {});
    expect(out).toEqual({ request_status: "queued" });
  });
});
