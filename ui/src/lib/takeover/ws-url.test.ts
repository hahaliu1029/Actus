import { describe, expect, it } from "vitest";

import { buildTakeoverShellWsUrl } from "./ws-url";

describe("buildTakeoverShellWsUrl", () => {
  it("应生成带 token 与 takeover_id 的 ws 地址", () => {
    const url = buildTakeoverShellWsUrl("sid-1", "tk-1", "token-1");
    expect(url.startsWith("ws://") || url.startsWith("wss://")).toBe(true);
    expect(url).toContain("/sessions/sid-1/takeover/shell/ws");
    expect(url).toContain("token=token-1");
    expect(url).toContain("takeover_id=tk-1");
  });
});
