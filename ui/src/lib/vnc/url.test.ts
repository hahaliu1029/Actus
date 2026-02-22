import { describe, expect, it } from "vitest";

import { buildVNCProxyUrl } from "@/lib/vnc/url";

describe("buildVNCProxyUrl", () => {
  it("会构造带 token 的 ws URL", () => {
    const url = buildVNCProxyUrl("session-1", "token-abc");

    expect(url).toContain("/api/sessions/session-1/vnc");
    expect(url).toContain("token=token-abc");
    expect(url.startsWith("ws://") || url.startsWith("wss://")).toBe(true);
  });
});
