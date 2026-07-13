import { describe, expect, it } from "vitest";

import { toDisplayImageUrl } from "@/components/tool-visual";

describe("toDisplayImageUrl", () => {
  it.each([
    "http://localhost:9000/a2a-mcp/screenshot.png?X-Amz-Signature=test",
    "http://127.0.0.1:19000/a2a-mcp/screenshot.png?X-Amz-Signature=test",
  ])("loopback 图片由浏览器直接访问: %s", (url) => {
    expect(toDisplayImageUrl(url)).toBe(url);
  });

  it("远程图片继续通过服务端代理", () => {
    const url = "https://images.example.com/screenshot.png";

    expect(toDisplayImageUrl(url)).toBe(
      `/api/image-proxy?url=${encodeURIComponent(url)}`
    );
  });
});
