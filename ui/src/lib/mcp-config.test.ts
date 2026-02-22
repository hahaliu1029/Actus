import { describe, expect, it } from "vitest";

import { normalizeMCPConfigInput } from "@/lib/mcp-config";

describe("mcp-config", () => {
  it("兼容 type 字段并映射为 transport=stdio", () => {
    const result = normalizeMCPConfigInput({
      mcpServers: {
        "zai-mcp-server": {
          type: "stdio",
          command: "npx",
          args: ["-y", "@z_ai/mcp-server"],
        },
      },
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.config.mcpServers["zai-mcp-server"]?.transport).toBe("stdio");
  });

  it("streamable_http 缺少 url 时返回可读错误", () => {
    const result = normalizeMCPConfigInput({
      mcpServers: {
        "remote-mcp": {
          transport: "streamable_http",
        },
      },
    });

    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }
    expect(result.error).toContain("必须传递 url");
  });
});

