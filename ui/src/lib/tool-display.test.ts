import { describe, expect, it } from "vitest";
import { FileText, Plug, Terminal, Wrench } from "lucide-react";

import type { ToolEventEnvelopeV1 } from "@/lib/api/types";
import {
  resolveToolDisplay,
  toolCardOverrideKey,
  TOOL_DISPLAY_REGISTRY,
} from "@/lib/tool-display";

function makeEnvelope(
  overrides: Partial<ToolEventEnvelopeV1> = {}
): ToolEventEnvelopeV1 {
  return {
    envelope_version: 1,
    tool_call_id: "tc-1",
    name: "file",
    function: "file_read",
    args: { filepath: "/workspace/report/a.txt" },
    status: "calling",
    activity_description: "",
    ...overrides,
  };
}

describe("resolveToolDisplay — icon fallback 链 (spec §5.1)", () => {
  it("第 1 级: envelope.display_icon 词表命中优先于 registry", () => {
    const d = resolveToolDisplay(makeEnvelope({ display_icon: "terminal" }));
    expect(d.icon).toBe(Terminal); // registry 本会给 FileText, envelope 赢
  });

  it("第 2 级: display_icon 缺失 → FE registry 按 function 名", () => {
    const d = resolveToolDisplay(makeEnvelope({ display_icon: null }));
    expect(d.icon).toBe(FileText);
  });

  it("第 3 级: 未知 function + 无 display_icon → generic (Wrench)", () => {
    const d = resolveToolDisplay(
      makeEnvelope({ function: "totally_unknown_tool", display_icon: null })
    );
    expect(d.icon).toBe(Wrench);
  });

  it("未知词表值不崩溃, 降级 registry → generic (§3.3 fail-closed)", () => {
    const d = resolveToolDisplay(
      makeEnvelope({ function: "totally_unknown_tool", display_icon: "hologram" })
    );
    expect(d.icon).toBe(Wrench);
  });

  it("词表 mcp → Plug", () => {
    const d = resolveToolDisplay(
      makeEnvelope({ function: "weather_lookup", display_icon: "mcp" })
    );
    expect(d.icon).toBe(Plug);
  });
});

describe("resolveToolDisplay — 双时态文案 (hermes B0.4)", () => {
  it("calling → verbPending", () => {
    const d = resolveToolDisplay(makeEnvelope({ status: "calling" }));
    expect(d.title).toBe("正在读取文件");
  });

  it("running → verbPending (与 calling 同分支)", () => {
    const d = resolveToolDisplay(makeEnvelope({ status: "running" }));
    expect(d.title).toBe("正在读取文件");
  });

  it("called → verbDone", () => {
    const d = resolveToolDisplay(makeEnvelope({ status: "called" }));
    expect(d.title).toBe("已读取文件");
  });

  it("未知 function 兜底文案不崩溃", () => {
    const d = resolveToolDisplay(
      makeEnvelope({ function: "totally_unknown_tool", status: "called" })
    );
    expect(d.title).toContain("已完成");
  });

  it("detail: file 类展示路径尾", () => {
    const d = resolveToolDisplay(makeEnvelope());
    expect(d.detail).toContain("a.txt");
  });

  it("detail: shell_execute 展示命令", () => {
    const d = resolveToolDisplay(
      makeEnvelope({ function: "shell_execute", args: { command: "ls -la" } })
    );
    expect(d.detail).toBe("命令：ls -la");
  });
});

describe("resolveToolDisplay — 策略位只信 envelope (INV-B10-4)", () => {
  it("read_only === true → collapsedByDefault", () => {
    const d = resolveToolDisplay(makeEnvelope({ read_only: true }));
    expect(d.collapsedByDefault).toBe(true);
  });

  it("read_only null/undefined/false → 不折叠", () => {
    expect(resolveToolDisplay(makeEnvelope({ read_only: null })).collapsedByDefault).toBe(false);
    expect(resolveToolDisplay(makeEnvelope()).collapsedByDefault).toBe(false);
    expect(resolveToolDisplay(makeEnvelope({ read_only: false })).collapsedByDefault).toBe(false);
  });

  it("registry 命中但 envelope 位全空 → 无折叠无高亮 (FE registry 禁止推断策略位)", () => {
    // file_read 在 BE registry 是 read_only=True, 但 FE 只信 envelope
    const d = resolveToolDisplay(makeEnvelope({ read_only: null, destructive: null }));
    expect(d.collapsedByDefault).toBe(false);
    expect(d.destructive).toBe(false);
  });

  it("destructive === true → destructive", () => {
    const d = resolveToolDisplay(
      makeEnvelope({ function: "shell_execute", destructive: true })
    );
    expect(d.destructive).toBe(true);
  });
});

describe("resolveToolDisplay — source badge (§5.3)", () => {
  it.each([
    ["mcp", "MCP"],
    ["skill", "Skill"],
    ["a2a", "A2A"],
  ] as const)("%s → %s badge + tooltip", (source, badge) => {
    const d = resolveToolDisplay(
      makeEnvelope({
        tool_source: { source, category: "cat x", canonical_name: "tool_y" },
      })
    );
    expect(d.sourceBadge).toBe(badge);
    expect(d.sourceTooltip).toBe("cat x / tool_y");
  });

  it("native → 无 badge", () => {
    const d = resolveToolDisplay(
      makeEnvelope({
        tool_source: { source: "native", category: "file", canonical_name: "file_read" },
      })
    );
    expect(d.sourceBadge).toBeNull();
    expect(d.sourceTooltip).toBeNull();
  });

  it("tool_source null/undefined → 无 badge", () => {
    expect(resolveToolDisplay(makeEnvelope({ tool_source: null })).sourceBadge).toBeNull();
    expect(resolveToolDisplay(makeEnvelope()).sourceBadge).toBeNull();
  });
});

describe("TOOL_DISPLAY_REGISTRY 完整性", () => {
  it("覆盖 38 个 canonical 工具", () => {
    expect(Object.keys(TOOL_DISPLAY_REGISTRY)).toHaveLength(38);
  });
});

describe("toolCardOverrideKey (R3#3)", () => {
  it("key 含 sessionId 前缀, 不同 session 同 tool_call_id 不碰撞", () => {
    expect(toolCardOverrideKey("s-a", "tc-1")).toBe("s-a:tc-1");
    expect(toolCardOverrideKey("s-b", "tc-1")).not.toBe(
      toolCardOverrideKey("s-a", "tc-1")
    );
  });
});
