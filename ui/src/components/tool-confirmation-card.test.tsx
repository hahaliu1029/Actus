import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ToolConfirmationCard } from "@/components/tool-confirmation-card";

vi.mock("next/navigation", () => ({
  useParams: () => ({ id: "s-1" }),
}));

const storeState = {
  currentSession: {
    session_id: "s-1",
    status: "waiting",
    events: [
      { event: "tool_confirmation", data: { tool_call_id: "tc-1" } },
    ],
  },
  sendChat: vi.fn(),
};

vi.mock("@/lib/store/session-store", () => ({
  useSessionStore: (selector: (s: typeof storeState) => unknown) =>
    selector(storeState),
}));

function makeData(
  overrides: Partial<Parameters<typeof ToolConfirmationCard>[0]["data"]> = {}
) {
  return {
    tool_call_id: "tc-1",
    tool_name: "shell_execute",
    tool_args: { command: "rm -rf /tmp/x" },
    risk_level: "high" as const,
    risk_reason: "dangerous command",
    matched_patterns: ["rm -rf"],
    suggested_alternative: null,
    approval_options: ["once", "session", "always", "deny"],
    timeout_seconds: 300,
    ...overrides,
  };
}

describe("decision_reason 渲染 allowlist (§5.4/R10#10)", () => {
  it("主文案恒为 risk_reason", () => {
    render(
      <ToolConfirmationCard
        data={makeData({
          decision_reason: {
            type: "smart_approve",
            code: "llm_escalate",
            message: "different explanation",
          },
        })}
      />
    );
    expect(screen.getByText(/dangerous command/)).toBeInTheDocument();
  });

  it("type badge 进 details; message 仅当 ≠ risk_reason 时出现", () => {
    render(
      <ToolConfirmationCard
        data={makeData({
          decision_reason: {
            type: "smart_approve",
            code: "llm_escalate",
            message: "different explanation",
          },
        })}
      />
    );
    expect(screen.getByText("smart_approve")).toBeInTheDocument();
    expect(screen.getByText(/different explanation/)).toBeInTheDocument();
  });

  it("message === risk_reason → 不重复展示", () => {
    render(
      <ToolConfirmationCard
        data={makeData({
          decision_reason: {
            type: "smart_approve",
            code: "c1",
            message: "dangerous command", // 与 risk_reason 同义反复 (R3#4)
          },
        })}
      />
    );
    expect(screen.getAllByText(/dangerous command/)).toHaveLength(1);
  });

  it("code 不出现在 DOM (R3#6: 诊断字段只进 log/audit)", () => {
    const { container } = render(
      <ToolConfirmationCard
        data={makeData({
          decision_reason: {
            type: "smart_approve",
            code: "llm_escalate_xyz",
            message: "m",
          },
        })}
      />
    );
    expect(screen.queryByText(/llm_escalate_xyz/)).not.toBeInTheDocument();
    // 属性面零泄漏 (title/aria/data-* 等): 全量 markup 不含 code 字符串
    expect(container.innerHTML).not.toContain("llm_escalate_xyz");
  });

  it("无 decision_reason → 渲染与现状一致 (INV-B10-8): 无决策来源区块", () => {
    render(<ToolConfirmationCard data={makeData()} />);
    expect(screen.queryByText("决策来源")).not.toBeInTheDocument();
    expect(screen.getByText(/dangerous command/)).toBeInTheDocument();
    expect(screen.getByText("HIGH")).toBeInTheDocument();
  });
});
