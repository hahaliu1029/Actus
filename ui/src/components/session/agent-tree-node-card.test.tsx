import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { AgentTreeNodeCard } from "@/components/session/agent-tree-node-card";
import type { AgentTreeNode } from "@/lib/agent-tree";

const NOW = Date.parse("2026-07-01T00:00:10.000Z");

function node(overrides: Partial<AgentTreeNode> = {}): AgentTreeNode {
  return {
    sessionId: "c1",
    parentSessionId: "root",
    workerType: "subagent",
    toolFilterPreset: "subagent_research",
    role: "research_child",
    status: "running",
    title: "研究子任务",
    createdAt: null,
    updatedAt: null,
    children: [],
    ...overrides,
  };
}

describe("AgentTreeNodeCard", () => {
  it("renders role + title and calls onSelect on click", () => {
    const onSelect = vi.fn();
    render(<AgentTreeNodeCard node={node()} onSelect={onSelect} nowMs={NOW} />);
    expect(screen.getByText("研究子 Agent")).toBeInTheDocument();
    expect(screen.getByText("研究子任务")).toBeInTheDocument();
    screen.getByRole("button").click();
    expect(onSelect).toHaveBeenCalledWith("c1");
  });

  it("disables the root node (not navigable) and shows the untitled fallback", () => {
    const onSelect = vi.fn();
    render(
      <AgentTreeNodeCard
        node={node({ role: "root", sessionId: "root", title: null })}
        onSelect={onSelect}
        nowMs={NOW}
      />,
    );
    const btn = screen.getByRole("button");
    expect(btn).toBeDisabled();
    expect(screen.getByText("未命名会话")).toBeInTheDocument();
  });
});

describe("AgentTreeNodeCard cost", () => {
  it("renders formatted cost when provided", () => {
    render(
      <AgentTreeNodeCard
        node={node()}
        onSelect={() => {}}
        nowMs={NOW}
        cost={{ totalUsd: "0.0075000000", status: "actual" }}
      />,
    );
    expect(screen.getByText("$0.0075")).toBeInTheDocument();
  });
  it("renders an em dash when cost is unavailable", () => {
    render(
      <AgentTreeNodeCard node={node()} onSelect={() => {}} nowMs={NOW} cost={null} />,
    );
    // both elapsed (null createdAt) and cost render "—"; expect at least two
    expect(screen.getAllByText("—").length).toBeGreaterThanOrEqual(2);
  });
});

describe("AgentTreeNodeCard tool-count", () => {
  it("renders the tool-call count when provided", () => {
    render(<AgentTreeNodeCard node={node()} onSelect={() => {}} nowMs={NOW} toolCount={3} />);
    expect(screen.getByText("🔧 3")).toBeInTheDocument();
  });
});
