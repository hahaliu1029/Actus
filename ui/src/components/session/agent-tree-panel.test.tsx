import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { push } = vi.hoisted(() => ({ push: vi.fn() }));
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));

import { AgentTreePanel } from "@/components/session/agent-tree-panel";
import { buildTree, flattenTree } from "@/lib/agent-tree";
import type { ChildSessionItem } from "@/lib/api/types";
import { useSessionStore } from "@/lib/store/session-store";

function seedTree(descendants: ChildSessionItem[], extra: Partial<{ truncated: boolean; error: string | null }> = {}) {
  const root = buildTree(
    { sessionId: "root", status: "running", title: "主会话", createdAt: null, updatedAt: null },
    descendants,
  );
  useSessionStore.setState({
    agentTree: {
      ...useSessionStore.getState().agentTree,
      root,
      rootId: "root",
      byId: flattenTree(root),
      truncated: extra.truncated ?? false,
      error: extra.error ?? null,
    },
  });
}
const child: ChildSessionItem = {
  id: "c1",
  parent_session_id: "root",
  worker_type: "subagent",
  tool_filter_preset: "subagent_research",
  status: "completed",
  title: "研究子任务",
  created_at: null,
  updated_at: null,
};

describe("AgentTreePanel", () => {
  beforeEach(() => {
    useSessionStore.getState().reset();
    push.mockClear();
    // The panel fires loadNodeCost on mount — stub it so no render hits a real fetch.
    useSessionStore.setState({ loadNodeCost: vi.fn() });
  });

  it("renders nothing in the single-agent state (no children, no error)", () => {
    const { container } = render(<AgentTreePanel />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders the tree and navigates to the child session page on click", () => {
    seedTree([child]);
    render(<AgentTreePanel />);
    expect(screen.getByText("Agent 树")).toBeInTheDocument();
    screen.getByText("研究子任务").closest("button")!.click();
    expect(push).toHaveBeenCalledWith("/sessions/c1");
  });

  it("shows the truncation affordance", () => {
    seedTree([child], { truncated: true });
    render(<AgentTreePanel />);
    expect(screen.getByText("仅显示前 10 个子 Agent")).toBeInTheDocument();
  });

  it("fetches per-node cost for each uncached node on render", () => {
    const loadNodeCost = vi.fn();
    seedTree([child]);
    useSessionStore.setState({ loadNodeCost });
    render(<AgentTreePanel />);
    expect(loadNodeCost).toHaveBeenCalledWith("root");
    expect(loadNodeCost).toHaveBeenCalledWith("c1");
  });

  it("renders the fetched cost snapshot for a node", () => {
    seedTree([child]);
    useSessionStore.setState({
      agentTree: {
        ...useSessionStore.getState().agentTree,
        costById: { c1: { totalUsd: "0.0075000000", status: "actual" }, root: null },
      },
      loadNodeCost: vi.fn(),
    });
    render(<AgentTreePanel />);
    expect(screen.getByText("$0.0075")).toBeInTheDocument();
  });

  it("shows per-node tool-count derived from eventsByAgent", () => {
    seedTree([child]);
    useSessionStore.setState({
      agentTree: {
        ...useSessionStore.getState().agentTree,
        eventsByAgent: {
          c1: {
            events: [
              { event: "tool", data: { tool_call_id: "a" } },
              { event: "tool", data: { tool_call_id: "b" } },
            ],
            lastSeq: null,
            lastEventId: null,
          },
        },
      },
      loadNodeCost: vi.fn(),
    });
    render(<AgentTreePanel />);
    expect(screen.getByText("🔧 2")).toBeInTheDocument();
  });
});
