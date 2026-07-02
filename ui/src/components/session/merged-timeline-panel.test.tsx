import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import { MergedTimelinePanel } from "@/components/session/merged-timeline-panel";
import { buildTree, flattenTree } from "@/lib/agent-tree";
import { useSessionStore } from "@/lib/store/session-store";

describe("MergedTimelinePanel", () => {
  beforeEach(() => useSessionStore.getState().reset());

  it("renders nothing when there are no merged items", () => {
    const { container } = render(<MergedTimelinePanel />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders merged items from root + descendant, ordered by created_at", () => {
    const tree = buildTree(
      { sessionId: "root", status: "running", title: null, createdAt: null, updatedAt: null },
      [{ id: "c1", parent_session_id: "root", worker_type: "subagent", tool_filter_preset: "subagent_research", status: "running", title: "c", created_at: null, updated_at: null }],
    );
    useSessionStore.setState({
      currentSession: {
        session_id: "root",
        title: null,
        status: "running",
        events: [{ event: "message", data: { role: "assistant", message: "hello world", created_at: 5, stream_id: "r" } }],
      },
      agentTree: {
        ...useSessionStore.getState().agentTree,
        rootId: "root",
        root: tree,
        byId: flattenTree(tree),
        eventsByAgent: {
          c1: {
            events: [
              // Legacy raw tool shape carries `tool_name`…
              { event: "tool", data: { tool_call_id: "t", tool_name: "file_read", status: "done", created_at: 3 } },
              // …while the V1 tool envelope carries `name` (no tool_name) — both
              // shapes reach the merge, so BOTH summarize() reads are load-bearing.
              { event: "tool", data: { tool_call_id: "t2", name: "shell_execute", status: "done", created_at: 4 } },
            ],
            lastSeq: null,
            lastEventId: null,
          },
        },
        agentColors: { root: "#111", c1: "#222" },
      },
    });
    render(<MergedTimelinePanel />);
    expect(screen.getByText("合并时间线")).toBeInTheDocument();
    expect(screen.getByText(/file_read/)).toBeInTheDocument();
    // V1-shaped tool row (name, no tool_name) must render via the `data.name`
    // fallback — a regression to tool_name-only would blank real V1 tool rows.
    expect(screen.getByText(/shell_execute/)).toBeInTheDocument();
    // message events store text in data.message (not data.content); the row
    // must render the visible message text, not a blank "assistant:" line.
    expect(screen.getByText(/hello world/)).toBeInTheDocument();
    expect(screen.getByText(/assistant: hello world/)).toBeInTheDocument();
  });
});
