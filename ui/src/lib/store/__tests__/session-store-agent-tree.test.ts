import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/api/session", () => ({
  sessionApi: {
    getSession: vi.fn(),
    getSessionChildren: vi.fn(),
    getSessionCost: vi.fn(),
    getEventsSince: vi.fn(),
    getSessions: vi.fn(),
    streamSessions: vi.fn(),
  },
}));
vi.mock("@/lib/api/file", () => ({ fileApi: { uploadFile: vi.fn(), downloadFile: vi.fn() } }));
vi.mock("@/lib/api/session-compaction", () => ({ fetchCompactionList: vi.fn(async () => []) }));

import { buildTree, flattenTree } from "@/lib/agent-tree";
import { sessionApi } from "@/lib/api/session";
import type {
  ChildrenListResponse,
  CostAggregateResponse,
  CostStatus,
  EventsSinceResponse,
  Session,
} from "@/lib/api/types";
import {
  useMergedTimeline,
  useSessionStore,
  useToolCallCount,
} from "@/lib/store/session-store";

const mockedApi = vi.mocked(sessionApi, { deep: true });

function readySession(): Session {
  return { session_id: "root", title: "主会话", status: "running", events: [] };
}
function childrenResponse(): ChildrenListResponse {
  return {
    parent_session_id: "root",
    descendants: [
      {
        id: "c1",
        parent_session_id: "root",
        worker_type: "subagent",
        tool_filter_preset: "subagent_research",
        status: "completed",
        title: "研究子任务",
        created_at: "2026-07-01T00:00:00.000Z",
        updated_at: "2026-07-01T00:00:05.000Z",
      },
    ],
    truncated: false,
    depth_applied: 1,
  };
}

describe("agentTree slice", () => {
  beforeEach(() => {
    useSessionStore.getState().reset();
    vi.clearAllMocks();
    mockedApi.getSessionChildren.mockResolvedValue(childrenResponse());
  });

  it("builds the tree once currentSession is the ready root", async () => {
    useSessionStore.setState({ currentSession: readySession(), activeSessionId: "root" });
    await useSessionStore.getState().loadAgentTree("root");

    const tree = useSessionStore.getState().agentTree;
    expect(mockedApi.getSessionChildren).toHaveBeenCalledWith("root", 1);
    expect(tree.rootId).toBe("root");
    expect(tree.root?.children.map((c) => c.sessionId)).toEqual(["c1"]);
    expect(tree.root?.children[0].role).toBe("research_child");
    expect(Object.keys(tree.byId).sort()).toEqual(["c1", "root"]);
    expect(tree.truncated).toBe(false);
    expect(tree.loading).toBe(false);
  });

  it("DEFERS (no fetch, no build) when currentSession is null or a different session (R6-P2)", async () => {
    // currentSession null:
    await useSessionStore.getState().loadAgentTree("root");
    expect(mockedApi.getSessionChildren).not.toHaveBeenCalled();
    expect(useSessionStore.getState().agentTree.root).toBeNull();

    // currentSession is a DIFFERENT session:
    useSessionStore.setState({ currentSession: { ...readySession(), session_id: "other" }, activeSessionId: "other" });
    await useSessionStore.getState().loadAgentTree("root");
    expect(mockedApi.getSessionChildren).not.toHaveBeenCalled();
    expect(useSessionStore.getState().agentTree.root).toBeNull();
  });

  it("resetAgentTree restores the initial slice", async () => {
    useSessionStore.setState({ currentSession: readySession(), activeSessionId: "root" });
    await useSessionStore.getState().loadAgentTree("root");
    useSessionStore.getState().resetAgentTree();
    expect(useSessionStore.getState().agentTree.root).toBeNull();
    expect(useSessionStore.getState().agentTree.rootId).toBeNull();
  });

  it("keeps the root reference stable when a reload is structurally unchanged (R3 P3)", async () => {
    useSessionStore.setState({ currentSession: readySession(), activeSessionId: "root" });
    await useSessionStore.getState().loadAgentTree("root");
    const firstRoot = useSessionStore.getState().agentTree.root;
    await useSessionStore.getState().loadAgentTree("root"); // identical children
    expect(useSessionStore.getState().agentTree.root).toBe(firstRoot); // same reference, no churn
  });

  it("rebuilds when a child title changes null → empty string (signature distinguishes them, R4 P3)", async () => {
    useSessionStore.setState({ currentSession: readySession(), activeSessionId: "root" });
    const base = childrenResponse();
    mockedApi.getSessionChildren.mockResolvedValue({
      ...base,
      descendants: [{ ...base.descendants[0], title: null }],
    });
    await useSessionStore.getState().loadAgentTree("root");
    const firstRoot = useSessionStore.getState().agentTree.root;
    mockedApi.getSessionChildren.mockResolvedValue({
      ...base,
      descendants: [{ ...base.descendants[0], title: "" }],
    });
    await useSessionStore.getState().loadAgentTree("root");
    expect(useSessionStore.getState().agentTree.root).not.toBe(firstRoot); // rebuilt, no false-collision
  });

  it("does NOT clobber a newer load when an OLDER load resolves last (loadSeq guard, R4 P2)", async () => {
    useSessionStore.setState({ currentSession: readySession(), activeSessionId: "root" });

    // Two deferred responses we resolve by hand. A is the OLDER load, B the NEWER one.
    let resolveA!: (v: ChildrenListResponse) => void;
    let resolveB!: (v: ChildrenListResponse) => void;
    const pA = new Promise<ChildrenListResponse>((r) => {
      resolveA = r;
    });
    const pB = new Promise<ChildrenListResponse>((r) => {
      resolveB = r;
    });
    mockedApi.getSessionChildren.mockReturnValueOnce(pA).mockReturnValueOnce(pB);

    // Distinguish A's descendant from B's by swapping the single child's id.
    const base = childrenResponse();
    const respA: ChildrenListResponse = {
      ...base,
      descendants: [{ ...base.descendants[0], id: "cA" }],
    };
    const respB: ChildrenListResponse = {
      ...base,
      descendants: [{ ...base.descendants[0], id: "cB" }],
    };

    // Fire both without awaiting: A captures loadSeq=1, B bumps it to loadSeq=2.
    const loadA = useSessionStore.getState().loadAgentTree("root");
    const loadB = useSessionStore.getState().loadAgentTree("root");

    // Resolve the NEWER (B) first so it writes cB, then the OLDER (A) last.
    resolveB(respB);
    await Promise.resolve(); // flush B's continuation before A resolves
    resolveA(respA);
    await loadA;
    await loadB;

    // A resolved last but saw loadSeq bumped to 2 (≠ its captured 1) and dropped:
    // the tree must still reflect B, never the stale A.
    const tree = useSessionStore.getState().agentTree;
    expect(Object.keys(tree.byId).sort()).toEqual(["cB", "root"]);
    expect(tree.byId.cA).toBeUndefined();
    expect(tree.root?.children.map((c) => c.sessionId)).toEqual(["cB"]);
  });

  it("resetAgentTree mid-flight: a load resolving AFTER reset does not repopulate the reset slice (R5 P2)", async () => {
    useSessionStore.setState({ currentSession: readySession(), activeSessionId: "root" });

    let resolve!: (v: ChildrenListResponse) => void;
    const pending = new Promise<ChildrenListResponse>((r) => {
      resolve = r;
    });
    mockedApi.getSessionChildren.mockReturnValueOnce(pending);

    // Start the load (loadSeq=1), then reset while it is in flight (bumps loadSeq to 2).
    const load = useSessionStore.getState().loadAgentTree("root");
    useSessionStore.getState().resetAgentTree();

    // Now let the in-flight load resolve; its captured loadSeq=1 ≠ current 2 → it must drop.
    resolve(childrenResponse());
    await load;

    const tree = useSessionStore.getState().agentTree;
    expect(tree.root).toBeNull();
    expect(tree.rootId).toBeNull();
  });

  it("refreshAgentTree delegates to loadAgentTree AND respects the readiness gate", async () => {
    // Case 1 — delegates: with a ready root it fetches and builds via loadAgentTree.
    useSessionStore.setState({ currentSession: readySession(), activeSessionId: "root" });
    await useSessionStore.getState().refreshAgentTree("root");

    expect(mockedApi.getSessionChildren).toHaveBeenCalledWith("root", 1);
    expect(useSessionStore.getState().agentTree.rootId).toBe("root");

    // Reset both the store slice and the mock call log before Case 2 so the two
    // cases don't interfere within this single `it`.
    useSessionStore.getState().resetAgentTree();
    useSessionStore.setState({ currentSession: null, activeSessionId: null });
    vi.clearAllMocks();

    // Case 2 — gate applies through refresh: with no currentSession the delegated
    // loadAgentTree bails at the readiness gate (no fetch, no build).
    await useSessionStore.getState().refreshAgentTree("root");

    expect(mockedApi.getSessionChildren).not.toHaveBeenCalled();
    expect(useSessionStore.getState().agentTree.root).toBeNull();
  });
});

function costResponse(total: string, status: CostStatus = "actual"): CostAggregateResponse {
  return {
    total_usd: total,
    record_count: 1,
    by_node: {},
    by_model: {},
    by_provider: {},
    pricing_version: "v1",
    cost_status: status,
    first_record_at: null,
    last_record_at: null,
    has_partial_records: false,
  };
}

describe("loadNodeCost", () => {
  beforeEach(() => {
    useSessionStore.getState().reset();
    vi.clearAllMocks();
  });

  it("stores the fetched cost as a decimal-string snapshot", async () => {
    mockedApi.getSessionCost.mockResolvedValue(costResponse("0.0075000000", "partial"));
    await useSessionStore.getState().loadNodeCost("c1");
    expect(useSessionStore.getState().agentTree.costById["c1"]).toEqual({
      totalUsd: "0.0075000000",
      status: "partial",
    });
  });

  it("stores null when the cost fetch fails (soft-degrade)", async () => {
    mockedApi.getSessionCost.mockRejectedValue(new Error("boom"));
    await useSessionStore.getState().loadNodeCost("c1");
    expect(useSessionStore.getState().agentTree.costById["c1"]).toBeNull();
  });
});

function seedTreeState(childStatus: "running" | "completed") {
  const tree = buildTree(
    { sessionId: "root", status: "running", title: null, createdAt: null, updatedAt: null },
    [
      {
        id: "c1",
        parent_session_id: "root",
        worker_type: "subagent",
        tool_filter_preset: "subagent_research",
        status: childStatus,
        title: "c",
        created_at: null,
        updated_at: null,
      },
    ],
  );
  useSessionStore.setState({
    currentSession: { session_id: "root", title: null, status: "running", events: [] },
    activeSessionId: "root",
    agentTree: { ...useSessionStore.getState().agentTree, root: tree, rootId: "root", byId: flattenTree(tree) },
  });
}

describe("loadMergedTimeline", () => {
  beforeEach(() => {
    useSessionStore.getState().reset();
    vi.clearAllMocks();
  });

  it("fetches + normalizes DESCENDANTS only (not root, INV-10) and assigns colors", async () => {
    seedTreeState("running");
    mockedApi.getSession.mockResolvedValue({
      session_id: "c1",
      title: null,
      status: "running",
      events: [
        { event: "title", data: { title: "x" } }, // normalize DROPS title events
        { event: "message", data: { stream_id: "s", content: "a", event_id: "e1", created_at: 3 } },
        { event: "message", data: { stream_id: "s", content: "ab", event_id: "e2", created_at: 3 } }, // dedups to latest
      ],
      last_seq: 7,
    } as Session);

    await useSessionStore.getState().loadMergedTimeline("root");

    expect(mockedApi.getSession).toHaveBeenCalledWith("c1");
    expect(mockedApi.getSession).not.toHaveBeenCalledWith("root");
    const at = useSessionStore.getState().agentTree;
    // 3 raw events → normalizeSessionEvents (title-skip + stream_id dedup) → 1. PROVES the shared
    // full normalizer ran (INV-9); a raw-store regression would leave 3 events here (R4 P3).
    expect(at.eventsByAgent["c1"].events).toHaveLength(1);
    expect(at.eventsByAgent["c1"].events[0].data.content).toBe("ab");
    expect(at.eventsByAgent["c1"].lastSeq).toBe(7);
    expect(at.eventsByAgent["c1"].lastEventId).toBe("e2");
    expect(at.agentColors["root"]).toBeTruthy();
    expect(at.agentColors["c1"]).toBeTruthy();
    expect("mergedEvents" in at).toBe(false); // derived, never stored (INV-11)
  });

  it("DROPS its stale write when mergeSeq is bumped mid-flight (concurrent-invalidation guard)", async () => {
    seedTreeState("running"); // byId = { root, c1 } → c1 is the single descendant to fetch

    // Defer the descendant fetch so the action is in-flight when we bump mergeSeq.
    let resolveFetch!: (v: Session) => void;
    const deferred = new Promise<Session>((r) => {
      resolveFetch = r;
    });
    mockedApi.getSession.mockReturnValueOnce(deferred);

    // Fire WITHOUT awaiting: the action captures the current mergeSeq (+1) in its loading set,
    // then parks on the deferred getSession.
    const p = useSessionStore.getState().loadMergedTimeline("root");

    // Concurrent invalidation: bump mergeSeq while the fetch is still pending, so the action's
    // captured token no longer matches the store on resume.
    useSessionStore.setState((s) => ({ agentTree: { ...s.agentTree, mergeSeq: s.agentTree.mergeSeq + 1 } }));

    // Resolve the fetch with a real Session carrying events for c1, then let the action resume.
    resolveFetch({
      session_id: "c1",
      title: null,
      status: "running",
      events: [{ event: "message", data: { stream_id: "s", content: "a", event_id: "e1", created_at: 3 } }],
      last_seq: 5,
    } as Session);
    await p;

    // Its captured mergeSeq ≠ the bumped store value → it bailed BEFORE the eventsByAgent write,
    // so c1 never got a populated bundle.
    const at = useSessionStore.getState().agentTree;
    expect(at.eventsByAgent["c1"]).toBeUndefined();
    expect(Object.keys(at.eventsByAgent)).toHaveLength(0);
  });
});

describe("pollActiveAgents", () => {
  beforeEach(() => {
    useSessionStore.getState().reset();
    vi.clearAllMocks();
  });

  function seedBundle(childStatus: "running" | "completed") {
    seedTreeState(childStatus);
    useSessionStore.setState({
      agentTree: {
        ...useSessionStore.getState().agentTree,
        eventsByAgent: {
          c1: {
            events: [{ event: "tool", data: { tool_call_id: "t1", event_id: "e1", created_at: 3 } }],
            lastSeq: 7,
            lastEventId: "e1",
          },
        },
      },
    });
  }

  it("polls non-terminal descendants with BOTH cursors and appends new events", async () => {
    seedBundle("running");
    mockedApi.getEventsSince.mockResolvedValue({
      events: [{ event: "message", data: { stream_id: "m", event_id: "e2", created_at: 4 } }],
      session_status: "running",
      has_more: false,
      last_seq: 8,
      supervisor_snapshot: null,
    } as EventsSinceResponse);

    await useSessionStore.getState().pollActiveAgents("root");

    expect(mockedApi.getEventsSince).toHaveBeenCalledWith("c1", "e1", 7);
    expect(useSessionStore.getState().agentTree.eventsByAgent["c1"].events).toHaveLength(2);
    expect(useSessionStore.getState().agentTree.eventsByAgent["c1"].lastSeq).toBe(8);
  });

  it("skips terminal descendants that already have a bundle (never re-polls a completed child)", async () => {
    seedBundle("completed");
    await useSessionStore.getState().pollActiveAgents("root");
    expect(mockedApi.getEventsSince).not.toHaveBeenCalled();
  });

  it("full re-fetches (never seq-only) a bundle that has a seq but no event-id cursor (INV-8)", async () => {
    seedTreeState("running"); // c1 is a RUNNING descendant → a poll target
    // Pathological bundle: has a lastSeq but NO lastEventId (e.g. a child whose events all
    // normalize away — title-only — so there is no last event id to floor the query on).
    useSessionStore.setState({
      agentTree: {
        ...useSessionStore.getState().agentTree,
        eventsByAgent: {
          c1: {
            events: [{ event: "tool", data: { tool_call_id: "t1", created_at: 3 } }],
            lastSeq: 5,
            lastEventId: null,
          },
        },
      },
    });
    mockedApi.getSession.mockResolvedValue({
      session_id: "c1",
      title: null,
      status: "running",
      events: [{ event: "tool", data: { tool_call_id: "t1", event_id: "e1", created_at: 3 } }],
      last_seq: 5,
    } as Session);

    await useSessionStore.getState().pollActiveAgents("root");

    // Full re-fetch taken because there is no event-id cursor; the incremental seq-only path
    // (which would violate INV-8 — backend drops seq-is-None events without the event-id floor)
    // is NOT taken.
    expect(mockedApi.getSession).toHaveBeenCalledWith("c1");
    expect(mockedApi.getEventsSince).not.toHaveBeenCalled();
  });

  it("retries a descendant with no bundle via a full getSession (self-heal, R4 P2)", async () => {
    seedTreeState("running"); // c1 is in byId (running) but NOT in eventsByAgent (initial fetch failed)
    mockedApi.getSession.mockResolvedValue({
      session_id: "c1",
      title: null,
      status: "running",
      events: [{ event: "tool", data: { tool_call_id: "t1", event_id: "e1", created_at: 3 } }],
      last_seq: 5,
    } as Session);

    await useSessionStore.getState().pollActiveAgents("root");

    expect(mockedApi.getSession).toHaveBeenCalledWith("c1"); // full fetch, not getEventsSince
    expect(mockedApi.getEventsSince).not.toHaveBeenCalled();
    expect(useSessionStore.getState().agentTree.eventsByAgent["c1"].events).toHaveLength(1);
  });
});

describe("useMergedTimeline (derived, live)", () => {
  beforeEach(() => useSessionStore.getState().reset());

  function seed() {
    const tree = buildTree(
      { sessionId: "root", status: "running", title: null, createdAt: null, updatedAt: null },
      [{ id: "c1", parent_session_id: "root", worker_type: "subagent", tool_filter_preset: "subagent_research", status: "running", title: "c", created_at: null, updated_at: null }],
    );
    useSessionStore.setState({
      currentSession: { session_id: "root", title: null, status: "running", events: [{ event: "message", data: { stream_id: "r", created_at: 5, seq: 1 } }] },
      agentTree: {
        ...useSessionStore.getState().agentTree,
        rootId: "root",
        root: tree,
        byId: flattenTree(tree),
        eventsByAgent: { c1: { events: [{ event: "tool", data: { tool_call_id: "t", created_at: 3 } }], lastSeq: null, lastEventId: null } },
        agentColors: { root: "#111", c1: "#222" },
      },
    });
  }

  it("merges root (currentSession) + descendant events ordered by created_at", () => {
    act(() => seed());
    const { result } = renderHook(() => useMergedTimeline());
    expect(result.current.map((m) => m.sourceSessionId)).toEqual(["c1", "root"]);
    expect(result.current[0].sourceColor).toBe("#222");
  });

  it("recomputes live when a root event streams in", () => {
    act(() => seed());
    const { result } = renderHook(() => useMergedTimeline());
    const before = result.current.length;
    act(() => {
      useSessionStore.setState((s) => ({
        currentSession: { ...s.currentSession!, events: [...s.currentSession!.events, { event: "message", data: { stream_id: "r2", created_at: 9, seq: 2 } }] },
      }));
    });
    expect(result.current.length).toBe(before + 1);
  });
});

describe("useToolCallCount", () => {
  beforeEach(() => useSessionStore.getState().reset());

  it("sources root vs descendant events and returns undefined when not fetched", () => {
    act(() => {
      useSessionStore.setState({
        currentSession: { session_id: "root", title: null, status: "running", events: [{ event: "tool", data: { tool_call_id: "r1" } }] },
        agentTree: {
          ...useSessionStore.getState().agentTree,
          rootId: "root",
          eventsByAgent: { c1: { events: [{ event: "tool", data: { tool_call_id: "a" } }, { event: "tool", data: { tool_call_id: "b" } }], lastSeq: null, lastEventId: null } },
        },
      });
    });
    expect(renderHook(() => useToolCallCount("root")).result.current).toBe(1);
    expect(renderHook(() => useToolCallCount("c1")).result.current).toBe(2);
    expect(renderHook(() => useToolCallCount("ghost")).result.current).toBeUndefined();
  });
});
