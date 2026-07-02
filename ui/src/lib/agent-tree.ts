import type { ChildSessionItem, SessionStatus } from "@/lib/api/types";

export type AgentRole = "root" | "coordinator_child" | "research_child" | "subagent";

/** One node of the agent tree. Purely structural — cost/color/tool-count are NOT stored here. */
export type AgentTreeNode = {
  sessionId: string;
  parentSessionId: string | null;
  workerType: "root" | "subagent";
  toolFilterPreset: string | null;
  role: AgentRole;
  status: SessionStatus;
  title: string | null;
  createdAt: string | null;
  updatedAt: string | null;
  children: AgentTreeNode[];
};

/** The root session's own metadata (from the live `currentSession`; NOT in /children). */
export type AgentTreeRoot = {
  sessionId: string;
  status: SessionStatus;
  title: string | null;
  createdAt: string | null; // Session has no timestamps → always null for the root (F0.3)
  updatedAt: string | null;
};

/** Pure, total: worker_type + tool_filter_preset → display role. */
export function deriveRole(
  workerType: string,
  toolFilterPreset: string | null,
): AgentRole {
  if (workerType === "root") {
    return "root";
  }
  if (toolFilterPreset === "coordinator_step") {
    return "coordinator_child";
  }
  if (toolFilterPreset === "subagent_research") {
    return "research_child";
  }
  return "subagent";
}

// Note: ChildSessionItem is imported so later tasks (buildTree) can reference it
// from the same module; deriveRole itself does not use it.
export type { ChildSessionItem };

/** Flat descendant list + the live root → a nested tree. O(n); pure. */
export function buildTree(
  root: AgentTreeRoot,
  descendants: ChildSessionItem[],
): AgentTreeNode {
  const rootNode: AgentTreeNode = {
    sessionId: root.sessionId,
    parentSessionId: null,
    workerType: "root",
    toolFilterPreset: null,
    role: "root",
    status: root.status,
    title: root.title,
    createdAt: root.createdAt,
    updatedAt: root.updatedAt,
    children: [],
  };

  const byId = new Map<string, AgentTreeNode>();
  byId.set(rootNode.sessionId, rootNode);
  for (const item of descendants) {
    if (item.id === root.sessionId) {
      continue; // never duplicate the root
    }
    byId.set(item.id, {
      sessionId: item.id,
      parentSessionId: item.parent_session_id,
      workerType: item.worker_type,
      toolFilterPreset: item.tool_filter_preset,
      role: deriveRole(item.worker_type, item.tool_filter_preset),
      status: item.status,
      title: item.title,
      createdAt: item.created_at,
      updatedAt: item.updated_at,
      children: [],
    });
  }

  for (const item of descendants) {
    if (item.id === root.sessionId) {
      continue;
    }
    const node = byId.get(item.id);
    if (!node) {
      continue;
    }
    const parent =
      item.parent_session_id !== null && item.parent_session_id !== item.id
        ? byId.get(item.parent_session_id)
        : undefined;
    if (parent && parent.sessionId !== node.sessionId) {
      parent.children.push(node);
    } else {
      rootNode.children.push(node); // orphan or self-parent → under root
    }
  }

  return rootNode;
}

/** Index every node in a built tree by sessionId (for O(1) lookup). */
export function flattenTree(root: AgentTreeNode): Record<string, AgentTreeNode> {
  const acc: Record<string, AgentTreeNode> = {};
  const walk = (node: AgentTreeNode) => {
    acc[node.sessionId] = node;
    node.children.forEach(walk);
  };
  walk(root);
  return acc;
}

const RUNNING_ISH: ReadonlySet<SessionStatus> = new Set<SessionStatus>([
  "pending",
  "running",
  "waiting",
  "finishing",
  "takeover_pending",
  "takeover",
]);

/** Elapsed ms. Running-ish → now − created; terminal → updated − created; null/unparseable → null. */
export function deriveElapsed(
  createdAt: string | null,
  updatedAt: string | null,
  status: SessionStatus,
  nowMs: number,
): number | null {
  if (!createdAt) {
    return null;
  }
  const created = Date.parse(createdAt);
  if (Number.isNaN(created)) {
    return null;
  }
  if (RUNNING_ISH.has(status)) {
    return Math.max(0, nowMs - created);
  }
  if (!updatedAt) {
    return null;
  }
  const updated = Date.parse(updatedAt);
  if (Number.isNaN(updated)) {
    return null;
  }
  return Math.max(0, updated - created);
}

/** Compact human-readable elapsed. `null` → "—". */
export function formatElapsed(ms: number | null): string {
  if (ms === null) {
    return "—";
  }
  const totalSec = Math.floor(ms / 1000);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  if (h > 0) {
    return `${h}h ${m}m`;
  }
  if (m > 0) {
    return `${m}m ${s}s`;
  }
  return `${s}s`;
}

import type { SessionEventRecord } from "@/lib/event-normalize";

export type AgentEventBundle = {
  sessionId: string;
  role: AgentRole;
  color: string;
  events: SessionEventRecord[]; // already normalized via the shared normalizeSessionEvents (INV-9)
};

export type MergedTimelineItem = {
  event: SessionEventRecord;
  sourceSessionId: string;
  sourceRole: AgentRole;
  sourceColor: string;
  sortKey: number;
};

export function eventSeq(rec: SessionEventRecord): number | null {
  const seq = rec.data.seq;
  return typeof seq === "number" ? seq : null;
}

export function eventSortKey(rec: SessionEventRecord): number {
  const createdAt = rec.data.created_at;
  if (typeof createdAt === "number") {
    return createdAt; // epoch seconds (backend: int(created_at.timestamp()))
  }
  if (typeof createdAt === "string") {
    const ms = Date.parse(createdAt);
    if (!Number.isNaN(ms)) {
      return ms / 1000; // ISO string → epoch seconds, comparable to the numeric branch
    }
  }
  return Number.POSITIVE_INFINITY; // absent / unparseable → sort last (deterministic)
}

export function countToolCalls(events: SessionEventRecord[]): number {
  const ids = new Set<string>();
  for (const rec of events) {
    if (rec.event !== "tool") {
      continue;
    }
    const id = rec.data.tool_call_id;
    if (typeof id === "string" && id.trim()) {
      ids.add(id);
    }
  }
  return ids.size;
}

const AGENT_PALETTE = [
  "#6366f1", "#10b981", "#f59e0b", "#ec4899", "#06b6d4",
  "#8b5cf6", "#ef4444", "#84cc16", "#f97316", "#14b8a6", "#a855f7",
];

/** Stable id → color map (cycles the palette). Root first → palette[0]. */
export function assignAgentColors(orderedSessionIds: string[]): Record<string, string> {
  const map: Record<string, string> = {};
  orderedSessionIds.forEach((id, index) => {
    map[id] = AGENT_PALETTE[index % AGENT_PALETTE.length];
  });
  return map;
}

/** Pure merge of PRE-NORMALIZED bundles. Cross-session order = created_at (second-granular, INV-4);
 *  seq is only an intra-session tiebreak. Stable. No extra dedup (each bundle already normalized). */
export function mergeAgentTimelines(bundles: AgentEventBundle[]): MergedTimelineItem[] {
  const staged = bundles.flatMap((bundle) =>
    bundle.events.map((rec, idx) => ({
      event: rec,
      sourceSessionId: bundle.sessionId,
      sourceRole: bundle.role,
      sourceColor: bundle.color,
      sortKey: eventSortKey(rec),
      intraSession: bundle.sessionId,
      intraOrder: eventSeq(rec) ?? idx,
    })),
  );
  staged.sort((a, b) => {
    if (a.sortKey !== b.sortKey) {
      return a.sortKey - b.sortKey;
    }
    if (a.intraSession !== b.intraSession) {
      return a.intraSession < b.intraSession ? -1 : 1;
    }
    return a.intraOrder - b.intraOrder;
  });
  return staged.map((item) => ({
    event: item.event,
    sourceSessionId: item.sourceSessionId,
    sourceRole: item.sourceRole,
    sourceColor: item.sourceColor,
    sortKey: item.sortKey,
  }));
}
