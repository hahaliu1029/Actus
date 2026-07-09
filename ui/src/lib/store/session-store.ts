"use client";

import { useMemo } from "react";
import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";

import { ApiError } from "@/lib/api/auth-utils";
import { fileApi } from "@/lib/api/file";
import { sessionApi } from "@/lib/api/session";
import { fetchCompactionList } from "@/lib/api/session-compaction";
import type {
  ChatParams,
  ChildOutcome,
  FileInfo,
  GetSessionFilesResponse,
  ListSessionItem,
  Session,
  SSEEventData,
  SupervisorSnapshot,
} from "@/lib/api/types";
import type { CompactionListItem } from "@/types/session-compaction";
import type { LocalCommandCard } from "@/lib/commands/types";
import { registerStoreResetter } from "@/lib/store/reset";
import { useUIStore } from "@/lib/store/ui-store";
import { normalizeSessionStatus } from "@/lib/utils/session-status";
import {
  assignAgentColors,
  buildTree,
  countToolCalls,
  flattenTree,
  mergeAgentTimelines,
  type AgentEventBundle,
  type AgentTreeNode,
  type MergedTimelineItem,
} from "@/lib/agent-tree";
import {
  applyProvisionalSignal,
  asRecord,
  createProvisionalState,
  eventIdOf,
  eventSemanticKey,
  pruneRecoveredLLMErrors,
  syncPlanStepsByStepEvent,
  upsertSessionEvent,
  type ProvisionalState,
  type SessionEventRecord,
} from "@/lib/event-normalize";
import { normalizeAndRouteSessionEvents } from "@/lib/lifecycle/dispatch";

// ---------------------------------------------------------------------------
// Phase 1 minimal subagent research — probe state slice
// ---------------------------------------------------------------------------

// "pending" is a UI-only marker for "child_started arrived but child_done has
// not"; the backend never sends it. The wire outcomes come from ChildOutcome.
export type ProbeChildOutcome = ChildOutcome | "pending";

export interface ProbeChildState {
  child_session_id: string;
  prompt: string;
  outcome: ProbeChildOutcome | null;
  final_answer: string | null;
}

export interface ProbeState {
  running: boolean;
  probe_run_id: string | null;
  children: ProbeChildState[];
  summary: string | null;
  validation_warnings: string[];
  // Codex R1 P1#2: surface preflight/stream errors so the panel can leave the
  // "进行中…" state when classifier / quota / conflict / network failures land.
  error: string | null;
}

type SessionState = {
  sessions: ListSessionItem[];
  activeSessionId: string | null;
  currentSession: Session | null;
  currentSessionFiles: FileInfo[];
  isLoadingSessions: boolean;
  isLoadingCurrentSession: boolean;
  isChatting: boolean;
  chatSessionId: string | null;
  chatAbort: (() => void) | null;
  sessionsAbort: (() => void) | null;
  _isRecovering: boolean;
  probeState: ProbeState;
  agentTree: AgentTreeState;
};

type SessionActions = {
  reset: () => void;
  setActiveSession: (sessionId: string | null) => void;
  loadAgentTree: (sessionId: string) => Promise<void>;
  refreshAgentTree: (sessionId: string) => Promise<void>;
  resetAgentTree: () => void;
  loadNodeCost: (childId: string) => Promise<void>;
  loadMergedTimeline: (sessionId: string) => Promise<void>;
  pollActiveAgents: (sessionId: string) => Promise<void>;
  isSessionStreaming: (sessionId: string) => boolean;
  getSessionStatus: (sessionId: string) => Session["status"] | null;
  fetchSessions: () => Promise<void>;
  streamSessions: () => void;
  stopStreamSessions: () => void;
  createSession: () => Promise<string>;
  fetchSessionById: (
    sessionId: string,
    options?: { silent?: boolean }
  ) => Promise<void>;
  fetchSessionFiles: (
    sessionId: string,
    options?: { silent?: boolean }
  ) => Promise<void>;
  sendChat: (sessionId: string, params: ChatParams) => Promise<void>;
  stopChat: () => void;
  updateSessionStatus: (sessionId: string, status: Session["status"]) => void;
  stopSession: (sessionId: string) => Promise<void>;
  deleteSession: (sessionId: string) => Promise<void>;
  clearUnread: (sessionId: string) => Promise<void>;
  uploadFile: (
    file: File,
    sessionId?: string,
    options?: { onProgress?: (loaded: number, total: number) => void; signal?: AbortSignal }
  ) => Promise<FileInfo>;
  downloadFile: (
    fileId: string,
    options?: { onProgress?: (loaded: number, total: number) => void; signal?: AbortSignal }
  ) => Promise<Blob>;
  downloadSandboxFile: (
    sessionId: string,
    filepath: string,
    options?: { onProgress?: (loaded: number, total: number) => void; signal?: AbortSignal }
  ) => Promise<Blob>;
  recoverSession: (sessionId: string) => Promise<void>;
  retryFromSuspend: (sessionId: string) => Promise<void>;
  mergeCompactionList: (items: CompactionListItem[]) => void;
  appendLocalCommandCard: (sessionId: string, card: LocalCommandCard) => void;
  // Phase 1 minimal subagent research
  getFilteredSessionsForList: () => ListSessionItem[];
  resetProbe: () => void;
  startProbe: (probeRunId: string, prompts: string[]) => void;
  updateChild: (childSessionId: string, update: Partial<ProbeChildState>) => void;
  updateChildByPrompt: (prompt: string, update: Partial<ProbeChildState>) => void;
  setProbeSummary: (summary: string, warnings: string[]) => void;
  setProbeError: (message: string) => void;
};

type SessionStore = SessionState & SessionActions;

// Re-export so existing importers (session-recovery.test.ts, session-mode-changed.test.ts) keep working.
export type { SessionEventRecord };

export type AgentTreeState = {
  rootId: string | null;
  root: AgentTreeNode | null;
  byId: Record<string, AgentTreeNode>;
  treeSignature: string | null; // structural signature → loadAgentTree skips no-op updates (R3 P3)
  loadSeq: number;              // monotonic load token → an older in-flight load can't clobber a newer one (R4 P2)
  mergeSeq: number;             // monotonic token for eventsByAgent writes (loadMergedTimeline/pollActiveAgents) (R5 P2)
  truncated: boolean;
  loading: boolean;
  error: string | null;
  lastFetchedAt: number | null;
  // C6b: per-node fetched cost snapshot (decimal string), keyed by session id.
  costById: Record<string, { totalUsd: string; status: string } | null>;
  // C6c: descendant event bundles (root events come from currentSession, INV-10).
  eventsByAgent: Record<
    string,
    { events: SessionEventRecord[]; lastSeq: number | null; lastEventId: string | null }
  >;
  agentColors: Record<string, string>;
  mergeLoading: boolean;
};

const initialAgentTree: AgentTreeState = {
  rootId: null,
  root: null,
  byId: {},
  treeSignature: null,
  loadSeq: 0,
  mergeSeq: 0,
  truncated: false,
  loading: false,
  error: null,
  lastFetchedAt: null,
  costById: {},
  eventsByAgent: {},
  agentColors: {},
  mergeLoading: false,
};

// Structural signature of the tree (per node: id + lineage + status + title + timestamps +
// role). loadAgentTree compares it so a 4s refresh with no real change keeps root/byId object
// identities stable — preventing the derived selectors (useMergedTimeline) and the cost effect
// from churning every poll (R3 P3). JSON.stringify each node so null / "" / undefined are
// DISTINCT (a plain join collapses null and "" to the same segment — R4 P3).
function agentTreeSignature(byId: Record<string, AgentTreeNode>): string {
  return Object.values(byId)
    .map((n) =>
      JSON.stringify([
        n.sessionId,
        n.parentSessionId,
        n.status,
        n.title,
        n.createdAt,
        n.updatedAt,
        n.role,
      ]),
    )
    .sort()
    .join("\n");
}

const initialProbeState: ProbeState = {
  running: false,
  probe_run_id: null,
  children: [],
  summary: null,
  validation_warnings: [],
  error: null,
};

const initialState: SessionState = {
  sessions: [],
  activeSessionId: null,
  currentSession: null,
  currentSessionFiles: [],
  isLoadingSessions: false,
  isLoadingCurrentSession: false,
  isChatting: false,
  chatSessionId: null,
  chatAbort: null,
  sessionsAbort: null,
  _isRecovering: false,
  probeState: initialProbeState,
  agentTree: initialAgentTree,
};

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

const CONTROL_MODE_STATUSES: ReadonlySet<string> = new Set([
  "running",
  "waiting",
  "takeover_pending",
  "takeover",
]);

const TERMINAL_OR_FINISHING_STATUSES: ReadonlySet<string> = new Set([
  "finishing",
  "completed",
  "timed_out",
]);

function resolveControlStatus(
  currentStatus: Session["status"],
  data: Record<string, unknown>
): Session["status"] {
  const action = asString(data.action).trim().toLowerCase();
  const reason = asString(data.reason).trim().toLowerCase();
  const handoffMode = asString(data.handoff_mode).trim().toLowerCase();

  if (action === "requested" || action === "reopened") {
    return "takeover_pending";
  }
  if (action === "started" || action === "renewed") {
    return "takeover";
  }
  if (action === "rejected") {
    if (reason === "terminate") {
      return "completed";
    }
    if (reason === "continue" || reason === "cancel_timeout") {
      return "running";
    }
    return currentStatus;
  }
  if (action === "ended") {
    return handoffMode === "continue" ? "running" : "completed";
  }
  if (action === "expired") {
    if (reason === "takeover_timeout") {
      return "takeover_pending";
    }
    if (reason === "pending_timeout") {
      return "completed";
    }
    return currentStatus;
  }
  if (action && process.env.NODE_ENV !== "production") {
    console.warn("[session-store] 未识别的 control action，保持当前状态", {
      action,
      data,
      currentStatus,
    });
  }
  return currentStatus;
}

function resolveStatusFromEvent(
  currentStatus: Session["status"],
  event: SSEEventData
): Session["status"] {
  if (event.type === "finishing") {
    return "finishing";
  }
  if (event.type === "wait" || event.type === "tool_confirmation") {
    return "waiting";
  }
  if (event.type === "session_mode_changed") {
    // A4-0: terminal/finishing precedence — a stale control-mode signal must
    // never regress a session already in finishing/completed/timed_out.
    if (TERMINAL_OR_FINISHING_STATUSES.has(currentStatus)) {
      return currentStatus;
    }
    const to = asString(asRecord(event.data).to);
    return CONTROL_MODE_STATUSES.has(to)
      ? (to as Session["status"])
      : currentStatus;
  }
  // D5: Watchdog health events. TERMINATED pins the session to timed_out so
  // the subsequent "done" event doesn't collapse it back to "completed".
  if (event.type === "health") {
    const data = asRecord(event.data);
    const healthStatus = typeof data.status === "string" ? data.status : "";
    if (healthStatus === "terminated" || healthStatus === "terminating") {
      return "timed_out";
    }
    // DEGRADED is informational — keep current status (running/finishing).
    return currentStatus;
  }
  // PR2: Sandbox lifecycle state change — only terminal `destroyed` pins status.
  // Preserve timed_out (same logic as done/error) so watchdog semantics aren't lost.
  if (event.type === "sandbox_state_changed") {
    const data = asRecord(event.data);
    const newState = typeof data.new_state === "string" ? data.new_state : "";
    if (newState === "destroyed") {
      return currentStatus === "timed_out" ? "timed_out" : "completed";
    }
    // `destroying` is transient — keep current status until `destroyed` arrives
    return currentStatus;
  }
  if (event.type === "done" || event.type === "error") {
    // D5: Preserve timed_out (set by preceding health event) across done.
    if (currentStatus === "timed_out") {
      return "timed_out";
    }
    return "completed";
  }
  if (event.type === "owner_conflict") {
    return currentStatus;
  }
  if (event.type === "control") {
    return resolveControlStatus(currentStatus, asRecord(event.data));
  }
  // Control-mode + finishing/terminal guard: bare content events must not
  // regress a non-running control mode (takeover / takeover_pending) or a
  // finishing/timed_out state back to running. The control mode only changes via
  // session_mode_changed / control events (handled above, before this point).
  if (
    currentStatus === "finishing" ||
    currentStatus === "timed_out" ||
    currentStatus === "takeover" ||
    currentStatus === "takeover_pending"
  ) {
    return currentStatus;
  }
  return "running";
}

function optionalNullableString(
  record: Record<string, unknown>,
  key: string,
  fallback: string | null | undefined
): string | null {
  if (!Object.prototype.hasOwnProperty.call(record, key)) {
    return fallback ?? null;
  }
  const value = record[key];
  return typeof value === "string" ? value : null;
}

function isSupervisorExecutionMode(
  value: unknown
): value is SupervisorSnapshot["execution_mode"] {
  return value === "foreground" || value === "background";
}

function isSupervisorExecutionPhase(
  value: unknown
): value is SupervisorSnapshot["execution_phase"] {
  return (
    value === "running" ||
    value === "recovering" ||
    value === "idle" ||
    value === "suspended" ||
    value === "terminating" ||
    value === "terminated"
  );
}

function isSupervisorBackgroundReason(
  value: unknown
): value is NonNullable<SupervisorSnapshot["background_reason"]> {
  return value === "explicit" || value === "auto_degrade";
}

function isBackgroundSuspendedSession(
  session: Session | ListSessionItem | null | undefined
): boolean {
  return (
    session?.supervisor_snapshot?.execution_mode === "background" &&
    session.supervisor_snapshot.execution_phase === "suspended"
  );
}

function retryExpiresAtToSnapshotValue(
  expiresAt: number | null | undefined
): string | null {
  if (typeof expiresAt !== "number" || !Number.isFinite(expiresAt)) {
    return null;
  }
  return new Date(expiresAt * 1000).toISOString();
}

function supervisorSnapshotFromExecutionStateEvent(
  event: SessionEventRecord,
  base: SupervisorSnapshot | null | undefined
): SupervisorSnapshot | null {
  if (event.event !== "execution_state_changed") {
    return base ?? null;
  }

  const payload = asRecord(event.data?.payload);
  const executionMode = payload.execution_mode;
  const executionPhase = payload.execution_phase;
  if (
    !isSupervisorExecutionMode(executionMode) ||
    !isSupervisorExecutionPhase(executionPhase)
  ) {
    return base ?? null;
  }

  const rawBackgroundReason = payload.background_reason;
  const backgroundReason =
    rawBackgroundReason == null
      ? null
      : isSupervisorBackgroundReason(rawBackgroundReason)
        ? rawBackgroundReason
        : base?.background_reason ?? null;
  const retryBudget =
    typeof payload.retry_budget_remaining === "number" &&
    Number.isFinite(payload.retry_budget_remaining)
      ? payload.retry_budget_remaining
      : base?.retry_budget_remaining ?? 0;

  return {
    execution_mode: executionMode,
    execution_phase: executionPhase,
    background_reason: backgroundReason,
    expires_at: optionalNullableString(payload, "expires_at", base?.expires_at),
    retry_budget_remaining: retryBudget,
    suspended_reason: optionalNullableString(
      payload,
      "suspended_reason",
      base?.suspended_reason
    ),
    terminal_reason: optionalNullableString(
      payload,
      "terminal_reason",
      base?.terminal_reason
    ),
    last_progress_at: base?.last_progress_at ?? null,
    is_alive: base?.is_alive ?? executionPhase === "running",
    cancellation_state: base?.cancellation_state ?? "none",
  };
}

function supervisorSnapshotFromEvents(
  events: SessionEventRecord[],
  base: SupervisorSnapshot | null | undefined
): SupervisorSnapshot | null {
  return events.reduce<SupervisorSnapshot | null>(
    (snapshot, event) => supervisorSnapshotFromExecutionStateEvent(event, snapshot),
    base ?? null
  );
}

function mergeSupervisorSnapshotByCursor(
  remoteSnapshot: SupervisorSnapshot | null | undefined,
  remoteLastSeq: number,
  localSnapshot: SupervisorSnapshot | null | undefined,
  localLastSeq: number | null | undefined
): SupervisorSnapshot | null {
  if (!remoteSnapshot) {
    return localSnapshot ?? null;
  }
  const localCursor =
    typeof localLastSeq === "number" && Number.isFinite(localLastSeq)
      ? localLastSeq
      : 0;
  return remoteLastSeq >= localCursor ? remoteSnapshot : localSnapshot ?? null;
}

// B1-2: read/write the session-scoped provisional-CALLING state machine. Optional
// fields + fallbacks (`?? 0` / `?? true`) keep every existing Session construction
// and fixture unchanged.
function provisionalStateOf(session: Session): ProvisionalState {
  return {
    watermark: session.provisional_prune_watermark ?? 0,
    turnClosed: session.provisional_turn_closed ?? true,
  };
}

function writeProvisionalState(
  session: Session,
  state: ProvisionalState
): Session {
  return {
    ...session,
    provisional_prune_watermark: state.watermark,
    provisional_turn_closed: state.turnClosed,
  };
}

// B1-2: the four event branches (tool/message/done/error) that carry
// provisional-CALLING lifecycle semantics. The helper replaces ONLY the upsert
// step — each branch's existing post-processing (plan-step sync, recovered-LLM
// -error prune) runs on the helper's output (R13#2).
const PROVISIONAL_SIGNAL_TYPES = new Set(["tool", "message", "done", "error"]);

function applySSEToSession(session: Session, event: SSEEventData): Session {
  const sessionWithSeq = advanceSessionLastSeq(session, eventSeqOf({
    event: event.type,
    data: event.data as Record<string, unknown>,
  }));

  if (event.type === "execution_state_changed") {
    const nextEvent: SessionEventRecord = {
      event: event.type,
      data: event.data as Record<string, unknown>,
    };
    return {
      ...sessionWithSeq,
      supervisor_snapshot: supervisorSnapshotFromExecutionStateEvent(
        nextEvent,
        sessionWithSeq.supervisor_snapshot
      ),
      events: upsertSessionEvent(
        sessionWithSeq.events as SessionEventRecord[],
        nextEvent
      ),
    };
  }

  if (event.type === "title") {
    const nextTitle =
      typeof event.data.title === "string" ? event.data.title : sessionWithSeq.title;
    return {
      ...sessionWithSeq,
      title: nextTitle,
    };
  }

  const nextEvent: SessionEventRecord = {
    event: event.type,
    data: event.data as Record<string, unknown>,
  };

  // B1-2: route tool/message/done/error through the provisional state machine.
  // The helper replaces ONLY the upsert step; each branch's existing
  // post-processing is preserved (R13#2). For `done` the helper returns the
  // events with residual provisional cards pruned but does NOT persist the
  // `done` event itself — matching the historical early-return that stored no
  // `done` event and ran no extra passes.
  if (PROVISIONAL_SIGNAL_TYPES.has(event.type)) {
    const provisional = applyProvisionalSignal(
      provisionalStateOf(session),
      session.events as SessionEventRecord[],
      nextEvent
    );
    // `done` historically did NOT run pruneRecoveredLLMErrors (it early-returned
    // sessionWithSeq); keep that exact behavior so INV-0 holds flag-OFF.
    const withRecoveredErrorsPruned =
      event.type === "done" || event.type === "error"
        ? provisional.events
        : pruneRecoveredLLMErrors(provisional.events);
    return writeProvisionalState(
      {
        ...sessionWithSeq,
        events: withRecoveredErrorsPruned,
      },
      provisional.state
    );
  }

  const events = upsertSessionEvent(
    session.events as SessionEventRecord[],
    nextEvent
  );
  const withPlanStepSynced =
    event.type === "step"
      ? syncPlanStepsByStepEvent(events, nextEvent)
      : events;
  const withRecoveredErrorsPruned = pruneRecoveredLLMErrors(withPlanStepSynced);

  return {
    ...sessionWithSeq,
    events: withRecoveredErrorsPruned,
  };
}

// [CXR2-P1-3] Test-only export so vitest can drive the real SSE merge path
// without bypassing upsertSessionEvent / eventSemanticKey. Do not import from
// production code — the `__test_` prefix marks this as a test-time API.
export const __test_applySSEToSession = applySSEToSession;

function pickTitle(session: Session): string | null {
  if (session.title) {
    return session.title;
  }
  const titleEvent = [...(session.events as SessionEventRecord[])]
    .reverse()
    .find((item) => item.event === "title");
  const title = titleEvent?.data?.title;
  return typeof title === "string" ? title : null;
}

function eventSeqOf(event: SessionEventRecord): number | null {
  const rawSeq = event.data?.seq;
  if (typeof rawSeq === "number" && Number.isFinite(rawSeq) && rawSeq > 0) {
    return rawSeq;
  }
  if (typeof rawSeq === "string" && rawSeq.trim()) {
    const parsed = Number(rawSeq);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
  }
  return null;
}

function maxSeqFromEvents(events: SessionEventRecord[]): number {
  return events.reduce((max, event) => {
    const seq = eventSeqOf(event);
    return seq === null ? max : Math.max(max, seq);
  }, 0);
}

function advanceSessionLastSeq(session: Session, seq: number | null): Session {
  if (seq === null) {
    return session;
  }
  const currentLastSeq = session.last_seq ?? 0;
  const nextLastSeq = Math.max(currentLastSeq, seq);
  if (nextLastSeq === currentLastSeq) {
    return session;
  }
  return {
    ...session,
    last_seq: nextLastSeq,
  };
}

function getLatestEventId(events: SessionEventRecord[]): string | undefined {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (!event) {
      continue;
    }
    const id = eventIdOf(event);
    if (id) {
      return id;
    }
  }
  return undefined;
}

// B1 R3-FIX (recovery≡live): when two events collide on the same `tool:<id>`
// semantic key, keep the one with the HIGHER seq instead of blindly letting the
// second argument win. mergeSessionEvents is called with OPPOSITE argument
// orders at its two sites — fetchSessionById passes (remote, local) so LOCAL
// wins, but recoverSession passes (local, recovered) so the RECOVERED payload
// wins. Under a cursor race the recovery response can carry a stale
// CALLING(a, seq=10) built before the FE's live CALLED(a, seq=11) landed;
// second-arg-wins would drop the local CALLED, and applyProvisionalReplay then
// folds from the LOCAL watermark (11) over a rebuilt list that no longer
// contains the CALLED — shouldDropReplayedCalling(10 < 11) fires and tool card
// `a` VANISHES, violating recovery≡live (R1#5 never-downgrade can't fire because
// the fold's rebuilt list never sees the local CALLED). A seq-compare guard
// (rather than a status-rank never-downgrade) is used because seq is the
// monotonic authority the whole B1 anti-replay state machine already keys on and
// it is direction-agnostic — it protects BOTH argument orders symmetrically.
// The guard is scoped to `tool:` keys ONLY: message/step/plan/compaction keep
// their existing overwrite semantics untouched. seq semantics follow
// eventSeqOf (null when absent/non-positive): an incoming event with no seq is
// treated as NOT-newer when the existing entry has a seq; when both lack a seq
// we fall back to the original overwrite (last-writer-wins).
function isToolSemanticKey(key: string | null): boolean {
  return key !== null && key.startsWith("tool:");
}

function shouldKeepExistingToolEvent(
  key: string | null,
  existing: SessionEventRecord,
  incoming: SessionEventRecord
): boolean {
  if (!isToolSemanticKey(key)) {
    return false;
  }
  const existingSeq = eventSeqOf(existing);
  const incomingSeq = eventSeqOf(incoming);
  if (existingSeq === null) {
    return false; // no anchor to protect — overwrite as before
  }
  if (incomingSeq === null) {
    return true; // incoming has no seq → cannot prove it is newer → keep existing
  }
  return incomingSeq < existingSeq; // keep existing only when it is strictly newer
}

function mergeSessionEvents(
  remoteEvents: SessionEventRecord[],
  localEvents: SessionEventRecord[]
): SessionEventRecord[] {
  const merged = [...remoteEvents];
  const indexByKey = new Map<string, number>();

  merged.forEach((event, index) => {
    const key = eventSemanticKey(event);
    if (key) {
      indexByKey.set(key, index);
      return;
    }
    indexByKey.set(`remote:${index}:${event.event}`, index);
  });

  localEvents.forEach((event, index) => {
    const semanticKey = eventSemanticKey(event);
    const key = semanticKey || `local:${index}:${event.event}`;
    const existingIndex = indexByKey.get(key);
    if (existingIndex !== undefined) {
      // Tool-key collisions keep the higher-seq event (recovery≡live);
      // all other keys retain second-arg-wins overwrite semantics.
      if (
        shouldKeepExistingToolEvent(semanticKey, merged[existingIndex], event)
      ) {
        return;
      }
      merged[existingIndex] = event;
      return;
    }
    indexByKey.set(key, merged.length);
    merged.push(event);
  });

  return merged;
}

// B1-2: applyProvisionalReplay — from (local ProvisionalState, empty list), fold
// applyProvisionalSignal over the merge OUTPUT order to rebuild events and
// advance the state. Order = merge output order, NEVER re-sorted on the FE (R12#1:
// the backend recovery stream is already seq-ordered; a FE re-sort would move a
// stale event ahead of the upgrade/error boundary that raised the watermark,
// bypassing anti-replay). Replay result is authoritative (R12#2): watermark /
// turnClosed take the fold's terminal values — no "local OR replay" merge, since
// turnClosed is non-monotonic. Used at the recovery merge site and both
// fetchSessionById full-refresh merge sites so recovered / refetched events are
// equivalent to the fully-online live path (R18).
function applyProvisionalReplay(
  local: ProvisionalState,
  mergedEvents: SessionEventRecord[]
): { state: ProvisionalState; events: SessionEventRecord[] } {
  let state = local;
  let events: SessionEventRecord[] = [];
  for (const e of mergedEvents) {
    const out = applyProvisionalSignal(state, events, e);
    state = out.state;
    events = out.events;
  }
  return { state, events };
}

const STATUS_ORDER: Record<string, number> = {
  pending: 0,
  running: 1,
  waiting: 2,
  takeover_pending: 3,
  takeover: 4,
  finishing: 5,
  completed: 6,
  timed_out: 7,
};

export function pickMoreAdvancedStatus(
  ...statuses: (Session["status"] | null | undefined)[]
): Session["status"] | null {
  let best: Session["status"] | null = null;
  let bestOrder = -1;
  for (const s of statuses) {
    if (s == null) continue;
    const order = STATUS_ORDER[s] ?? -1;
    if (order > bestOrder) {
      bestOrder = order;
      best = s;
    }
  }
  return best;
}

const SIGNAL_EVENT_TYPES = new Set([
  "done", "error", "wait", "tool_confirmation",
  "control", "health", "finishing", "sandbox_state_changed",
  "session_mode_changed",
]);

/**
 * A4-0: the authoritative control mode = the `to` of the session_mode_changed
 * event with the MAX mode_revision (LWW). Ties / missing revisions fall back to
 * latest-by-array-order. Returns null when no control-mode signal is present.
 */
export function deriveLatestControlMode(
  events: SessionEventRecord[]
): Session["status"] | null {
  // NOTE: a plain `for` loop, NOT `events.forEach(...)`. Under TS `strict`,
  // assigning the closure-captured `best` inside a forEach callback narrows it
  // to `never` at the post-loop return (verified: `TS2339: Property 'mode' does
  // not exist on type 'never'`). The `for` loop keeps control-flow narrowing
  // local and compiles clean (R2#P1).
  let best: { rev: number | null; mode: Session["status"]; idx: number } | null =
    null;
  for (let idx = 0; idx < events.length; idx += 1) {
    const event = events[idx];
    if (event.event !== "session_mode_changed") {
      continue;
    }
    const data = asRecord(event.data);
    const to = asString(data.to);
    if (!CONTROL_MODE_STATUSES.has(to)) {
      continue;
    }
    const rawRev = data.mode_revision;
    const rev: number | null =
      typeof rawRev === "number" && Number.isFinite(rawRev) ? rawRev : null;
    if (best === null) {
      best = { rev, mode: to as Session["status"], idx };
      continue;
    }
    // Both sides carry a revision → LWW by revision (tie → later array order).
    // Either side MISSING a revision (in-txn read-fail → omitted, INV-2) → fall
    // back to event order: the later event wins (idx strictly increases).
    const wins =
      rev !== null && best.rev !== null
        ? rev > best.rev || (rev === best.rev && idx > best.idx)
        : idx > best.idx;
    if (wins) {
      best = { rev, mode: to as Session["status"], idx };
    }
  }
  return best === null ? null : best.mode;
}

export function deriveStatusFromEvents(
  events: SessionEventRecord[]
): Session["status"] | null {
  let derived: Session["status"] = "running";
  let sawSignal = false;

  for (const event of events) {
    if (!SIGNAL_EVENT_TYPES.has(event.event)) {
      continue;
    }
    sawSignal = true;
    const sseEvent = {
      type: event.event,
      data: event.data ?? {},
    } as SSEEventData;
    derived = resolveStatusFromEvent(derived, sseEvent);
  }

  if (!sawSignal) {
    return null;
  }
  // E1 normalization: a historical "finishing" means the session completed.
  const normalizedDerived: Session["status"] =
    derived === "finishing" ? "completed" : derived;
  // A4-0: control mode (max mode_revision) is authoritative over event-order
  // derivation, EXCEPT when a lifecycle/terminal status already won (terminal
  // precedence — a stale control mode must not mask completed/timed_out).
  const controlMode = deriveLatestControlMode(events);
  if (
    controlMode !== null &&
    !TERMINAL_OR_FINISHING_STATUSES.has(normalizedDerived)
  ) {
    return controlMode;
  }
  return normalizedDerived;
}

/**
 * A4-0 (R7): resolve a merged session status. `deriveStatusFromEvents(events)`
 * is the order-aware authority — it already applies max-mode_revision LWW AND
 * terminal precedence by event order. When it yields a CONTROL mode, that mode
 * is authoritative (a stale local/remote status must not monotonically win);
 * otherwise fall back to the caller's existing monotonic pick.
 */
export function resolveMergedSessionStatus(
  events: SessionEventRecord[],
  remoteStatus: Session["status"] | null,
  monotonicFallback: Session["status"] | null
): Session["status"] | null {
  // Terminal precedence (R10#P1): the authoritative remote (DB) status wins when
  // it is terminal/finishing. `done`/terminal events are NOT in the event list —
  // `applySSEToSession` drops `done` (session-store.ts:417) — so the events alone
  // cannot carry terminal context; `remoteStatus` (the DB authority) is the only
  // reliable "is this session terminal now". A stale local control-mode event
  // must never resurrect a completed/timed_out session.
  if (remoteStatus !== null && TERMINAL_OR_FINISHING_STATUSES.has(remoteStatus)) {
    return remoteStatus;
  }
  // R9#P1: gate the override on a REAL session_mode_changed signal.
  // deriveStatusFromEvents ALSO derives control modes from LEGACY wait/control
  // events, so trusting it unconditionally would let a historical wait/control
  // pull the UI back on a log with NO A4-0 mode event (legacy/pre-A4-0 sessions).
  // deriveLatestControlMode considers ONLY session_mode_changed events.
  if (deriveLatestControlMode(events) === null) {
    return monotonicFallback;
  }
  // A mode signal exists and the session is NOT terminal → the order-aware
  // control mode (max mode_revision, with control inference) is authoritative
  // over the monotonic stale-local pick.
  return deriveStatusFromEvents(events) ?? monotonicFallback;
}

function showMessage(type: "success" | "error" | "info", text: string) {
  useUIStore.getState().setMessage({ type, text });
}

function stringifyEventData(data: Record<string, unknown>): string {
  try {
    return JSON.stringify(data);
  } catch {
    return "";
  }
}

function isSameEvents(
  left: SessionEventRecord[],
  right: SessionEventRecord[]
): boolean {
  if (left.length !== right.length) {
    return false;
  }

  for (let index = 0; index < left.length; index += 1) {
    const leftEvent = left[index];
    const rightEvent = right[index];
    if (!leftEvent || !rightEvent) {
      return false;
    }
    if (leftEvent.event !== rightEvent.event) {
      return false;
    }
    if (stringifyEventData(leftEvent.data) !== stringifyEventData(rightEvent.data)) {
      return false;
    }
  }

  return true;
}

function isSameSessionSnapshot(left: Session, right: Session): boolean {
  return (
    left.session_id === right.session_id &&
    left.status === right.status &&
    left.title === right.title &&
    (left.last_seq ?? 0) === (right.last_seq ?? 0) &&
    stringifySnapshot(left.supervisor_snapshot) ===
      stringifySnapshot(right.supervisor_snapshot) &&
    isSameEvents(
      (left.events || []) as SessionEventRecord[],
      (right.events || []) as SessionEventRecord[]
    )
  );
}

function stringifySnapshot(snapshot: unknown): string {
  try {
    return JSON.stringify(snapshot ?? null);
  } catch {
    return String(snapshot);
  }
}

function isSameFileList(left: FileInfo[], right: FileInfo[]): boolean {
  if (left.length !== right.length) {
    return false;
  }
  for (let index = 0; index < left.length; index += 1) {
    const leftFile = left[index];
    const rightFile = right[index];
    if (!leftFile || !rightFile) {
      return false;
    }
    if (
      leftFile.id !== rightFile.id ||
      leftFile.filename !== rightFile.filename ||
      leftFile.filepath !== rightFile.filepath ||
      leftFile.key !== rightFile.key ||
      leftFile.extension !== rightFile.extension ||
      leftFile.mime_type !== rightFile.mime_type ||
      leftFile.size !== rightFile.size
    ) {
      return false;
    }
  }
  return true;
}

function updateSessionListStatus(
  sessions: ListSessionItem[],
  sessionId: string,
  status: Session["status"]
): ListSessionItem[] {
  let changed = false;
  const next = sessions.map((item) => {
    if (item.session_id !== sessionId || item.status === status) {
      return item;
    }
    changed = true;
    return { ...item, status };
  });
  return changed ? next : sessions;
}

export const useSessionStore = create<SessionStore>()(
  subscribeWithSelector((set, get) => ({
    ...initialState,

    reset: () => {
      get().stopChat();
      get().stopStreamSessions();
      set(initialState);
    },

    setActiveSession: (sessionId: string | null) => {
      set((state) => {
        if (state.activeSessionId === sessionId) {
          return {};
        }
        return { activeSessionId: sessionId };
      });
    },

    loadAgentTree: async (sessionId: string) => {
      const state = get();
      // Readiness gate (R6-P2): only build once currentSession is the ready root.
      if (!state.currentSession || state.currentSession.session_id !== sessionId) {
        return;
      }
      if (state.activeSessionId && state.activeSessionId !== sessionId) {
        return;
      }
      // Capture a monotonic load token so an OLDER in-flight load can't overwrite a newer one
      // out of order (R4 P2 — e.g. a stale "running" poll landing after the terminal-final load).
      let loadSeq = 0;
      set((s) => {
        loadSeq = s.agentTree.loadSeq + 1;
        return {
          agentTree: { ...s.agentTree, loading: true, error: null, rootId: sessionId, loadSeq },
        };
      });
      try {
        const res = await sessionApi.getSessionChildren(sessionId, 1);
        // Re-check the guard after the await (user may have navigated; a newer load may have started).
        const after = get();
        if (
          after.agentTree.loadSeq !== loadSeq ||
          !after.currentSession ||
          after.currentSession.session_id !== sessionId ||
          (after.activeSessionId && after.activeSessionId !== sessionId)
        ) {
          return;
        }
        const root = buildTree(
          {
            sessionId,
            status: after.currentSession.status,
            title: after.currentSession.title,
            createdAt: null, // Session carries no timestamps (F0.3)
            updatedAt: null,
          },
          res.descendants,
        );
        const byId = flattenTree(root);
        const signature = agentTreeSignature(byId);
        set((s) => {
          // Structurally unchanged (e.g. a 4s poll with nothing new): keep the existing
          // root/byId object identities so derived selectors don't churn (R3 P3).
          if (s.agentTree.treeSignature === signature && s.agentTree.root) {
            return {
              agentTree: {
                ...s.agentTree,
                truncated: res.truncated,
                loading: false,
                error: null,
                lastFetchedAt: Date.now(),
              },
            };
          }
          return {
            agentTree: {
              ...s.agentTree,
              root,
              byId,
              treeSignature: signature,
              truncated: res.truncated,
              loading: false,
              error: null,
              lastFetchedAt: Date.now(),
            },
          };
        });
      } catch (err) {
        // Guard the error write too (R3 P2): a stale rejection after navigation must not
        // write an error into another (now-current) session's tree.
        const at = get();
        if (
          at.agentTree.loadSeq !== loadSeq ||
          !at.currentSession ||
          at.currentSession.session_id !== sessionId ||
          (at.activeSessionId && at.activeSessionId !== sessionId)
        ) {
          return;
        }
        set((s) => ({
          agentTree: {
            ...s.agentTree,
            loading: false,
            error: err instanceof Error ? err.message : "加载 Agent 树失败",
          },
        }));
      }
    },

    refreshAgentTree: async (sessionId: string) => {
      // Re-fetch is identical to load; the readiness gate + post-await guard make
      // it safe to call on SSE-invalidation or a bounded poll without clobbering.
      await get().loadAgentTree(sessionId);
    },

    resetAgentTree: () =>
      set((s) => ({
        agentTree: {
          ...initialAgentTree,
          // Keep the tokens MONOTONIC across reset (do NOT zero them) so an in-flight
          // load/fetch that predates the reset can't ABA-collide with a post-reset one (R5 P2).
          loadSeq: s.agentTree.loadSeq + 1,
          mergeSeq: s.agentTree.mergeSeq + 1,
        },
      })),

    loadNodeCost: async (childId: string) => {
      try {
        const res = await sessionApi.getSessionCost(childId);
        set((s) => ({
          agentTree: {
            ...s.agentTree,
            costById: {
              ...s.agentTree.costById,
              [childId]: { totalUsd: res.total_usd, status: res.cost_status },
            },
          },
        }));
      } catch {
        set((s) => ({
          agentTree: {
            ...s.agentTree,
            costById: { ...s.agentTree.costById, [childId]: null },
          },
        }));
      }
    },

    loadMergedTimeline: async (sessionId: string) => {
      const state = get();
      if (
        !state.currentSession ||
        state.currentSession.session_id !== sessionId ||
        (state.activeSessionId && state.activeSessionId !== sessionId)
      ) {
        return;
      }
      const descendantIds = Object.keys(state.agentTree.byId).filter((id) => id !== sessionId);
      // Tokenize this eventsByAgent write so an older full rebuild can't clobber a newer poll (R5 P2).
      let mergeSeq = 0;
      set((s) => {
        mergeSeq = s.agentTree.mergeSeq + 1;
        return { agentTree: { ...s.agentTree, mergeLoading: true, mergeSeq } };
      });
      const fetched = await Promise.all(
        descendantIds.map(async (id) => {
          try {
            const session = await sessionApi.getSession(id);
            const events = normalizeAndRouteSessionEvents(session.events as SessionEventRecord[]);
            const rawLastId = events.length ? events[events.length - 1].data.event_id : undefined;
            const lastEventId = typeof rawLastId === "string" ? rawLastId : null;
            return { id, events, lastSeq: session.last_seq ?? null, lastEventId };
          } catch {
            return null;
          }
        }),
      );
      // Re-check readiness after the awaits (user may have navigated). Guard on currentSession
      // AND activeSessionId (R1#2 — activeSessionId flips first on navigation) AND the merge token
      // so an older rebuild can't overwrite a newer poll's bundles (R5 P2).
      const after = get();
      if (
        after.agentTree.mergeSeq !== mergeSeq ||
        !after.currentSession ||
        after.currentSession.session_id !== sessionId ||
        (after.activeSessionId && after.activeSessionId !== sessionId)
      ) {
        return;
      }
      // Build FRESH, scoped to the current descendants (do NOT spread the existing map),
      // so a stale entry from a previous session can never survive into this merge (R1#2).
      const eventsByAgent: Record<
        string,
        { events: SessionEventRecord[]; lastSeq: number | null; lastEventId: string | null }
      > = {};
      for (const bundle of fetched) {
        if (bundle) {
          eventsByAgent[bundle.id] = {
            events: bundle.events,
            lastSeq: bundle.lastSeq,
            lastEventId: bundle.lastEventId,
          };
        }
      }
      set((s) => ({
        agentTree: {
          ...s.agentTree,
          eventsByAgent,
          agentColors: assignAgentColors([sessionId, ...descendantIds]),
          mergeLoading: false,
        },
      }));
    },

    pollActiveAgents: async (sessionId: string) => {
      const state = get();
      if (
        !state.currentSession ||
        state.currentSession.session_id !== sessionId ||
        (state.activeSessionId && state.activeSessionId !== sessionId)
      ) {
        return;
      }
      const { byId, eventsByAgent } = state.agentTree;
      const TERMINAL = new Set<string>(["completed", "timed_out"]);
      // Iterate the TREE (byId), not just already-fetched agents, so a descendant whose initial
      // fetch failed (or one that just appeared) is RETRIED instead of silently dropped (R4 P2).
      // Poll a descendant when it is non-terminal (may have new events) OR has no bundle yet
      // (never successfully fetched — a terminal-but-missing child still needs its one fetch).
      const targetIds = Object.keys(byId).filter((id) => {
        if (id === sessionId) {
          return false; // root events come from currentSession (INV-10)
        }
        const node = byId[id];
        if (!node) {
          return false;
        }
        return !TERMINAL.has(node.status) || !(id in eventsByAgent);
      });
      if (targetIds.length === 0) {
        return;
      }
      // Tokenize this eventsByAgent write (R5 P2): a newer merged load/poll bumps mergeSeq, so
      // this older write is dropped instead of overwriting fresher bundles.
      let mergeSeq = 0;
      set((s) => {
        mergeSeq = s.agentTree.mergeSeq + 1;
        return { agentTree: { ...s.agentTree, mergeSeq } };
      });
      const updates = await Promise.all(
        targetIds.map(async (id) => {
          const bundle = eventsByAgent[id];
          try {
            if (bundle && bundle.lastEventId != null) {
              // Incremental ONLY when we have an event-id cursor (BOTH cursors, INV-8). A bundle
              // with no event-id floor (e.g. a title-only child whose events all normalize away)
              // falls through to the full re-fetch below so we never degrade to a seq-only query.
              const res = await sessionApi.getEventsSince(
                id,
                bundle.lastEventId ?? undefined,
                bundle.lastSeq ?? undefined,
              );
              if (!res.events.length) {
                return null;
              }
              const events = normalizeAndRouteSessionEvents([
                ...bundle.events,
                ...(res.events as SessionEventRecord[]),
              ]);
              const rawLastId = events.length ? events[events.length - 1].data.event_id : undefined;
              const lastEventId = typeof rawLastId === "string" ? rawLastId : bundle.lastEventId;
              return { id, events, lastSeq: res.last_seq ?? bundle.lastSeq, lastEventId };
            }
            // No bundle (failed/new) OR a bundle with no event-id cursor → full fetch + normalize
            // (INV-9, never seq-only), like loadMergedTimeline.
            const session = await sessionApi.getSession(id);
            const events = normalizeAndRouteSessionEvents(session.events as SessionEventRecord[]);
            const rawLastId = events.length ? events[events.length - 1].data.event_id : undefined;
            const lastEventId = typeof rawLastId === "string" ? rawLastId : null;
            return { id, events, lastSeq: session.last_seq ?? null, lastEventId };
          } catch {
            return null;
          }
        }),
      );
      const after = get();
      if (
        after.agentTree.mergeSeq !== mergeSeq ||
        !after.currentSession ||
        after.currentSession.session_id !== sessionId ||
        (after.activeSessionId && after.activeSessionId !== sessionId)
      ) {
        return;
      }
      if (updates.every((u) => u === null)) {
        return;
      }
      const next = { ...after.agentTree.eventsByAgent };
      for (const u of updates) {
        if (u) {
          next[u.id] = { events: u.events, lastSeq: u.lastSeq, lastEventId: u.lastEventId };
        }
      }
      set((s) => ({ agentTree: { ...s.agentTree, eventsByAgent: next } }));
    },

    isSessionStreaming: (sessionId: string) => {
      const state = get();
      return state.isChatting && state.chatSessionId === sessionId;
    },

    getSessionStatus: (sessionId: string) => {
      const state = get();
      if (state.currentSession?.session_id === sessionId) {
        return state.currentSession.status;
      }
      return state.sessions.find((item) => item.session_id === sessionId)?.status ?? null;
    },

    fetchSessions: async () => {
      set({ isLoadingSessions: true });
      try {
        const sessions = await sessionApi.getSessions();
        set({
          sessions: sessions.map((s) => ({
            ...s,
            status: normalizeSessionStatus(s.status),
          })),
        });
      } catch (error) {
        showMessage(
          "error",
          error instanceof Error ? error.message : "加载会话失败"
        );
      } finally {
        set({ isLoadingSessions: false });
      }
    },

    streamSessions: () => {
      const previousAbort = get().sessionsAbort;
      if (previousAbort) {
        previousAbort();
      }

      const abort = sessionApi.streamSessions(
        (event) => {
          if (event.type !== "sessions") {
            return;
          }
          const remote = event.data.sessions;
          set((state) => {
            // 用远端列表为基础，但保留本地已通过 SSE chat 事件
            // 推进到 completed/waiting 的状态，避免竞态回退
            const localStatusMap = new Map(
              state.sessions.map((s) => [s.session_id, s.status])
            );
            const merged = remote.map((item) => {
              const normalizedStatus = normalizeSessionStatus(item.status);
              const localStatus = localStatusMap.get(item.session_id);
              if (
                localStatus &&
                localStatus !== normalizedStatus &&
                (localStatus === "completed" || localStatus === "waiting") &&
                normalizedStatus === "running"
              ) {
                return { ...item, status: localStatus };
              }
              return { ...item, status: normalizedStatus };
            });
            return { sessions: merged };
          });
        },
        (error) => {
          showMessage("error", error.message || "会话流连接异常");
        }
      );

      set({ sessionsAbort: abort });
    },

    stopStreamSessions: () => {
      const abort = get().sessionsAbort;
      if (abort) {
        abort();
      }
      set({ sessionsAbort: null });
    },

    createSession: async () => {
      const created = await sessionApi.createSession();
      await get().fetchSessions();
      showMessage("success", "新任务已创建");
      return created.session_id;
    },

    fetchSessionById: async (sessionId: string, options = {}) => {
      const silent = options.silent ?? false;
      if (!silent) {
        set({ isLoadingCurrentSession: true });
      }
      try {
        const session = await sessionApi.getSession(sessionId);
        const normalizedEvents = normalizeAndRouteSessionEvents(session.events as SessionEventRecord[]);
        const normalizedRemote: Session = {
          ...session,
          status: normalizeSessionStatus(session.status),
            title: pickTitle(session),
            events: normalizedEvents,
            last_seq: Math.max(session.last_seq ?? 0, maxSeqFromEvents(normalizedEvents)),
            supervisor_snapshot: session.supervisor_snapshot ?? null,
          };

        set((state) => {
          if (state.activeSessionId && state.activeSessionId !== sessionId) {
            return {};
          }

          const localSession = state.currentSession;
          if (!localSession || localSession.session_id !== sessionId) {
            // B1-2: no local session — fold the remote events from a fresh
            // provisional state so orphan CALLING cards from a crashed turn are
            // pruned and the watermark is rebuilt (R12#2). The remote payload
            // carries no watermark keys, so start from createProvisionalState().
            const freshReplay = applyProvisionalReplay(
              createProvisionalState(),
              normalizedRemote.events as SessionEventRecord[]
            );
            return {
              currentSession: writeProvisionalState(
                { ...normalizedRemote, events: freshReplay.events },
                freshReplay.state
              ),
            };
          }

          // B1-2: fold from the LOCAL provisional state over the merge output so
          // a full refresh is equivalent to the fully-online live path — the
          // remote payload's absent watermark keys are overwritten by the fold's
          // terminal state (R12#2); merge output order is authoritative (R12#1).
          const mergedEventsRaw = mergeSessionEvents(
            normalizedRemote.events as SessionEventRecord[],
            localSession.events as SessionEventRecord[]
          );
          const provisionalReplay = applyProvisionalReplay(
            provisionalStateOf(localSession),
            mergedEventsRaw
          );
          const mergedEvents = provisionalReplay.events;
          const nextLastSeq = Math.max(
            normalizedRemote.last_seq ?? 0,
            localSession.last_seq ?? 0,
            maxSeqFromEvents(mergedEvents)
          );
            const nextSession: Session = {
              ...normalizedRemote,
              title: normalizedRemote.title || localSession.title,
              events: mergedEvents,
              last_seq: nextLastSeq,
              provisional_prune_watermark: provisionalReplay.state.watermark,
              provisional_turn_closed: provisionalReplay.state.turnClosed,
              supervisor_snapshot: normalizedRemote.supervisor_snapshot ?? null,
              // E2 + A4-0 (R7): control-mode transitions (end-takeover→running,
              // reopen→takeover_pending) must win over a stale local status; a
              // later terminal/finishing event still takes precedence by order.
              // The monotonic pick is preserved as the fallback when no mode
              // signal is present (R9#P1).
              status:
                resolveMergedSessionStatus(
                  mergedEvents,
                  normalizedRemote.status, // remote (DB) = terminal authority (R10#P1)
                  pickMoreAdvancedStatus(
                    normalizedRemote.status,
                    localSession.status
                  ) ?? normalizedRemote.status
                ) ?? normalizedRemote.status,
          };

          if (isSameSessionSnapshot(localSession, nextSession)) {
            return {};
          }

          return {
            currentSession: nextSession,
          };
        });

        const stateAfterFetch = get();

        // [CXR4-P2-1] List-on-load: merge compaction list so reload-after-crash
        // recovers fold indicators. Anchor HERE — currentSession is now written
        // by the set(...) above, AND we hold the active-session guard so a user
        // who navigated away mid-fetch doesn't get list rows merged into the
        // wrong session.
        if (
          stateAfterFetch.currentSession?.session_id === sessionId &&
          stateAfterFetch.activeSessionId === sessionId
        ) {
          try {
            const items = await fetchCompactionList(sessionId);
            // Re-check the guard after the await — user may have navigated during fetch
            const stateAfterList = get();
            if (
              stateAfterList.currentSession?.session_id === sessionId &&
              stateAfterList.activeSessionId === sessionId
            ) {
              stateAfterList.mergeCompactionList(items);
            }
          } catch (err) {
            // Non-fatal — SSE will still drive live events
            console.warn("compaction list-on-load failed:", err);
          }
        }

        const fetchedSession =
          stateAfterFetch.currentSession &&
          stateAfterFetch.currentSession.session_id === sessionId
            ? stateAfterFetch.currentSession
            : null;
        const shouldResumeStream =
          fetchedSession?.status === "running" &&
          !isBackgroundSuspendedSession(fetchedSession) &&
          !stateAfterFetch.isChatting &&
          !stateAfterFetch.chatAbort;

        if (shouldResumeStream) {
          const latestEventId = getLatestEventId(
            (fetchedSession?.events || []) as SessionEventRecord[]
          );
          void get().sendChat(sessionId, { event_id: latestEventId });
        }
      } catch (error) {
        if (!silent) {
          showMessage(
            "error",
            error instanceof Error ? error.message : "加载会话详情失败"
          );
        }
      } finally {
        if (!silent) {
          const activeSessionId = get().activeSessionId;
          if (!activeSessionId || activeSessionId === sessionId) {
            set({ isLoadingCurrentSession: false });
          }
        }
      }
    },

    recoverSession: async (sessionId: string) => {
      const state = get();
      const localSession = state.currentSession;
      if (!localSession || localSession.session_id !== sessionId) {
        return;
      }
      if (localSession.status === "completed") {
        return;
      }
      if (get()._isRecovering) {
        return;
      }
      set({ _isRecovering: true });

      try {
        const lastEventId = getLatestEventId(
          localSession.events as SessionEventRecord[]
        );
        // B3-core PR-1 §3.3 — pass seq cursor for sequenced events while keeping
        // event_id as the backend's legacy-event fallback.
        const sinceSeq =
          typeof localSession.last_seq === "number" && localSession.last_seq > 0
            ? localSession.last_seq
            : undefined;
        const response = await sessionApi.getEventsSince(
          sessionId,
          lastEventId,
          sinceSeq,
        );

        const recoveredEvents = (response.events ?? []) as SessionEventRecord[];
        const remoteStatus = normalizeSessionStatus(response.session_status as Session["status"]);
        const eventDerivedStatus = deriveStatusFromEvents(recoveredEvents);

        // B3-core PR-1 §3.3 — capture supervisor cursor + snapshot for next reconnect.
        const remoteLastSeq = Math.max(
          typeof response.last_seq === "number" ? response.last_seq : 0,
          maxSeqFromEvents(recoveredEvents)
        );
        const remoteSnapshot = supervisorSnapshotFromEvents(
          recoveredEvents,
          response.supervisor_snapshot ?? null
        );

        // [Codex holistic R3+R4 P2] Backend's `/sessions/{id}/events?since=...`
        // returns persisted SSE events only — it does NOT include
        // conversation_compactions rows. If a Path A compaction was missed
        // (e.g., shielded background persist completed AFTER the cancelled
        // FINISHING branch — see agent_task_runner.py NOTE), an SSE reconnect
        // alone won't restore the fold indicator. Fire-and-forget the
        // compaction list refresh so it runs independently of the
        // events-branch status logic below — `mergeCompactionList` only
        // mutates `events`, never `status`, so racing with the status `set()`
        // is safe (idempotent dedup via `compaction:<id>` semantic key).
        // Awaiting here would delay the existing zero-event status sync
        // (breaking session-recovery tests' timing expectations).
        void (async () => {
          try {
            const items = await fetchCompactionList(sessionId);
            const stateAfterList = get();
            if (
              stateAfterList.currentSession?.session_id === sessionId &&
              stateAfterList.activeSessionId === sessionId
            ) {
              stateAfterList.mergeCompactionList(items);
            }
          } catch (err) {
            // Non-fatal — missing compactions will be recovered on next
            // session activation via `fetchSessionById`'s list-on-load.
            console.warn("compaction list-on-reconnect failed:", err);
          }
        })();

        if (recoveredEvents.length === 0) {
          set((s) => {
            const local = s.currentSession;
            if (!local || local.session_id !== sessionId) return {};

            // R5b-5 Codex round-8 HIGH: winner 已赢 claim 并开始执行但尚未产出
            // 新事件时，后端 session_status="running"，local 仍是 "waiting"。
            // pickMoreAdvancedStatus 里 waiting > running，zero-event 分支下
            // 会错误保留 waiting，UI 卡在 confirmation card 而非跳回聊天流。
            // 显式处理：local=waiting 且 remote 是非 waiting 的 active 状态时，
            // remote 胜出（winner 已跑完 claim→进入执行是合法恢复语义）。
            let finalStatus: Session["status"];
            if (
              local.status === "waiting" &&
              remoteStatus !== null &&
              remoteStatus !== "waiting"
            ) {
              finalStatus = remoteStatus;
            } else {
              const monotonic =
                pickMoreAdvancedStatus(remoteStatus, local.status) ??
                local.status;
              // A4-0 (R7/R10): a local control-mode event (e.g. a backward
              // takeover→running mode) wins over the monotonic pick, UNLESS the
              // remote (DB) status is terminal/finishing.
              finalStatus =
                resolveMergedSessionStatus(
                  local.events as SessionEventRecord[],
                  remoteStatus,
                  monotonic
                ) ?? monotonic;
            }
            // B3-core PR-1 — only short-circuit when nothing actually changes.
            const nextLastSeq = Math.max(remoteLastSeq, local.last_seq ?? 0);
            const nextSnapshot = mergeSupervisorSnapshotByCursor(
              remoteSnapshot,
              remoteLastSeq,
              local.supervisor_snapshot,
              local.last_seq
            );
            if (
              finalStatus === local.status &&
              local.last_seq === nextLastSeq &&
              local.supervisor_snapshot === nextSnapshot
            ) {
              return {};
            }
            return {
              currentSession: {
                ...local,
                status: finalStatus,
                // B3-core PR-1 — advance cursor monotonically; preserve local
                // snapshot when remote returns null. Mirrors the precomputed
                // values above so the equality short-circuit and the persisted
                // values agree.
                last_seq: nextLastSeq,
                supervisor_snapshot: nextSnapshot,
              },
            };
          });
          return;
        }

        const normalized = normalizeAndRouteSessionEvents(recoveredEvents);

        set((s) => {
          const local = s.currentSession;
          if (!local || local.session_id !== sessionId) return {};
          const mergedRaw = mergeSessionEvents(
            local.events as SessionEventRecord[],
            normalized
          );
          // B1-2: fold from the LOCAL provisional state over the merge output so a
          // reconnect-recovery is equivalent to the fully-online live path — a
          // replayed CALLING below the watermark cannot resurrect a pruned orphan,
          // and a recovered `done`/message still prunes residual provisional cards
          // (R18). Merge output order is authoritative — no FE re-sort (R12#1).
          const provisionalReplay = applyProvisionalReplay(
            provisionalStateOf(local),
            mergedRaw
          );
          const merged = provisionalReplay.events;
          const monotonicStatus =
            pickMoreAdvancedStatus(
              remoteStatus,
              eventDerivedStatus,
              local.status
            ) ?? local.status;
          // A4-0 (R7/R10): control mode (order-aware, max mode_revision) wins over
          // the monotonic pick, UNLESS the remote (DB) status is terminal/finishing
          // (resolveMergedSessionStatus checks `remoteStatus` first).
          const finalStatus =
            resolveMergedSessionStatus(merged, remoteStatus, monotonicStatus) ??
            monotonicStatus;
          const nextLastSeq = Math.max(remoteLastSeq, local.last_seq ?? 0);
          const nextSnapshot = mergeSupervisorSnapshotByCursor(
            remoteSnapshot,
            remoteLastSeq,
            local.supervisor_snapshot,
            local.last_seq
          );
          return {
            currentSession: {
              ...local,
              events: merged,
              provisional_prune_watermark: provisionalReplay.state.watermark,
              provisional_turn_closed: provisionalReplay.state.turnClosed,
              status: finalStatus,
              // B3-core PR-1 — advance cursor monotonically; preserve local snapshot
              // when remote returns null (avoids stomping good cursor on transient
              // backend that hasn't populated supervisor_snapshot yet).
              last_seq: nextLastSeq,
              supervisor_snapshot: nextSnapshot,
            },
          };
        });

        showMessage("info", `连接已恢复，已同步 ${recoveredEvents.length} 条新事件`);
      } catch {
        // 静默忽略，下次触发重试
      } finally {
        set({ _isRecovering: false });
      }
    },

    retryFromSuspend: async (sessionId: string) => {
      const result = await sessionApi.retryFromSuspend(sessionId);
      set((state) => {
        const current = state.currentSession;
        if (!current || current.session_id !== sessionId) {
          return {};
        }
        const previousSnapshot = current.supervisor_snapshot;
        return {
          currentSession: {
            ...current,
            status: normalizeSessionStatus(result.status),
            supervisor_snapshot: {
              execution_mode: "background",
              execution_phase: "running",
              background_reason: previousSnapshot?.background_reason ?? null,
              expires_at: retryExpiresAtToSnapshotValue(result.expires_at),
              retry_budget_remaining: result.retry_budget_remaining,
              suspended_reason: null,
              terminal_reason: null,
              last_progress_at: previousSnapshot?.last_progress_at ?? null,
              is_alive: true,
              cancellation_state: "none",
            },
          },
        };
      });
      const refreshes = [get().fetchSessions()];
      if (get().currentSession?.session_id === sessionId) {
        refreshes.push(get().fetchSessionById(sessionId, { silent: true }));
      }
      await Promise.all(refreshes);
    },

    mergeCompactionList: (items: CompactionListItem[]) => {
      set((state) => {
        if (!state.currentSession) return state;
        if (items.length === 0) return {};
        const existing = (state.currentSession.events ?? []) as SessionEventRecord[];
        let next: SessionEventRecord[] = existing;
        for (const item of items) {
          const synthetic: SessionEventRecord = {
            event: "compaction",
            data: {
              compaction_id: item.compaction_id,
              // P2 fix: synthetic events MUST NOT carry event_id.
              // eventSemanticKey() already returns `compaction:<id>` for
              // events that have compaction_id, so event_id is never read
              // for dedup. A synthetic `list-${compaction_id}` id would
              // become the `latestEventId` cursor and be sent as
              // `?since=list-${id}` on reconnect — a backend-unknown id
              // that breaks the SSE replay cursor contract.
              level: item.kinds.includes("hard_truncate") ? 3 : 2,
              tokens_before: item.tokens_before_total,
              tokens_after: item.tokens_after_total,
              messages_removed: item.messages_removed_total,
              usage_ratio_after: 0,
              created_at: item.created_at,
            },
          };
          next = upsertSessionEvent(next, synthetic);  // reuses existing dedup path
        }
        if (next === existing) return {};
        return {
          ...state,
          currentSession: { ...state.currentSession, events: next },
        };
      });
    },

    appendLocalCommandCard: (sessionId: string, card: LocalCommandCard) => {
      set((state) => {
        if (!state.currentSession || state.currentSession.session_id !== sessionId) {
          return {};
        }
        const existing = (state.currentSession.events ?? []) as SessionEventRecord[];
        const synthetic: SessionEventRecord = {
          event: "message",
          data: {
            role: card.role,
            message: card.markdown,
            // INV-B11-2: synthetic cards MUST NOT carry event_id or seq — they
            // would poison the SSE replay cursor / provisional watermark (F3).
            // stream_id carries a local-cmd- prefix + uuid for semantic-key dedup.
            created_at: Math.floor(Date.now() / 1000),
            attachments: [],
            stream_id: `local-cmd-${card.commandName}-${card.role}-${crypto.randomUUID()}`,
          },
        };
        const next = upsertSessionEvent(existing, synthetic);
        return { ...state, currentSession: { ...state.currentSession, events: next } };
      });
    },

    fetchSessionFiles: async (sessionId: string, options = {}) => {
      const silent = options.silent ?? false;
      try {
        const result: GetSessionFilesResponse =
          await sessionApi.getSessionFiles(sessionId);
        set((state) => {
          if (state.activeSessionId && state.activeSessionId !== sessionId) {
            return {};
          }
          if (isSameFileList(state.currentSessionFiles, result.files)) {
            return {};
          }
          return { currentSessionFiles: result.files };
        });
      } catch (error) {
        if (!silent) {
          showMessage(
            "error",
            error instanceof Error ? error.message : "加载会话文件失败"
          );
        }
      }
    },

    sendChat: async (sessionId, params) => {
      get().stopChat();

      // Reset FINISHING → RUNNING before opening new SSE
      const currentSession = get().currentSession;
      if (
        currentSession?.session_id === sessionId &&
        currentSession.status === "finishing"
      ) {
        get().updateSessionStatus(sessionId, "running");
      }

      set({ isChatting: true, chatSessionId: sessionId });

      const current = get().currentSession;
      const fallbackEventId =
        params.event_id ??
        (current?.session_id === sessionId
          ? getLatestEventId((current.events || []) as SessionEventRecord[])
          : undefined);
      const requestParams = fallbackEventId
        ? {
            ...params,
            event_id: fallbackEventId,
          }
        : params;

      let abortRef: (() => void) | null = null;
      let shouldClearAbortAfterBind = false;
      let sawTerminalEvent = false;
      let streamConnected = false;

      const clearChatState = () => {
        if (!abortRef) {
          shouldClearAbortAfterBind = true;
          set({ isChatting: false, chatSessionId: null });
          return;
        }
        set((state) => {
          if (!abortRef || state.chatAbort !== abortRef) {
            return {};
          }
          return { isChatting: false, chatSessionId: null, chatAbort: null };
        });
      };

      const abort = sessionApi.chat(
        sessionId,
        requestParams,
        (event) => {
          if (
            event.type === "tool" &&
            (event.data.name === "file" || String(event.data.name || "").startsWith("file_")) &&
            event.data.status === "called"
          ) {
            void get().fetchSessionFiles(sessionId);
          }

          // E2: 标记是否收到终止事件（在 set() 外部）
          if (
            event.type === "done" ||
            event.type === "error" ||
            event.type === "wait" ||
            event.type === "tool_confirmation" ||
            event.type === "control" ||
            event.type === "owner_conflict" ||
            event.type === "sandbox_state_changed"
          ) {
            sawTerminalEvent = true;
          }

          set((state) => {
            if (process.env.NODE_ENV === "development") {
              console.debug("[session-store] chat-event", {
                session_id: sessionId,
                event_type: event.type,
                chat_session_id: state.chatSessionId,
                is_chatting: state.isChatting,
              });
            }

            const current =
              state.currentSession && state.currentSession.session_id === sessionId
                ? state.currentSession
                : ({
                    session_id: sessionId,
                    title: null,
                    status: "running",
                    events: [],
                  } as Session);
            const currentStatus = current.status || "running";
            const nextStatus = resolveStatusFromEvent(currentStatus, event);
            const nextSessions = updateSessionListStatus(
              state.sessions,
              sessionId,
              nextStatus
            );

            if (state.activeSessionId && state.activeSessionId !== sessionId) {
              return nextSessions === state.sessions ? {} : { sessions: nextSessions };
            }

            const next = applySSEToSession(current, event);

            // A4-0: session_mode_changed updates the status authority ONLY — it
            // does not end the stream or reset streaming flags (the trigger
            // Wait/Control event that follows owns stream-end semantics). Without
            // this branch the event would fall through to the unknown-type
            // fallback below and wrongly reset status to "running".
            if (event.type === "session_mode_changed") {
              // Terminal precedence (R10#P1): if the session is already
              // terminal/finishing, keep it — a stale live mode event must not
              // resurrect it. `nextStatus = resolveStatusFromEvent(currentStatus,
              // event)` returns the (terminal) currentStatus when it is terminal
              // (the session_mode_changed case is terminal-guarded), and `done`
              // is not in `next.events`, so we must check nextStatus, not derive
              // from events. Otherwise: deriveStatusFromEvents over the accumulated
              // `next.events` is the order-aware authority (max-mode_revision LWW),
              // matching the merge paths.
              const liveStatus = TERMINAL_OR_FINISHING_STATUSES.has(nextStatus)
                ? nextStatus
                : deriveStatusFromEvents(next.events as SessionEventRecord[]) ??
                  nextStatus;
              const liveSessions = updateSessionListStatus(
                state.sessions,
                sessionId,
                liveStatus
              );
              return {
                currentSession: { ...next, status: liveStatus },
                sessions: liveSessions,
              };
            }

            const shouldResetStreaming = state.chatSessionId === sessionId;

            if (
              event.type === "done" ||
              event.type === "wait" ||
              event.type === "tool_confirmation" ||
              event.type === "error" ||
              event.type === "control" ||
              event.type === "owner_conflict" ||
              event.type === "finishing" ||
              event.type === "health" ||
              event.type === "sandbox_state_changed"
            ) {
              const isFinishing = event.type === "finishing";
              const isHealth = event.type === "health";
              shouldClearAbortAfterBind = !isFinishing && !isHealth;
              return {
                currentSession: {
                  ...next,
                  status: nextStatus,
                },
                sessions: nextSessions,
                ...(shouldResetStreaming && !isFinishing && !isHealth
                  ? { isChatting: false, chatSessionId: null }
                  : isFinishing
                    ? { isChatting: false }
                    : {}),
              };
            }

            // A4-0 follow-up (b): nextStatus = resolveStatusFromEvent(...) is the
            // single source of truth for the status a content event implies. It
            // already preserves BOTH timed_out (:276-279) and finishing
            // (:272-274). The old hand-rolled fallbackStatus only remembered
            // timed_out, so a content/compaction event arriving while finishing
            // regressed the open session to "running" while the list (also fed
            // nextStatus at the top of this reducer) stayed "finishing" — a
            // split-brain. Reusing nextStatus makes detail == list by construction.
            return {
              currentSession: {
                ...next,
                status: nextStatus,
              },
              sessions: nextSessions,
            };
          });
        },
        (error) => {
          // R5b-5: tool_confirmation 提交收到 HTTP 409（losing-claim / late-duplicate
          // / status=processing）→ 走 /events?since=<last_event_id> 自动 reconnect 复播
          // 已完成的 tool result events，而不是给用户弹错误 toast。409 是"已被处理"
          // 的合法契约信号，背面是 winner 已在跑或已跑完。
          if (
            requestParams.tool_confirmation &&
            error instanceof ApiError &&
            error.httpStatus === 409
          ) {
            clearChatState();
            setTimeout(() => {
              void get().recoverSession(sessionId);
            }, 0);
            return;
          }

          // E2: 仅当 SSE 连接已建立且未收到终止事件时抑制错误（即将触发恢复）。
          // createSSEStream() 未成功时 streamConnected=false，必须报错。
          const isRecoverableDisconnect = streamConnected && !sawTerminalEvent;
          if (!isRecoverableDisconnect) {
            showMessage("error", error.message || "聊天流中断");
          }
          clearChatState();
        },
        () => {
          clearChatState();
          // E2: 仅在"SSE 连接已建立但意外断开"时触发恢复。
          // createSSEStream() 未成功时不触发（避免对 401/5xx 做无意义恢复）。
          if (streamConnected && !sawTerminalEvent) {
            const sessionAfterClose = get().currentSession;
            if (
              sessionAfterClose &&
              sessionAfterClose.session_id === sessionId
            ) {
              setTimeout(() => {
                void get().recoverSession(sessionId);
              }, 2000);
            }
          }
        },
        // E2: onConnected — createSSEStream() 成功后调用
        () => {
          streamConnected = true;
        }
      );

      abortRef = abort;
      set({ chatAbort: abort });
      if (shouldClearAbortAfterBind) {
        clearChatState();
      }
    },

    stopChat: () => {
      const chatAbort = get().chatAbort;
      if (chatAbort) {
        chatAbort();
      }
      set({ chatAbort: null, isChatting: false, chatSessionId: null });
    },

    updateSessionStatus: (sessionId: string, status: Session["status"]) => {
      set((state) => ({
        currentSession:
          state.currentSession?.session_id === sessionId
            ? { ...state.currentSession, status }
            : state.currentSession,
        sessions: state.sessions.map((s) =>
          s.session_id === sessionId ? { ...s, status } : s
        ),
      }));
    },

    stopSession: async (sessionId: string) => {
      await sessionApi.stopSession(sessionId);
      if (get().chatSessionId === sessionId) {
        get().stopChat();
      }
      set((state) => ({
        sessions: updateSessionListStatus(state.sessions, sessionId, "completed"),
        currentSession:
          state.currentSession?.session_id === sessionId
            ? { ...state.currentSession, status: "completed" }
            : state.currentSession,
      }));
      showMessage("success", "任务已停止");
    },

    deleteSession: async (sessionId: string) => {
      await sessionApi.deleteSession(sessionId);
      const sessions = get().sessions.filter(
        (item) => item.session_id !== sessionId
      );
      set({ sessions });
      if (get().currentSession?.session_id === sessionId) {
        set({ currentSession: null, currentSessionFiles: [] });
      }
      showMessage("success", "任务已删除");
    },

    clearUnread: async (sessionId: string) => {
      await sessionApi.clearUnreadMessageCount(sessionId);
      set({
        sessions: get().sessions.map((item) =>
          item.session_id === sessionId
            ? { ...item, unread_message_count: 0 }
            : item
        ),
      });
    },

    uploadFile: async (file: File, sessionId?: string, options?: { onProgress?: (loaded: number, total: number) => void; signal?: AbortSignal }) => {
      const uploaded = await fileApi.uploadFile({
        file,
        session_id: sessionId,
        onProgress: options?.onProgress,
        signal: options?.signal,
      });
      set({ currentSessionFiles: [...get().currentSessionFiles, uploaded] });
      showMessage("success", `已上传文件：${uploaded.filename}`);
      return uploaded;
    },

    downloadFile: async (fileId: string, options?: { onProgress?: (loaded: number, total: number) => void; signal?: AbortSignal }) => {
      return fileApi.downloadFile(fileId, options);
    },

    downloadSandboxFile: async (sessionId: string, filepath: string, options?: { onProgress?: (loaded: number, total: number) => void; signal?: AbortSignal }) => {
      return sessionApi.downloadSandboxFile(sessionId, filepath, options);
    },

    // -----------------------------------------------------------------------
    // Phase 1 minimal subagent research
    // -----------------------------------------------------------------------

    getFilteredSessionsForList: () =>
      get().sessions.filter((s) => s.parent_session_id === null),

    resetProbe: () => set({ probeState: initialProbeState }),

    startProbe: (probeRunId, prompts) =>
      set({
        probeState: {
          running: true,
          probe_run_id: probeRunId,
          children: prompts.map((p) => ({
            child_session_id: "",
            prompt: p,
            outcome: null,
            final_answer: null,
          })),
          summary: null,
          validation_warnings: [],
          error: null,
        },
      }),

    updateChild: (childSessionId, update) =>
      set((st) => ({
        probeState: {
          ...st.probeState,
          children: st.probeState.children.map((c) =>
            c.child_session_id === childSessionId ? { ...c, ...update } : c,
          ),
        },
      })),

    // Match by prompt for the initial ChildStartedEvent payload — child_session_id
    // is empty until then. Update the FIRST row with matching prompt that has no
    // session_id; later ChildDoneEvent updates by child_session_id.
    updateChildByPrompt: (prompt, update) =>
      set((st) => {
        let updated = false;
        const children = st.probeState.children.map((c) => {
          if (!updated && c.prompt === prompt && !c.child_session_id) {
            updated = true;
            return { ...c, ...update };
          }
          return c;
        });
        return { probeState: { ...st.probeState, children } };
      }),

    setProbeSummary: (summary, warnings) =>
      set((st) => ({
        probeState: {
          ...st.probeState,
          running: false,
          summary,
          validation_warnings: warnings,
        },
      })),

    setProbeError: (message) =>
      set((st) => ({
        probeState: {
          ...st.probeState,
          running: false,
          error: message,
        },
      })),
  }))
);

// Hook export — selector returns filtered session list (excludes probe child
// sessions whose parent_session_id is non-null).
export function useFilteredSessionsForList(): ListSessionItem[] {
  const sessions = useSessionStore((s) => s.sessions);
  return useMemo(
    () => sessions.filter((s) => s.parent_session_id === null),
    [sessions]
  );
}

export function useMergedTimeline(): MergedTimelineItem[] {
  const rootId = useSessionStore((s) => s.agentTree.rootId);
  const rootEvents = useSessionStore((s) => s.currentSession?.events);
  const byId = useSessionStore((s) => s.agentTree.byId);
  const eventsByAgent = useSessionStore((s) => s.agentTree.eventsByAgent);
  const agentColors = useSessionStore((s) => s.agentTree.agentColors);
  return useMemo(() => {
    if (!rootId) {
      return [];
    }
    const bundles: AgentEventBundle[] = [
      {
        sessionId: rootId,
        role: "root",
        color: agentColors[rootId] ?? "#6366f1",
        events: (rootEvents ?? []) as SessionEventRecord[],
      },
    ];
    for (const [id, bundle] of Object.entries(eventsByAgent)) {
      bundles.push({
        sessionId: id,
        role: byId[id]?.role ?? "subagent",
        color: agentColors[id] ?? "#94a3b8",
        events: bundle.events,
      });
    }
    return mergeAgentTimelines(bundles);
  }, [rootId, rootEvents, byId, eventsByAgent, agentColors]);
}

export function useToolCallCount(nodeId: string): number | undefined {
  const rootId = useSessionStore((s) => s.agentTree.rootId);
  const rootEvents = useSessionStore((s) => s.currentSession?.events);
  const descEvents = useSessionStore((s) => s.agentTree.eventsByAgent[nodeId]?.events);
  return useMemo(() => {
    const events = nodeId === rootId ? rootEvents : descEvents;
    if (!events) {
      return undefined;
    }
    return countToolCalls(events as SessionEventRecord[]);
  }, [nodeId, rootId, rootEvents, descEvents]);
}

registerStoreResetter("session", () => {
  useSessionStore.getState().reset();
});
