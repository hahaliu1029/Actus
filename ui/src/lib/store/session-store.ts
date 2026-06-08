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
import { registerStoreResetter } from "@/lib/store/reset";
import { useUIStore } from "@/lib/store/ui-store";
import { normalizeSessionStatus } from "@/lib/utils/session-status";

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
};

type SessionActions = {
  reset: () => void;
  setActiveSession: (sessionId: string | null) => void;
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

export type SessionEventRecord = {
  event: string;
  data: Record<string, unknown>;
};

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
};

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null
    ? (value as Record<string, unknown>)
    : {};
}

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

  if (event.type === "done") {
    return sessionWithSeq;
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

  const events = upsertSessionEvent(
    session.events as SessionEventRecord[],
    nextEvent
  );
  const withPlanStepSynced =
    event.type === "step"
      ? syncPlanStepsByStepEvent(events, nextEvent)
      : events;
  const withRecoveredErrorsPruned =
    event.type !== "error"
      ? pruneRecoveredLLMErrors(withPlanStepSynced)
      : withPlanStepSynced;

  return {
    ...sessionWithSeq,
    events: withRecoveredErrorsPruned,
  };
}

// [CXR2-P1-3] Test-only export so vitest can drive the real SSE merge path
// without bypassing upsertSessionEvent / eventSemanticKey. Do not import from
// production code — the `__test_` prefix marks this as a test-time API.
export const __test_applySSEToSession = applySSEToSession;

function eventSemanticKey(event: SessionEventRecord): string | null {
  if (event.event === "message") {
    const streamId = event.data?.stream_id;
    if (typeof streamId === "string" && streamId.trim()) {
      return `message:${streamId}`;
    }
  }

  if (event.event === "plan") {
    return "plan:latest";
  }

  if (event.event === "tool") {
    const toolCallId = event.data?.tool_call_id;
    if (typeof toolCallId === "string" && toolCallId.trim()) {
      return `tool:${toolCallId}`;
    }
  }

  if (event.event === "step") {
    const stepId = event.data?.id;
    if (typeof stepId === "string" && stepId.trim()) {
      return `step:${stepId}`;
    }
  }

  if (event.event === "tool_confirmation") {
    const toolCallId = event.data?.tool_call_id;
    if (typeof toolCallId === "string" && toolCallId.trim()) {
      return `tool_confirmation:${toolCallId}`;
    }
    const eventId = event.data?.event_id;
    if (typeof eventId === "string" && eventId.trim()) {
      return `tool_confirmation:${eventId}`;
    }
  }

  if (event.event === "compaction") {
    const compactionId = event.data?.compaction_id;
    if (typeof compactionId === "string" && compactionId.trim()) {
      return `compaction:${compactionId}`;
    }
    // Legacy pre-B6 event with no compaction_id — fall through to event_id
  }

  const eventId = eventIdOf(event);
  if (eventId) {
    return `event:${eventId}`;
  }

  return null;
}

function upsertSessionEvent(
  events: SessionEventRecord[],
  nextEvent: SessionEventRecord
): SessionEventRecord[] {
  const nextKey = eventSemanticKey(nextEvent);
  if (!nextKey) {
    return [...events, nextEvent];
  }

  const existingIndex = events.findIndex(
    (item) => eventSemanticKey(item) === nextKey
  );
  if (existingIndex < 0) {
    return [...events, nextEvent];
  }

  const updated = [...events];
  updated[existingIndex] = nextEvent;
  return updated;
}

function normalizeSessionEvents(
  events: SessionEventRecord[]
): SessionEventRecord[] {
  let normalized: SessionEventRecord[] = [];
  events.forEach((event) => {
    if (event.event === "title") {
      return;
    }
    normalized = upsertSessionEvent(normalized, event);
    if (event.event === "step") {
      normalized = syncPlanStepsByStepEvent(normalized, event);
    }
  });
  return pruneRecoveredLLMErrors(normalized);
}

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

function syncPlanStepsByStepEvent(
  events: SessionEventRecord[],
  stepEvent: SessionEventRecord
): SessionEventRecord[] {
  const stepId = stepEvent.data?.id;
  if (typeof stepId !== "string" || !stepId) {
    return events;
  }

  const planIndex = [...events]
    .map((item, index) => ({ item, index }))
    .reverse()
    .find(({ item }) => item.event === "plan")?.index;

  if (planIndex === undefined) {
    return events;
  }

  const planEvent = events[planIndex];
  const rawSteps = planEvent.data?.steps;
  if (!Array.isArray(rawSteps)) {
    return events;
  }

  const nextSteps = rawSteps.map((rawStep) => {
    const step = asRecord(rawStep);
    if (String(step.id || "") !== stepId) {
      return step;
    }
    return {
      ...step,
      status: stepEvent.data.status || step.status,
      description: stepEvent.data.description || step.description,
    };
  });

  const nextEvents = [...events];
  nextEvents[planIndex] = {
    ...planEvent,
    data: {
      ...planEvent.data,
      steps: nextSteps,
    },
  };
  return nextEvents;
}

function eventIdOf(event: SessionEventRecord): string | null {
  const eventId = event.data?.event_id;
  if (typeof eventId === "string" && eventId.trim()) {
    return eventId;
  }
  return null;
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
      merged[existingIndex] = event;
      return;
    }
    indexByKey.set(key, merged.length);
    merged.push(event);
  });

  return merged;
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

function isRecoverableLLMErrorEvent(event: SessionEventRecord): boolean {
  if (event.event !== "error") {
    return false;
  }
  const text = String(event.data?.error || "");
  if (!text) {
    return false;
  }
  return (
    text.includes("调用语言模型失败") ||
    text.includes("调用OpenAI客户端向LLM发起请求出错")
  );
}

function hasFollowingRecoveryEvent(
  events: SessionEventRecord[],
  fromIndex: number
): boolean {
  for (let index = fromIndex + 1; index < events.length; index += 1) {
    const event = events[index];
    if (!event) {
      continue;
    }
    if (event.event === "error" || event.event === "done" || event.event === "wait") {
      continue;
    }
    if (event.event === "message") {
      const role = String(event.data?.role || "assistant");
      if (role !== "assistant") {
        continue;
      }
    }
    return true;
  }
  return false;
}

function pruneRecoveredLLMErrors(
  events: SessionEventRecord[]
): SessionEventRecord[] {
  return events.filter((event, index) => {
    if (!isRecoverableLLMErrorEvent(event)) {
      return true;
    }
    return !hasFollowingRecoveryEvent(events, index);
  });
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
        const normalizedEvents = normalizeSessionEvents(session.events as SessionEventRecord[]);
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
            return { currentSession: normalizedRemote };
          }

          const mergedEvents = mergeSessionEvents(
            normalizedRemote.events as SessionEventRecord[],
            localSession.events as SessionEventRecord[]
          );
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

        const normalized = normalizeSessionEvents(recoveredEvents);

        set((s) => {
          const local = s.currentSession;
          if (!local || local.session_id !== sessionId) return {};
          const merged = mergeSessionEvents(
            local.events as SessionEventRecord[],
            normalized
          );
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

registerStoreResetter("session", () => {
  useSessionStore.getState().reset();
});
